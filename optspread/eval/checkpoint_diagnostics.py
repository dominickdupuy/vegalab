"""C0 diagnostics for a trained distributional checkpoint.

Two pre-registered checks motivated by the Wave-2 flat-collapse investigation
(see ``MODEL_CANDIDATES_RESEARCH.md``):

(a) **Quantile-spread check** — is the learned return distribution alive, or
    has it collapsed toward its mean (a published risk of the asymmetric Huber
    quantile loss)? If collapsed, CVaR-of-quantiles action selection is
    degenerate and scoring-side remedies are moot.

(b) **Decision-gap check** — Q(best trade) - Q(FLAT) under mean scoring and
    CVaR scoring, split by the ``iv_rank`` entry signal (BV_2's conditioning
    variable). This is the quantity every candidate intervention is trying to
    move, evaluated on the env the checkpoint was trained for and on the
    no-edge Wave-0 control.

All quantities are computed from fixed-grid tau evaluations of the IQN network;
no gradient steps are taken (frozen-checkpoint discipline).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from optspread.agents.distributional.iqn_agent import IQNAgent
from optspread.market.snapshot import REGIME_FEATURE_KEYS
from optspread.training.env_factory import EnvFactory

IV_RANK_INDEX = REGIME_FEATURE_KEYS.index("iv_rank")
FLAT_ACTION = 0


@dataclass(frozen=True, slots=True)
class GapStats:
    """Decision-gap statistics under one scoring rule."""

    gap_all: float
    gap_high_signal: float
    gap_low_signal: float
    pct_states_trade_preferred: float


@dataclass(frozen=True, slots=True)
class CheckpointDiagnostics:
    """Result of the two C0 checks on one environment."""

    n_states: int
    spread_mean: float
    spread_median: float
    value_scale: float
    spread_to_value_ratio: float
    mean_scoring: GapStats
    cvar_scoring: GapStats
    cvar_alpha: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def collect_observations(
    factory: EnvFactory,
    agent: IQNAgent,
    *,
    episodes: int,
    seed_start: int,
) -> NDArray[np.float32]:
    """Roll the agent deterministically and stack every observation visited."""
    env = factory.make()
    collected: list[NDArray[np.float32]] = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed_start + episode)
        done = False
        while not done:
            collected.append(np.asarray(obs, dtype=np.float32))
            action = agent.act(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
    env.close()
    return np.stack(collected)


def _fixed_tau_quantiles(
    agent: IQNAgent, obs: NDArray[np.float32], taus: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Z(s, a, tau) on a fixed tau grid for every state: (N, n_actions, n_taus)."""
    x = agent.prepare_obs(obs, update=False)
    t = (
        torch.as_tensor(taus, dtype=torch.float32, device=agent.device)
        .unsqueeze(0)
        .expand(x.shape[0], -1)
    )
    with torch.no_grad():
        z = agent.quantiles(x, t, use_target=False)
    return np.asarray(z.cpu().numpy(), dtype=np.float64)


def _gap_stats(
    action_values: NDArray[np.float64],
    high_mask: NDArray[np.bool_],
    low_mask: NDArray[np.bool_],
) -> GapStats:
    gaps = action_values[:, FLAT_ACTION + 1 :].max(axis=1) - action_values[:, FLAT_ACTION]
    return GapStats(
        gap_all=float(gaps.mean()),
        gap_high_signal=float(gaps[high_mask].mean()) if high_mask.any() else float("nan"),
        gap_low_signal=float(gaps[low_mask].mean()) if low_mask.any() else float("nan"),
        pct_states_trade_preferred=float((gaps > 0.0).mean() * 100.0),
    )


def diagnose_checkpoint(
    agent: IQNAgent,
    observations: NDArray[np.float32],
    *,
    n_taus: int = 19,
) -> CheckpointDiagnostics:
    """Run both C0 checks over a set of collected observations."""
    alpha = agent.risk_measure.alpha
    taus = np.linspace(0.05, 0.95, n_taus).astype(np.float64)
    z = _fixed_tau_quantiles(agent, observations, taus)

    i10 = int(np.argmin(np.abs(taus - 0.10)))
    i90 = int(np.argmin(np.abs(taus - 0.90)))
    spread = z[:, :, i90] - z[:, :, i10]
    q_mean = z.mean(axis=2)
    tail = taus <= alpha
    q_cvar = z[:, :, tail].mean(axis=2) if bool(tail.any()) else z[:, :, :1].mean(axis=2)

    signal = observations[:, IV_RANK_INDEX]
    high_mask = signal >= np.quantile(signal, 0.75)
    low_mask = signal <= np.quantile(signal, 0.25)

    value_scale = float(np.abs(q_mean).mean())
    spread_mean = float(spread.mean())
    return CheckpointDiagnostics(
        n_states=int(observations.shape[0]),
        spread_mean=spread_mean,
        spread_median=float(np.median(spread)),
        value_scale=value_scale,
        spread_to_value_ratio=spread_mean / max(value_scale, 1e-12),
        mean_scoring=_gap_stats(q_mean, high_mask, low_mask),
        cvar_scoring=_gap_stats(q_cvar, high_mask, low_mask),
        cvar_alpha=alpha,
    )


def format_report(label: str, diag: CheckpointDiagnostics) -> str:
    """Human-readable block for one environment's diagnostics."""
    m, c = diag.mean_scoring, diag.cvar_scoring
    return (
        f"=== {label} (N={diag.n_states} states) ===\n"
        f"(a) quantile spread q90-q10: mean={diag.spread_mean:+.4f} "
        f"median={diag.spread_median:+.4f} | value scale {diag.value_scale:.4f} "
        f"| spread/value = {diag.spread_to_value_ratio:.2f}\n"
        f"(b) gap Q(best trade)-Q(FLAT):\n"
        f"    mean-scoring: all={m.gap_all:+.5f} hi-signal={m.gap_high_signal:+.5f} "
        f"lo-signal={m.gap_low_signal:+.5f} (trade preferred {m.pct_states_trade_preferred:.1f}%)\n"
        f"    cvar-scoring (a={diag.cvar_alpha:.2f}): hi-signal={c.gap_high_signal:+.5f} "
        f"lo-signal={c.gap_low_signal:+.5f} (trade preferred {c.pct_states_trade_preferred:.1f}%)"
    )
