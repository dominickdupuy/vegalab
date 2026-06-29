"""Wave-0 economic sanity gate (see CLAUDE.md).

Runs the scripted baselines over many independent GBM paths and reports mean
P&L per episode in two cost regimes. The fair-IV generator prices the chain at
the same sigma that drives the path, so the expectations are pinned:

    NO COSTS:   every structure has ~zero expectancy. The always-on agent's mean
                P&L is ~0 within Monte-Carlo noise.
    WITH COSTS: every structure has negative expectancy and FLAT dominates. The
                always-on agent's mean P&L is < 0, and the random agent (which
                churns the book) bleeds faster.

If the always-on agent MAKES money with fair IV and no costs, that is a BUG
(pricing inconsistency, cost sign error, or look-ahead) — not a discovery.

Run: ``python -m optspread.cli.smoke_run --episodes 300``
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from optspread.agents.baselines import Agent, AlwaysOnAgent, FlatAgent, RandomAgent
from optspread.config import CostConfig
from optspread.envs.builder import EnvBundle, build_default_env


@dataclass(frozen=True, slots=True)
class Stats:
    mean: float
    std: float
    stderr: float
    n: int


def run_episode(bundle: EnvBundle, agent: Agent, seed: int) -> float:
    """Run one episode and return total P&L (equity change over the episode)."""
    env = build_default_env(bundle)
    obs, _ = env.reset(seed=seed)
    agent.reset()
    total = 0.0
    terminated = truncated = False
    while not (terminated or truncated):
        action = agent.act(obs)
        obs, _, terminated, truncated, info = env.step(action)
        total += float(info["pnl"])
    return total


def evaluate(
    bundle: EnvBundle,
    make_agent: Callable[[int], Agent],
    episodes: int,
    base_seed: int = 10_000,
) -> Stats:
    pnls = np.array(
        [run_episode(bundle, make_agent(base_seed + i), base_seed + i) for i in range(episodes)]
    )
    n = len(pnls)
    std = float(pnls.std(ddof=1)) if n > 1 else 0.0
    return Stats(mean=float(pnls.mean()), std=std, stderr=std / np.sqrt(n), n=n)


def _no_cost_bundle() -> EnvBundle:
    return EnvBundle(cost=CostConfig(half_spread_bps=0.0, min_cost_per_leg=0.0))


def _with_cost_bundle() -> EnvBundle:
    return EnvBundle()  # default Wave-0 costs


def _fmt(s: Stats) -> str:
    return f"{s.mean:+10.2f}  ± {s.stderr:7.2f}  (sd {s.std:8.2f}, n={s.n})"


def main() -> None:
    parser = argparse.ArgumentParser(description="Wave-0 economic sanity gate")
    parser.add_argument("--episodes", type=int, default=300)
    args = parser.parse_args()
    eps = args.episodes

    agents: dict[str, Callable[[int], Agent]] = {
        "FLAT": lambda _seed: FlatAgent(),
        "ALWAYS-ON (strangle)": lambda _seed: AlwaysOnAgent(action_id=14),
        "RANDOM": lambda seed: RandomAgent(seed=seed),
    }

    no_cost = _no_cost_bundle()
    with_cost = _with_cost_bundle()

    print(f"\nWave-0 sanity gate — {eps} episodes per cell, fair-IV GBM\n")
    header = f"{'agent':24s}{'no costs':>34s}{'with costs':>34s}"
    print(header)
    print("-" * len(header))

    results: dict[str, tuple[Stats, Stats]] = {}
    for label, make in agents.items():
        nc = evaluate(no_cost, make, eps)
        wc = evaluate(with_cost, make, eps)
        results[label] = (nc, wc)
        print(f"{label:24s}{_fmt(nc):>34s}{_fmt(wc):>34s}")

    print()
    _report_gate(results)


def _report_gate(results: dict[str, tuple[Stats, Stats]]) -> None:
    nc_on, wc_on = results["ALWAYS-ON (strangle)"]
    _, wc_rand = results["RANDOM"]

    # 1) No-cost always-on mean P&L is ~0 (within ~3 standard errors).
    zero_ok = abs(nc_on.mean) <= 3.0 * max(nc_on.stderr, 1e-9)
    # 2) With costs, always-on bleeds (mean < 0).
    bleed_ok = wc_on.mean < 0.0
    # 3) Random churns costs faster than always-on (more negative mean).
    churn_ok = wc_rand.mean < wc_on.mean

    print("Sanity gate:")
    print(
        f"  [{'PASS' if zero_ok else 'FAIL'}] no-cost always-on ~ 0   "
        f"(mean {nc_on.mean:+.2f}, |mean| <= 3*se {3 * nc_on.stderr:.2f})"
    )
    print(f"  [{'PASS' if bleed_ok else 'FAIL'}] with-cost always-on < 0 (mean {wc_on.mean:+.2f})")
    print(
        f"  [{'PASS' if churn_ok else 'FAIL'}] random bleeds faster    "
        f"(random {wc_rand.mean:+.2f} < always-on {wc_on.mean:+.2f})"
    )
    if zero_ok and bleed_ok and churn_ok:
        print("\n  Wave-0 expectation HOLDS. [OK]")
    else:
        print("\n  Wave-0 expectation VIOLATED - investigate pricing/cost/look-ahead. [X]")


if __name__ == "__main__":
    main()
