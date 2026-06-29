"""The Phase-2 no-edge gate — the definition of done.

On fair-IV, zero-drift Wave 0 every structure is a zero-expectancy martingale
before costs and strictly loss-making after. So a correctly-built environment must
leave a trained agent with NO systematic edge. This module turns that prediction
into a falsifiable pass/fail check on an ``EvalReport``.

Two ways it is used (brief section 5):

* **Part (a), risk-adjusted reward:** FLAT should be strictly preferred (high
  ``flat_threshold``), and mean PnL must not be systematically positive. Call with
  ``flat_threshold`` ~0.8.
* **Part (b), pure-PnL ablation:** no FLAT preference is expected (the agent is
  ~indifferent), only that NO structure prints systematically positive PnL. Call
  with ``flat_threshold=0.0`` so only the profitability check binds.

The decisive failure mode — the one that means "Phase 1 has a leak, STOP" — is a
confidence interval on mean PnL whose LOWER bound is above zero: the agent is
reliably making money where none should exist.
"""

from __future__ import annotations

from dataclasses import dataclass

from optspread.eval.metrics import EvalReport


@dataclass(frozen=True, slots=True)
class NoEdgeResult:
    passed: bool
    flat_frequency: float
    mean_pnl_ci: tuple[float, float]
    reason: str


def evaluate_no_edge(
    report: EvalReport,
    *,
    with_costs: bool,
    flat_threshold: float,
    profit_tol: float = 0.0,
) -> NoEdgeResult:
    """Decide whether ``report`` shows no systematic edge.

    Parameters
    ----------
    with_costs:
        Whether costs were enabled (affects only the explanatory text — the
        profitability bar is the same: an agent must never be reliably positive).
    flat_threshold:
        Minimum FLAT frequency required (use ~0.8 for the risk-adjusted gate,
        0.0 for the pure-PnL ablation where indifference, not flatness, is wanted).
    profit_tol:
        The CI lower bound must not exceed this (default 0.0). A positive lower
        bound is the signature of phantom edge.
    """
    lo, hi = report.pnl_ci
    reasons: list[str] = []

    flat_ok = report.flat_frequency >= flat_threshold
    if not flat_ok:
        reasons.append(
            f"FLAT frequency {report.flat_frequency:.3f} < threshold {flat_threshold:.3f}"
        )

    not_profitable = lo <= profit_tol
    if not not_profitable:
        regime = "with costs" if with_costs else "no costs"
        reasons.append(
            f"mean-PnL 95% CI lower bound {lo:+.2f} > {profit_tol:.2f} ({regime}): "
            "systematic positive edge — investigate Phase 1 (pricing/cost/look-ahead)"
        )

    passed = not reasons
    reason = "no systematic edge detected" if passed else "; ".join(reasons)
    return NoEdgeResult(
        passed=passed,
        flat_frequency=report.flat_frequency,
        mean_pnl_ci=(lo, hi),
        reason=reason,
    )
