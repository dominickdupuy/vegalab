"""Assemble standard thesis exhibit placeholders."""

from __future__ import annotations

from optspread.reporting.manifest import Exhibit


def standard_exhibits() -> list[Exhibit]:
    return [
        Exhibit("wave0-no-edge", "Robustness", "Wave-0 No-Edge Gate", "synthetic"),
        Exhibit("wave3-headline", "Results", "Wave-3 Tail Headline", "synthetic"),
        Exhibit("real-walkforward", "Results", "Real Walk-Forward OOS", "real"),
        Exhibit("regime-map", "Results", "Distilled Regime-to-Structure Map", "mixed"),
        Exhibit("cost-frontier", "Robustness", "Cost Sensitivity Frontier", "mixed"),
    ]
