"""Run the C0 checkpoint diagnostics (quantile spread + decision gap by signal).

Evaluates a saved IQN checkpoint on its training wave and on the Wave-0
no-edge control, plus an untrained-network control on the same states. Takes
zero gradient steps. Optionally writes the numbers as JSON for the gate report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from optspread.agents.distributional.iqn_agent import IQNAgent
from optspread.eval.checkpoint_diagnostics import (
    collect_observations,
    diagnose_checkpoint,
    format_report,
)
from optspread.training.curriculum_factory import wave_factory


def main() -> None:
    parser = argparse.ArgumentParser(description="C0 diagnostics on a saved IQN checkpoint")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--wave", type=int, default=2, help="Wave the checkpoint was trained on.")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--eval-seed-start", type=int, default=70_000)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    agent = IQNAgent.from_checkpoint(args.checkpoint)
    print(f"checkpoint: {args.checkpoint}")
    print(
        f"config: hidden={agent.config.hidden_sizes} "
        f"risk={agent.risk_measure.name} alpha={agent.risk_measure.alpha}"
    )

    wave_obs = collect_observations(
        wave_factory(args.wave), agent, episodes=args.episodes, seed_start=args.eval_seed_start
    )
    control_obs = collect_observations(
        wave_factory(0), agent, episodes=args.episodes, seed_start=args.eval_seed_start + 10_000
    )

    trained_wave = diagnose_checkpoint(agent, wave_obs)
    trained_control = diagnose_checkpoint(agent, control_obs)
    fresh = IQNAgent(agent.obs_dim, agent.n_actions, agent.config, risk_measure=agent.risk_measure)
    fresh.normalizer = agent.normalizer
    untrained_wave = diagnose_checkpoint(fresh, wave_obs)

    print(format_report(f"TRAINED on Wave {args.wave} (edge sim)", trained_wave))
    print(format_report("TRAINED on Wave 0 (no-edge control)", trained_control))
    print(format_report(f"UNTRAINED control on Wave {args.wave}", untrained_wave))

    signal_ordering = (
        trained_wave.mean_scoring.gap_high_signal - trained_wave.mean_scoring.gap_low_signal
    )
    collapse = "point-mass COLLAPSE" if trained_wave.spread_to_value_ratio < 0.05 else "alive"
    print("=== VERDICTS ===")
    print(f"(a) spread/value-scale = {trained_wave.spread_to_value_ratio:.2f} -> {collapse}")
    print(f"(b) mean-gap signal ordering (hi - lo): {signal_ordering:+.5f}")

    if args.json_out is not None:
        args.json_out.write_text(
            json.dumps(
                {
                    "checkpoint": str(args.checkpoint),
                    "wave": args.wave,
                    "episodes": args.episodes,
                    "trained_wave": trained_wave.to_dict(),
                    "trained_no_edge_control": trained_control.to_dict(),
                    "untrained_control_same_states": untrained_wave.to_dict(),
                },
                indent=2,
            )
        )
        print(f"json written: {args.json_out}")


if __name__ == "__main__":
    main()
