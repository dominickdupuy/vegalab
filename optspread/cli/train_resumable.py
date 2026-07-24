"""Train a distributional agent with progress logging and crash-resumable snapshots.

Same algorithm and recipe surface as ``train_distributional`` (the loop drives
``DistributionalTrainer``'s own machinery); adds ``progress.log`` streaming and
periodic full-state snapshots so a killed run restarts from its last snapshot
by re-running the identical command. See ``optspread.training.resumable``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from optspread.agents.distributional.config import IQNConfig
from optspread.agents.distributional.iqn_agent import IQNAgent
from optspread.agents.distributional.risk import RiskMeasure
from optspread.agents.distributional.trainer import DistributionalTrainer
from optspread.training.curriculum_factory import wave_factory
from optspread.training.resumable import run_resumable


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable IQN training with progress log")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--wave", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--env-seed", type=int, default=30_000)
    parser.add_argument("--total-timesteps", type=int, default=150_000)
    parser.add_argument("--cvar-alpha", type=float, default=0.2)
    parser.add_argument("--epsilon-end", type=float, default=0.04)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--snapshot-every", type=int, default=5_000)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument(
        "--seek-start",
        type=float,
        default=None,
        help="Enable risk-seeking behavior annealing: start band U(1-x, 1) -> U(0, 1).",
    )
    parser.add_argument("--seek-decay-steps", type=int, default=None)
    args = parser.parse_args()

    config = IQNConfig(
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        cvar_alpha=args.cvar_alpha,
        epsilon_end=args.epsilon_end,
        hidden_sizes=(args.hidden_size,),
        behavior_seek_start=args.seek_start,
        behavior_seek_decay_steps=args.seek_decay_steps,
    )
    factory = wave_factory(args.wave)
    risk = RiskMeasure.cvar(config.cvar_alpha)
    agent = IQNAgent(factory.obs_dim, factory.n_actions, config, risk_measure=risk)
    trainer = DistributionalTrainer(agent, factory, config, env_seed=args.env_seed)

    final = run_resumable(
        trainer,
        agent,
        args.run_dir,
        snapshot_every=args.snapshot_every,
        log_every=args.log_every,
    )
    print(f"checkpoint: {final}")


if __name__ == "__main__":
    main()
