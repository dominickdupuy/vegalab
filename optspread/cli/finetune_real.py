"""Light fine-tuning of a synthetic distributional agent on TRAIN-only real windows."""

from __future__ import annotations

import argparse
from pathlib import Path

from optspread.agents.distributional.config import DistributionalConfig, IQNConfig, QRDQNConfig
from optspread.agents.distributional.iqn_agent import IQNAgent
from optspread.agents.distributional.qrdqn_agent import QRDQNAgent
from optspread.agents.distributional.risk import RiskMeasure
from optspread.agents.distributional.trainer import DistributionalTrainer
from optspread.config import CostConfig, EnvConfig, GBMConfig
from optspread.data.optionmetrics_loader import SurfaceRow, load_surface_csv
from optspread.data.real_generator import RealDataReplay
from optspread.envs.builder import EnvBundle
from optspread.training.env_factory import EnvFactory
from optspread.training.logging import MetricLogger
from optspread.training.phase2 import curriculum_reward

DistributionalFineTuneAgent = IQNAgent | QRDQNAgent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune a synthetic Phase-5 distributional checkpoint on real TRAIN rows."
    )
    parser.add_argument(
        "--surface-csv", type=Path, default=Path("data/optionmetrics_spx_surface.csv")
    )
    parser.add_argument("--in-checkpoint", type=Path, required=True)
    parser.add_argument("--out-checkpoint", type=Path, required=True)
    parser.add_argument("--algo", choices=("iqn", "qrdqn"), default="iqn")
    parser.add_argument("--cvar-alpha", type=float, default=0.1)
    parser.add_argument("--train-frac", type=float, default=0.6)
    parser.add_argument("--episode-length", type=int, default=63)
    parser.add_argument("--warmup", type=int, default=21)
    parser.add_argument("--total-timesteps", type=int, default=15_000)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--learning-starts", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=5_000)
    parser.add_argument("--no-tensorboard", action="store_true")
    args = parser.parse_args()

    _validate_args(args)
    rows = load_surface_csv(args.surface_csv)
    split_idx = int(len(rows) * args.train_frac)
    train_rows = rows[:split_idx]
    _validate_split(rows, train_rows, split_idx, args.episode_length, args.warmup)
    print(f"train rows: rows[:{split_idx}] {train_rows[0].date} -> {train_rows[-1].date}")
    print(f"reserved test rows: rows[{split_idx}:] ({len(rows) - split_idx} rows; unused here)")

    factory = _build_factory(train_rows, args.episode_length, args.warmup)
    agent, cfg = _load_finetune_agent(
        args.algo,
        args.in_checkpoint,
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        learning_starts=args.learning_starts,
        seed=args.seed,
        cvar_alpha=args.cvar_alpha,
    )

    args.out_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    logger = (
        None
        if args.no_tensorboard
        else MetricLogger(args.out_checkpoint.parent / "tb" / args.out_checkpoint.stem)
    )
    trainer = DistributionalTrainer(agent, factory, cfg, logger=logger, env_seed=args.seed)
    try:
        trainer.train()
    finally:
        if logger is not None:
            logger.close()
    agent.save(args.out_checkpoint)
    print(f"FINETUNE_REAL_OK {args.out_checkpoint}")


def _validate_args(args: argparse.Namespace) -> None:
    if not 0.0 < args.train_frac < 1.0:
        raise SystemExit("--train-frac must be in (0, 1)")
    if args.episode_length < 2:
        raise SystemExit("--episode-length must be at least 2")
    if args.warmup < 0:
        raise SystemExit("--warmup must be non-negative")
    if args.total_timesteps <= 0:
        raise SystemExit("--total-timesteps must be positive")
    if args.learning_starts < 0:
        raise SystemExit("--learning-starts must be non-negative")
    if args.learning_rate <= 0.0:
        raise SystemExit("--learning-rate must be positive")
    if not 0.0 < args.cvar_alpha <= 1.0:
        raise SystemExit("--cvar-alpha must be in (0, 1]")


def _validate_split(
    rows: list[SurfaceRow],
    train_rows: list[SurfaceRow],
    split_idx: int,
    episode_length: int,
    warmup: int,
) -> None:
    if split_idx <= 0 or split_idx >= len(rows):
        raise SystemExit("--train-frac must leave non-empty train and reserved test spans")
    required_train_rows = warmup + episode_length
    if len(train_rows) < required_train_rows:
        raise SystemExit(
            "training span is too short for "
            f"--warmup {warmup} plus --episode-length {episode_length}"
        )


def _build_factory(
    train_rows: list[SurfaceRow],
    episode_length: int,
    warmup: int,
) -> EnvFactory:
    gbm = GBMConfig(n_days=episode_length)

    def generator_factory() -> RealDataReplay:
        return RealDataReplay(
            train_rows,
            gbm,
            warmup_rows=warmup,
            sample_window=episode_length,
        )

    return EnvFactory(
        EnvBundle(
            env=EnvConfig(episode_length=episode_length),
            gbm=gbm,
            cost=CostConfig(),
            reward=curriculum_reward(),
            generator_factory=generator_factory,
        )
    )


def _load_finetune_agent(
    algo: str,
    checkpoint: Path,
    *,
    total_timesteps: int,
    learning_rate: float,
    learning_starts: int,
    seed: int,
    cvar_alpha: float,
) -> tuple[DistributionalFineTuneAgent, DistributionalConfig]:
    if algo == "iqn":
        return _load_iqn_finetune_agent(
            checkpoint,
            total_timesteps=total_timesteps,
            learning_rate=learning_rate,
            learning_starts=learning_starts,
            seed=seed,
            cvar_alpha=cvar_alpha,
        )
    if algo == "qrdqn":
        return _load_qrdqn_finetune_agent(
            checkpoint,
            total_timesteps=total_timesteps,
            learning_rate=learning_rate,
            learning_starts=learning_starts,
            seed=seed,
            cvar_alpha=cvar_alpha,
        )
    raise ValueError(f"unsupported algo: {algo}")


def _load_iqn_finetune_agent(
    checkpoint: Path,
    *,
    total_timesteps: int,
    learning_rate: float,
    learning_starts: int,
    seed: int,
    cvar_alpha: float,
) -> tuple[IQNAgent, IQNConfig]:
    risk = RiskMeasure.cvar(cvar_alpha)
    base_agent = IQNAgent.from_checkpoint(checkpoint)
    _require_agent_side_tail_risk(base_agent.config)
    cfg = _finetune_iqn_config(
        base_agent.config,
        total_timesteps=total_timesteps,
        learning_rate=learning_rate,
        learning_starts=learning_starts,
        seed=seed,
    )
    agent = IQNAgent(base_agent.obs_dim, base_agent.n_actions, cfg, risk_measure=risk)
    agent.load(checkpoint)
    agent.risk_measure = risk
    return agent, cfg


def _load_qrdqn_finetune_agent(
    checkpoint: Path,
    *,
    total_timesteps: int,
    learning_rate: float,
    learning_starts: int,
    seed: int,
    cvar_alpha: float,
) -> tuple[QRDQNAgent, QRDQNConfig]:
    risk = RiskMeasure.cvar(cvar_alpha)
    base_agent = QRDQNAgent.from_checkpoint(checkpoint)
    _require_agent_side_tail_risk(base_agent.config)
    cfg = _finetune_qrdqn_config(
        base_agent.config,
        total_timesteps=total_timesteps,
        learning_rate=learning_rate,
        learning_starts=learning_starts,
        seed=seed,
    )
    agent = QRDQNAgent(base_agent.obs_dim, base_agent.n_actions, cfg, risk_measure=risk)
    agent.load(checkpoint)
    agent.risk_measure = risk
    return agent, cfg


def _require_agent_side_tail_risk(config: DistributionalConfig) -> None:
    if config.bootstrap_risk != "mean":
        raise SystemExit("checkpoint config must use bootstrap_risk='mean'")


def _finetune_iqn_config(
    config: IQNConfig,
    *,
    total_timesteps: int,
    learning_rate: float,
    learning_starts: int,
    seed: int,
) -> IQNConfig:
    return config.model_copy(
        update={
            "total_timesteps": total_timesteps,
            "learning_rate": learning_rate,
            "learning_starts": learning_starts,
            "seed": seed,
        }
    )


def _finetune_qrdqn_config(
    config: QRDQNConfig,
    *,
    total_timesteps: int,
    learning_rate: float,
    learning_starts: int,
    seed: int,
) -> QRDQNConfig:
    return config.model_copy(
        update={
            "total_timesteps": total_timesteps,
            "learning_rate": learning_rate,
            "learning_starts": learning_starts,
            "seed": seed,
        }
    )


if __name__ == "__main__":
    main()
