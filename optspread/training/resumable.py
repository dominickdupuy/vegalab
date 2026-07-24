"""Resumable training loop for the distributional trainer.

Long Heston-wave runs (hours of wall clock) need two properties the plain
``DistributionalTrainer.train`` loop does not provide: visible progress and
crash recovery. This module drives the trainer's own action-selection and
gradient-step machinery (the algorithm is unchanged) while adding periodic
full-state snapshots — networks, optimizer, replay buffer, step counter, and
both RNG streams — plus a streaming ``progress.log``.

Resume semantics: everything is restored except the in-flight episode, which
restarts at a fresh seed drawn from the restored numpy stream. A resumed run
is therefore scientifically equivalent but not byte-identical to an
uninterrupted one; gate-grade headline runs should note when a resume occurred
(the progress log records it).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from optspread.agents.distributional.iqn_agent import IQNAgent
from optspread.agents.distributional.qrdqn_agent import QRDQNAgent
from optspread.agents.distributional.trainer import DistributionalTrainer
from optspread.training.seeding import seed_everything

DistributionalValueAgent = IQNAgent | QRDQNAgent

SNAPSHOT_NAME = "state.pt"
PROGRESS_NAME = "progress.log"
STATUS_NAME = "status.json"
FINAL_AGENT_NAME = "agent.pt"


def save_snapshot(
    path: Path,
    trainer: DistributionalTrainer,
    agent: DistributionalValueAgent,
    *,
    step: int,
    elapsed: float,
) -> None:
    """Atomically write a full training snapshot (write temp file, then replace)."""
    payload: dict[str, Any] = {
        "step": step,
        "elapsed": elapsed,
        "agent": {
            "network": agent.network.state_dict(),
            "target_network": agent.target_network.state_dict(),
            "normalizer": agent.normalizer.state_dict() if agent.normalizer else None,
        },
        "optimizer": trainer.optimizer.state_dict(),
        "replay": trainer.replay.state_dict(),
        "np_rng": trainer.rng.bit_generator.state,
        "torch_rng": torch.get_rng_state(),
    }
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def restore_snapshot(
    path: Path,
    trainer: DistributionalTrainer,
    agent: DistributionalValueAgent,
) -> tuple[int, float]:
    """Restore a snapshot written by :func:`save_snapshot`; returns (step, elapsed)."""
    payload = torch.load(path, map_location=agent.device, weights_only=False)
    agent.network.load_state_dict(payload["agent"]["network"])
    agent.target_network.load_state_dict(payload["agent"]["target_network"])
    if agent.normalizer is not None and payload["agent"]["normalizer"] is not None:
        agent.normalizer.load_state_dict(payload["agent"]["normalizer"])
    trainer.optimizer.load_state_dict(payload["optimizer"])
    trainer.replay.load_state_dict(payload["replay"])
    trainer.rng.bit_generator.state = payload["np_rng"]
    torch.set_rng_state(payload["torch_rng"])
    return int(payload["step"]), float(payload["elapsed"])


def run_resumable(
    trainer: DistributionalTrainer,
    agent: DistributionalValueAgent,
    run_dir: Path,
    *,
    snapshot_every: int = 5_000,
    log_every: int = 500,
) -> Path:
    """Train to ``trainer.config.total_timesteps`` with snapshots and progress.

    If ``run_dir/state.pt`` exists the run resumes from it; otherwise it starts
    fresh (seeding exactly as the plain trainer would). Returns the path of the
    final agent checkpoint.
    """
    cfg = trainer.config
    run_dir.mkdir(parents=True, exist_ok=True)
    snap_path = run_dir / SNAPSHOT_NAME

    start_step, prior_elapsed = 0, 0.0
    if snap_path.exists():
        start_step, prior_elapsed = restore_snapshot(snap_path, trainer, agent)
        mode = f"RESUMED from step {start_step}"
    else:
        seed_everything(cfg.seed)
        mode = "FRESH start"

    with open(run_dir / PROGRESS_NAME, "a", encoding="utf-8") as log:

        def say(msg: str) -> None:
            line = f"[{time.strftime('%H:%M:%S')}] {msg}"
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()

        say(
            f"{mode} | target {cfg.total_timesteps} steps | snapshot every {snapshot_every}"
            f" | seed {cfg.seed}"
        )

        env = trainer.env_factory.make()
        reset_seed = (
            trainer.env_seed if start_step == 0 else int(trainer.rng.integers(0, 2**31 - 1))
        )
        obs, _ = env.reset(seed=reset_seed)
        agent.prepare_obs(obs.reshape(1, -1), update=True)
        episode_reward = 0.0
        recent: list[float] = []
        last_loss = 0.0
        t0 = time.time()

        for step in range(start_step + 1, cfg.total_timesteps + 1):
            epsilon = trainer._epsilon(step)
            action = trainer._select_behavior_action(
                obs, epsilon, risk=trainer.behavior_risk_at(step)
            )
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.prepare_obs(next_obs.reshape(1, -1), update=True)
            trainer.replay.add(obs, action, float(reward), next_obs, done)
            episode_reward += float(reward)
            obs = next_obs
            if done:
                recent.append(episode_reward)
                recent = recent[-30:]
                episode_reward = 0.0
                obs, _ = env.reset(seed=int(trainer.rng.integers(0, 2**31 - 1)))
                agent.prepare_obs(obs.reshape(1, -1), update=True)

            if step >= cfg.learning_starts and step % cfg.train_freq == 0:
                for _ in range(cfg.gradient_steps):
                    last_loss, _ = trainer._gradient_step()
            if step % cfg.target_update_interval == 0:
                agent.sync_target()

            if step % log_every == 0:
                elapsed = prior_elapsed + (time.time() - t0)
                rate = (step - start_step) / max(time.time() - t0, 1e-9)
                eta = (cfg.total_timesteps - step) / max(rate, 1e-9)
                mean_recent = float(np.mean(recent)) if recent else float("nan")
                say(
                    f"step {step}/{cfg.total_timesteps} "
                    f"({100 * step / cfg.total_timesteps:5.1f}%) loss={last_loss:8.4f} "
                    f"eps={epsilon:.3f} ep_rew(30)={mean_recent:+8.3f} {rate:5.1f} st/s "
                    f"ETA {int(eta // 3600)}h{int(eta % 3600 // 60):02d}m "
                    f"elapsed {int(elapsed // 3600)}h{int(elapsed % 3600 // 60):02d}m"
                )
            if step % snapshot_every == 0:
                save_snapshot(
                    snap_path,
                    trainer,
                    agent,
                    step=step,
                    elapsed=prior_elapsed + (time.time() - t0),
                )
                (run_dir / STATUS_NAME).write_text(
                    json.dumps(
                        {
                            "step": step,
                            "of": cfg.total_timesteps,
                            "pct": round(100 * step / cfg.total_timesteps, 1),
                        }
                    )
                )
                say(f"  snapshot saved @ {step}")

        final_path = run_dir / FINAL_AGENT_NAME
        agent.save(final_path)
        say(f"DONE. final checkpoint: {final_path}")
        env.close()
    return final_path
