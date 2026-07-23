"""Snapshot/resume support for the distributional trainer (Wave-0 smoke scale)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from optspread.agents.distributional.config import IQNConfig
from optspread.agents.distributional.iqn_agent import IQNAgent
from optspread.agents.distributional.replay import UniformReplayBuffer
from optspread.agents.distributional.risk import RiskMeasure
from optspread.agents.distributional.trainer import DistributionalTrainer
from optspread.training.curriculum_factory import wave_factory
from optspread.training.resumable import restore_snapshot, run_resumable, save_snapshot


def _make(total_timesteps: int) -> tuple[IQNAgent, DistributionalTrainer]:
    config = IQNConfig(
        seed=7,
        total_timesteps=total_timesteps,
        learning_starts=50,
        batch_size=16,
        hidden_sizes=(32,),
        cvar_alpha=0.2,
    )
    factory = wave_factory(0)
    agent = IQNAgent(factory.obs_dim, factory.n_actions, config, risk_measure=RiskMeasure.cvar(0.2))
    trainer = DistributionalTrainer(agent, factory, config, env_seed=123)
    return agent, trainer


def test_replay_state_dict_round_trip() -> None:
    buf = UniformReplayBuffer(8, 3)
    for i in range(5):
        obs = np.full(3, float(i), dtype=np.float32)
        buf.add(obs, i % 2, float(i), obs + 1.0, done=(i == 4))
    clone = UniformReplayBuffer(8, 3)
    clone.load_state_dict(buf.state_dict())
    assert clone.size == buf.size
    np.testing.assert_array_equal(clone.obs[: buf.size], buf.obs[: buf.size])
    np.testing.assert_array_equal(clone.rewards[: buf.size], buf.rewards[: buf.size])


def test_snapshot_restore_recovers_state(tmp_path: Path) -> None:
    agent, trainer = _make(total_timesteps=200)
    run_resumable(trainer, agent, tmp_path, snapshot_every=100, log_every=10_000)
    snap = tmp_path / "state.pt"
    assert snap.exists()

    agent2, trainer2 = _make(total_timesteps=200)
    step, elapsed = restore_snapshot(snap, trainer2, agent2)
    assert step == 200
    assert elapsed > 0.0
    assert trainer2.replay.size == trainer.replay.size
    for p_old, p_new in zip(agent.network.parameters(), agent2.network.parameters(), strict=True):
        assert torch.equal(p_old, p_new)


def test_run_resumes_from_snapshot_and_finishes(tmp_path: Path) -> None:
    agent, trainer = _make(total_timesteps=150)
    run_resumable(trainer, agent, tmp_path, snapshot_every=50, log_every=10_000)

    agent2, trainer2 = _make(total_timesteps=300)
    final = run_resumable(trainer2, agent2, tmp_path, snapshot_every=50, log_every=10_000)
    assert final.exists()
    log_text = (tmp_path / "progress.log").read_text(encoding="utf-8")
    assert "RESUMED from step 150" in log_text
    assert "DONE" in log_text
    restored = IQNAgent.from_checkpoint(final)
    assert restored.obs_dim == agent2.obs_dim


def test_atomic_snapshot_never_leaves_partial_file(tmp_path: Path) -> None:
    agent, trainer = _make(total_timesteps=60)
    save_snapshot(tmp_path / "state.pt", trainer, agent, step=60, elapsed=1.0)
    assert (tmp_path / "state.pt").exists()
    assert not (tmp_path / "state.tmp").exists()
