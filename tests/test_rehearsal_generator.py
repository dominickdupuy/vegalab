"""Earlier-wave rehearsal generator and curriculum-factory integration."""

from __future__ import annotations

import numpy as np

from optspread.config import GBMConfig
from optspread.market.gbm import GBMGenerator
from optspread.market.gbm_vrp import GBMVRPGenerator
from optspread.market.generator import PriceGenerator
from optspread.market.heston import HestonGenerator
from optspread.market.rehearsal_generator import RehearsalGenerator
from optspread.market.snapshot import REGIME_FEATURE_KEYS, MarketSnapshot
from optspread.training.curriculum_factory import wave_factory


def test_rehearsal_generator_is_deterministic_for_identical_seed() -> None:
    gen_a = RehearsalGenerator(
        primary=GBMGenerator(GBMConfig(n_days=4, sigma=0.20)),
        others=(GBMGenerator(GBMConfig(n_days=4, sigma=0.35)),),
        rehearsal_fraction=0.5,
    )
    gen_b = RehearsalGenerator(
        primary=GBMGenerator(GBMConfig(n_days=4, sigma=0.20)),
        others=(GBMGenerator(GBMConfig(n_days=4, sigma=0.35)),),
        rehearsal_fraction=0.5,
    )

    assert _collect_sequence(gen_a, seed=123, steps=3) == _collect_sequence(
        gen_b, seed=123, steps=3
    )


def test_rehearsal_fraction_zero_matches_standalone_primary() -> None:
    cfg = GBMConfig(n_days=4, sigma=0.20)
    wrapped = RehearsalGenerator(
        primary=GBMGenerator(cfg),
        others=(GBMGenerator(GBMConfig(n_days=4, sigma=0.35)),),
        rehearsal_fraction=0.0,
    )
    standalone = GBMGenerator(cfg)

    assert _collect_sequence(wrapped, seed=77, steps=3) == _collect_sequence(
        standalone, seed=77, steps=3
    )


def test_rehearsal_fraction_one_uses_single_other_generator() -> None:
    gen = RehearsalGenerator(
        primary=GBMGenerator(GBMConfig(n_days=2, sigma=0.20)),
        others=(GBMGenerator(GBMConfig(n_days=2, sigma=0.35)),),
        rehearsal_fraction=1.0,
    )

    for seed in range(10):
        snapshot = gen.reset(np.random.default_rng(seed))
        assert snapshot.regime_features["atm_iv"] == 0.35


def test_rehearsal_fraction_frequency_tracks_configured_rate() -> None:
    n_seeds = 400
    rehearsal_count = 0
    for seed in range(n_seeds):
        gen = RehearsalGenerator(
            primary=GBMGenerator(GBMConfig(n_days=1, sigma=0.20)),
            others=(GBMGenerator(GBMConfig(n_days=1, sigma=0.35)),),
            rehearsal_fraction=0.25,
        )
        snapshot = gen.reset(np.random.default_rng(seed))
        if snapshot.regime_features["atm_iv"] == 0.35:
            rehearsal_count += 1

    assert abs(rehearsal_count / n_seeds - 0.25) <= 0.06


def test_wave_factory_rehearsal_resets_steps_and_varies_selected_wave() -> None:
    factory = wave_factory(1, episode_length=3, rehearsal_fraction=0.3)
    env = factory.make()
    vrp_idx = REGIME_FEATURE_KEYS.index("vrp")
    try:
        obs, _info = env.reset(seed=123)
        assert obs.shape == (factory.obs_dim,)

        terminated = truncated = False
        while not (terminated or truncated):
            obs, _reward, terminated, truncated, _info = env.step(0)
            assert obs.shape == (factory.obs_dim,)

        seen_wave0_like = False
        seen_wave1_like = False
        for seed in range(200, 240):
            obs, _info = env.reset(seed=seed)
            vrp = float(obs[vrp_idx])
            seen_wave0_like = seen_wave0_like or abs(vrp) < 1.0e-8
            seen_wave1_like = seen_wave1_like or abs(vrp) > 1.0e-5

        assert seen_wave0_like
        assert seen_wave1_like
    finally:
        env.close()


def test_rehearsal_generator_snapshots_do_not_expose_hidden_params() -> None:
    gen = RehearsalGenerator(
        primary=GBMVRPGenerator(GBMConfig(n_days=2), sigma=0.16, vrp_vol_premium=0.05),
        others=(HestonGenerator(GBMConfig(n_days=2)),),
        rehearsal_fraction=1.0,
    )
    snapshot = gen.reset(np.random.default_rng(9))
    hidden = {
        "kappa",
        "theta",
        "sigma",
        "sigma_v",
        "rho",
        "v0",
        "variance",
        "v",
        "vrp_vol_premium",
    }

    assert hidden.isdisjoint(snapshot.regime_features)
    assert gen.current_params


def _collect_sequence(
    generator: PriceGenerator, *, seed: int, steps: int
) -> list[tuple[float, tuple[float, ...]]]:
    rng = np.random.default_rng(seed)
    snapshot = generator.reset(rng)
    signatures = [_signature(snapshot)]
    for _ in range(steps):
        signatures.append(_signature(generator.step()))
    return signatures


def _signature(snapshot: MarketSnapshot) -> tuple[float, tuple[float, ...]]:
    return (snapshot.chain.spot, tuple(snapshot.feature_vector()))
