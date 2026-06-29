"""SpreadEnv: the Gymnasium-conformant spread-selection environment.

This is the only mutable trading object besides ``Portfolio`` and the reward
EMAs. It owns NO economic logic and constructs NONE of its dependencies — the
generator, cost model, margin model, reward and observation builder are all
injected (invariant #1). The market path advances ONLY here, inside ``step()``,
via the injected generator (invariant #2).

Step semantics (one trading day):
    1. Apply the chosen action at the CURRENT chain (decision uses only info at
       t): hold if the target structure is already open, otherwise close what is
       held and/or open the new structure, paying transaction costs now.
    2. Advance the market one day (the single path move).
    3. Mark to the new chain; step reward P&L = equity_after_day - equity_before_action
       (market move on the held structure minus the costs paid in step 1).
    4. Build the next observation from the new chain.

With fair-IV GBM this makes every structure a zero-expectancy martingale before
costs, so an always-on agent reads ~0 mean P&L with no costs and < 0 with costs.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from optspread.actions.library import ACTION_LIBRARY, FLAT_ACTION_ID, N_ACTIONS
from optspread.config import EnvConfig
from optspread.costs.cost_model import CostModel
from optspread.envs.observation import ObservationBuilder
from optspread.instruments.leg import OptionLeg
from optspread.margin.margin_model import MarginModel
from optspread.market.generator import PriceGenerator
from optspread.market.snapshot import MarketSnapshot
from optspread.portfolio import pnl
from optspread.portfolio.position import Portfolio, Position
from optspread.reward.composite import CompositeReward
from optspread.reward.context import StepContext
from optspread.rng import make_rng


class SpreadEnv(gym.Env[NDArray[np.float32], np.int64]):
    """Discrete spread-selection env over a single underlying.

    All dependencies are injected; the env wires them together and enforces the
    determinism / no-look-ahead contracts.
    """

    def __init__(
        self,
        *,
        config: EnvConfig,
        generator: PriceGenerator,
        cost_model: CostModel,
        margin_model: MarginModel,
        reward: CompositeReward,
        observation_builder: ObservationBuilder,
    ) -> None:
        super().__init__()
        self.config = config
        self.generator = generator
        self.cost_model = cost_model
        self.margin_model = margin_model
        self.reward = reward
        self.observation_builder = observation_builder

        self.portfolio = Portfolio(multiplier=config.multiplier)
        self._snapshot: MarketSnapshot | None = None
        self._day = 0
        self._equity_high = config.initial_cash  # peak equity, for drawdown

        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_builder.dim,),
            dtype=np.float32,
        )

    # -- gym API ----------------------------------------------------------- #

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        resolved_seed = self.config.seed if seed is None else seed
        # Seed gym's own RNG too, for API conformance, then thread OUR rng.
        super().reset(seed=resolved_seed)
        rng = make_rng(resolved_seed)
        self._snapshot = self.generator.reset(rng)
        self._day = 0
        self.portfolio.reset(self.config.initial_cash)
        self.reward.reset()
        self._equity_high = self.portfolio.equity(self.chain)
        obs = self._observe()
        info: dict[str, Any] = {"day": self._day, "equity": self.portfolio.equity(self.chain)}
        return obs, info

    def step(
        self, action: np.int64 | int
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        if self._snapshot is None:
            raise RuntimeError("step() called before reset()")
        action_id = int(action)
        chain = self.chain

        equity_before = self.portfolio.equity(chain)
        # Settle any position that has reached expiry at intrinsic (no spread cost
        # on settlement) BEFORE the agent acts, so it can re-establish if it wants.
        settled = self._settle_if_expired(chain)
        did_trade, cost = self._apply_action(action_id, chain)
        did_trade = did_trade or settled

        # The single path move.
        self._snapshot = self.generator.step()
        self._day += 1
        new_chain = self.chain

        equity_after = self.portfolio.equity(new_chain)
        step_pnl = equity_after - equity_before  # market move minus costs paid

        ctx = StepContext(
            pnl=step_pnl,
            margin=self.portfolio.margin_used,
            equity=equity_after,
            day=self._day,
            did_trade=did_trade,
        )
        reward = self.reward.update(ctx)
        self._equity_high = max(self._equity_high, equity_after)

        terminated = self.generator.done
        truncated = self._day >= self.config.episode_length and not terminated
        obs = self._observe()
        info: dict[str, Any] = {
            "day": self._day,
            "action_id": action_id,
            "did_trade": did_trade,
            "cost": cost,
            "pnl": step_pnl,
            "margin": self.portfolio.margin_used,
            "equity": equity_after,
            "realized_pnl": self.portfolio.realized_pnl,
            "reward_breakdown": self.reward.last_breakdown,
        }
        return obs, reward, terminated, truncated, info

    # -- helpers ----------------------------------------------------------- #

    def _observe(self) -> NDArray[np.float32]:
        """Build the observation, threading in the env-owned Markov risk state.

        Drawdown (peak-to-current equity) and the reward's exposed EMA state are
        information available at the close, so feeding them keeps the
        risk-adjusted reward optimisable without any look-ahead.
        """
        assert self._snapshot is not None
        drawdown = max(0.0, self._equity_high - self.portfolio.equity(self.chain))
        return self.observation_builder.build(
            self._snapshot,
            self.portfolio,
            self._day,
            drawdown=drawdown,
            risk_state=self.reward.observable_state(),
        )

    @property
    def chain(self) -> Any:
        assert self._snapshot is not None
        return self._snapshot.chain

    def _settle_if_expired(self, chain: Any) -> bool:
        """Close a held position that has reached expiry, marking legs at intrinsic.

        Settlement is exercise/assignment, not a market trade, so it crosses no
        spread (cost 0). Returns whether a settlement occurred.
        """
        if not self.portfolio.has_position:
            return False
        assert self.portfolio.position is not None
        if any(leg.expiry_day >= 0 and leg.expiry_day <= self._day for leg in self._held_legs()):
            self.portfolio.close(chain, cost=0.0)
            return True
        return False

    def _apply_action(self, action_id: int, chain: Any) -> tuple[bool, float]:
        """Reconcile the held position to the action's target. Returns (traded?, cost)."""
        # FLAT: close anything held, then hold no position.
        if action_id == FLAT_ACTION_ID:
            if self.portfolio.has_position:
                cost = self.cost_model.cost(self._held_legs(), chain)
                self.portfolio.close(chain, cost)
                return True, cost
            return False, 0.0

        # Already holding exactly this structure: do nothing (no churn cost).
        if self.portfolio.has_position:
            assert self.portfolio.position is not None
            if self.portfolio.position.action_id == action_id:
                return False, 0.0

        # Otherwise: close the old structure (if any), open the new one.
        total_cost = 0.0
        if self.portfolio.has_position:
            close_cost = self.cost_model.cost(self._held_legs(), chain)
            self.portfolio.close(chain, close_cost)
            total_cost += close_cost

        legs = self._build_legs(action_id, chain)
        open_cost = self.cost_model.cost(legs, chain)
        total_cost += open_cost
        margin = self._margin_for(legs, action_id, chain)
        self.portfolio.open(legs, action_id, margin, self._day, open_cost)
        return True, total_cost

    def _build_legs(self, action_id: int, chain: Any) -> list[OptionLeg]:
        spec = ACTION_LIBRARY[action_id]
        legs = spec.template.build(chain, spec.delta_bucket)
        # Scale by contracts_per_trade while preserving each leg's qty sign/ratio.
        n = self.config.contracts_per_trade
        if n == 1:
            return legs
        return [
            OptionLeg(leg.right, leg.strike, leg.expiry_idx, leg.qty * n, leg.entry_price)
            for leg in legs
        ]

    def _margin_for(self, legs: list[OptionLeg], action_id: int, chain: Any) -> float:
        provisional = Position(
            legs=tuple(legs),
            action_id=action_id,
            margin=0.0,
            open_day=self._day,
            entry_cash_flow=pnl.opening_cash_flow(legs, self.config.multiplier),
        )
        return self.margin_model.margin(provisional, chain)

    def _held_legs(self) -> tuple[OptionLeg, ...]:
        assert self.portfolio.position is not None
        return self.portfolio.position.legs
