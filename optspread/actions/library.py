"""The locked, ordered action library.

ORDERING CONTRACT (see CLAUDE.md): ``ACTION_LIBRARY`` is an ordered list whose
index IS the ``action_id``. Action 0 is always FLAT. Entries are append-only;
never reorder or delete, or saved policies and diagnostics become meaningless.

Each non-flat entry pairs a ``SpreadTemplate`` with a delta bucket. The library
spans {directional vs neutral} x {credit vs debit} x {defined vs undefined risk}
x {single vs multi-expiry}, with a total of 19 actions (within the 12-20 range).
"""

from __future__ import annotations

from dataclasses import dataclass

from optspread.actions.margin_class import MarginClass
from optspread.actions.templates import (
    BearCallSpreadTemplate,
    BearPutSpreadTemplate,
    BullCallSpreadTemplate,
    BullPutSpreadTemplate,
    CalendarSpreadTemplate,
    FlatTemplate,
    IronButterflyTemplate,
    IronCondorTemplate,
    LongCallTemplate,
    LongPutTemplate,
    RatioCallSpreadTemplate,
    ShortStraddleTemplate,
    ShortStrangleTemplate,
    SpreadTemplate,
)

DELTA_BUCKETS: tuple[float, ...] = (0.10, 0.16, 0.25, 0.40)

# Wing width in grid steps, shared by all defined-risk wing structures.
WING_STRIKES = 5


@dataclass(frozen=True, slots=True)
class ActionSpec:
    """One entry of the action library: a template + the bucket it trades at.

    ``delta_bucket`` is ``None`` only for FLAT (which takes no bucket) and for
    ATM-anchored structures where the bucket is nominal; in those cases the
    template ignores it during ``build``.
    """

    template: SpreadTemplate
    delta_bucket: float

    @property
    def name(self) -> str:
        return f"{self.template.name}@{self.delta_bucket:.2f}"

    def margin_class(self) -> MarginClass:
        return self.template.margin_class()


# Singleton template instances (stateless aside from wing config).
_flat = FlatTemplate()
_long_call = LongCallTemplate()
_long_put = LongPutTemplate()
_bull_put = BullPutSpreadTemplate(WING_STRIKES)
_bear_call = BearCallSpreadTemplate(WING_STRIKES)
_bull_call = BullCallSpreadTemplate(WING_STRIKES)
_bear_put = BearPutSpreadTemplate(WING_STRIKES)
_condor = IronCondorTemplate(WING_STRIKES)
_butterfly = IronButterflyTemplate(WING_STRIKES)
_strangle = ShortStrangleTemplate()
_straddle = ShortStraddleTemplate()
_calendar = CalendarSpreadTemplate()
_ratio = RatioCallSpreadTemplate(WING_STRIKES)


# action_id 0 is FLAT. A nominal bucket (0.40) is recorded but FLAT ignores it.
ACTION_LIBRARY: tuple[ActionSpec, ...] = (
    ActionSpec(_flat, 0.40),  # 0  FLAT (null action)
    ActionSpec(_long_call, 0.25),  # 1  directional debit
    ActionSpec(_long_call, 0.40),  # 2
    ActionSpec(_long_put, 0.25),  # 3  directional debit
    ActionSpec(_long_put, 0.40),  # 4
    ActionSpec(_bull_put, 0.16),  # 5  credit, defined
    ActionSpec(_bull_put, 0.25),  # 6
    ActionSpec(_bear_call, 0.16),  # 7  credit, defined
    ActionSpec(_bear_call, 0.25),  # 8
    ActionSpec(_bull_call, 0.40),  # 9  debit, defined
    ActionSpec(_bear_put, 0.40),  # 10 debit, defined
    ActionSpec(_condor, 0.10),  # 11 neutral credit, defined
    ActionSpec(_condor, 0.16),  # 12
    ActionSpec(_butterfly, 0.40),  # 13 neutral credit, defined (ATM body)
    ActionSpec(_strangle, 0.10),  # 14 neutral credit, UNDEFINED
    ActionSpec(_strangle, 0.16),  # 15
    ActionSpec(_straddle, 0.40),  # 16 neutral credit, UNDEFINED (ATM)
    ActionSpec(_calendar, 0.25),  # 17 two-expiry, defined
    ActionSpec(_ratio, 0.25),  # 18 ratio, UNDEFINED
)

N_ACTIONS = len(ACTION_LIBRARY)
FLAT_ACTION_ID = 0

assert 12 <= N_ACTIONS <= 20, f"action count {N_ACTIONS} outside locked 12-20 range"
assert ACTION_LIBRARY[FLAT_ACTION_ID].template.margin_class() is MarginClass.FLAT
