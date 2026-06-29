"""Walk-forward splitter with purge and embargo."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Fold:
    train_start: int
    train_end: int
    test_start: int
    test_end: int

    @property
    def train_indices(self) -> range:
        return range(self.train_start, self.train_end)

    @property
    def test_indices(self) -> range:
        return range(self.test_start, self.test_end)


class WalkForwardSplitter:
    """Expanding-window folds with explicit purge and embargo gaps."""

    def __init__(
        self, *, train_size: int, test_size: int, purge: int = 0, embargo: int = 0
    ) -> None:
        if train_size <= 0 or test_size <= 0:
            raise ValueError("train_size and test_size must be positive")
        if purge < 0 or embargo < 0:
            raise ValueError("purge and embargo must be non-negative")
        self.train_size = train_size
        self.test_size = test_size
        self.purge = purge
        self.embargo = embargo

    def split(self, n_samples: int) -> list[Fold]:
        folds: list[Fold] = []
        test_start = self.train_size + self.purge
        while test_start + self.test_size <= n_samples:
            train_end = test_start - self.purge
            folds.append(
                Fold(
                    train_start=0,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_start + self.test_size,
                )
            )
            test_start += self.test_size + self.embargo
        return folds
