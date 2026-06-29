from __future__ import annotations

import numpy as np
import pytest

from optspread.data.optionmetrics_loader import load_surface_csv
from optspread.data.wrds_optionmetrics import (
    SurfaceQueryConfig,
    annual_tables,
    standardize_surface_records,
    write_surface_csv,
)
from optspread.market.surface import DEFAULT_DELTA_GRID, DEFAULT_MATURITY_GRID_DAYS


def test_annual_tables_expands_inclusive_years() -> None:
    assert annual_tables("vsurfd{year}", start_date="2020-06-01", end_date="2022-01-01") == (
        "vsurfd2020",
        "vsurfd2021",
        "vsurfd2022",
    )


def test_standardize_surface_records_writes_loader_compatible_csv(tmp_path) -> None:
    records: list[dict[str, object]] = []
    for maturity in DEFAULT_MATURITY_GRID_DAYS:
        for delta in DEFAULT_DELTA_GRID:
            records.append(
                {
                    "date": "2020-01-02",
                    "spot": 3200.0,
                    "days": maturity,
                    "delta": delta * 100.0,
                    "impl_volatility": 20.0,
                }
            )

    rows = standardize_surface_records(
        records,
        date_col="date",
        maturity_col="days",
        delta_col="delta",
        iv_col="impl_volatility",
        spot_col="spot",
    )
    path = tmp_path / "surface.csv"
    write_surface_csv(rows, path)

    loaded = load_surface_csv(path)

    assert len(loaded) == 1
    assert loaded[0].date == "2020-01-02"
    assert loaded[0].spot == 3200.0
    assert loaded[0].surface.iv_at_delta_maturity(0.5, 21.0) == pytest.approx(0.20)


def test_surface_query_rejects_bad_identifiers() -> None:
    config = SurfaceQueryConfig(
        library="optionm",
        tables=("vsurfd2020;drop",),
        secid=108105,
        start_date="2020-01-01",
        end_date="2020-12-31",
    )

    with pytest.raises(ValueError):
        # Fetch builds SQL after identifier validation; a fake DB is not reached.
        from optspread.data.wrds_optionmetrics import fetch_surface_records

        fetch_surface_records(object(), config)


def test_standardize_surface_records_interpolates_grid() -> None:
    records = [
        {"date": "2020-01-02", "spot": 100.0, "days": 10.0, "delta": 10.0, "iv": 0.20},
        {"date": "2020-01-02", "spot": 100.0, "days": 10.0, "delta": 90.0, "iv": 0.28},
        {"date": "2020-01-02", "spot": 100.0, "days": 30.0, "delta": 10.0, "iv": 0.24},
        {"date": "2020-01-02", "spot": 100.0, "days": 30.0, "delta": 90.0, "iv": 0.32},
    ]

    rows = standardize_surface_records(
        records,
        date_col="date",
        maturity_col="days",
        delta_col="delta",
        iv_col="iv",
        spot_col="spot",
        target_deltas=np.asarray([0.10, 0.50, 0.90]),
        target_maturities=np.asarray([10.0, 20.0, 30.0]),
    )

    assert rows[0]["iv_20_50"] == pytest.approx(0.26)
