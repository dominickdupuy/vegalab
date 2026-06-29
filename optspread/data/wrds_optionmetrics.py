"""WRDS OptionMetrics extraction helpers.

The WRDS table layout can differ by subscription (`optionm`, `optionm_all`,
sample libraries, and annual partitions), so this module keeps the extraction
schema configurable while writing one stable project artifact:

    date, spot, iv_<maturity_days>_<delta_pct>, ...

That is the CSV consumed by `load_surface_csv` and `RealDataReplay`.
"""

from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import numpy as np

from optspread.market.surface import DEFAULT_DELTA_GRID, DEFAULT_MATURITY_GRID_DAYS

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True, slots=True)
class SurfaceQueryConfig:
    library: str
    tables: tuple[str, ...]
    secid: int
    start_date: str
    end_date: str
    date_col: str = "date"
    secid_col: str = "secid"
    maturity_col: str = "days"
    delta_col: str = "delta"
    iv_col: str = "impl_volatility"
    spot_col: str | None = "spot"
    # OptionMetrics `vsurfd` stores both call and put rows; the signed delta of a
    # call (e.g. +0.90 -> low strike) and a put (-0.90 -> high strike) at the same
    # |delta| describe opposite strike wings, so they must NOT be averaged. Filter
    # to one option type to obtain a single, monotonic smile across strikes.
    cp_flag_col: str | None = "cp_flag"
    cp_flag: str | None = "C"


@dataclass(frozen=True, slots=True)
class SpotQueryConfig:
    library: str
    tables: tuple[str, ...]
    secid: int
    start_date: str
    end_date: str
    date_col: str = "date"
    secid_col: str = "secid"
    spot_col: str = "close"


def annual_tables(pattern: str, *, start_date: str, end_date: str) -> tuple[str, ...]:
    """Expand a table pattern like `vsurfd{year}` across inclusive years."""
    start_year = int(start_date[:4])
    end_year = int(end_date[:4])
    if end_year < start_year:
        raise ValueError("end_date must be >= start_date")
    return tuple(pattern.format(year=year) for year in range(start_year, end_year + 1))


def describe_candidate_tables(
    db: Any,
    libraries: tuple[str, ...],
    *,
    keywords: tuple[str, ...] = ("surf", "vol", "opprc", "secpr", "security"),
) -> list[dict[str, object]]:
    """Return table/column metadata for likely OptionMetrics tables."""
    candidates: list[dict[str, object]] = []
    lowered_keywords = tuple(k.lower() for k in keywords)
    for library in libraries:
        for table in db.list_tables(library=library):
            table_name = str(table)
            if not any(keyword in table_name.lower() for keyword in lowered_keywords):
                continue
            columns = describe_columns(db, library, table_name)
            candidates.append({"library": library, "table": table_name, "columns": columns})
    return candidates


def describe_columns(db: Any, library: str, table: str) -> list[str]:
    """List column names for a WRDS table using `describe_table`."""
    description = db.describe_table(library=library, table=table)
    if hasattr(description, "__getitem__") and "name" in getattr(description, "columns", []):
        names = description["name"].astype(str).tolist()
        return [str(name) for name in names]
    return [str(column) for column in getattr(description, "columns", [])]


def fetch_surface_records(db: Any, config: SurfaceQueryConfig) -> list[dict[str, object]]:
    """Fetch long-form surface records from one or more WRDS tables."""
    columns = [
        config.date_col,
        config.secid_col,
        config.maturity_col,
        config.delta_col,
        config.iv_col,
    ]
    if config.spot_col is not None:
        columns.append(config.spot_col)
    query = _surface_sql(config, columns)
    params: dict[str, object] = {
        "secid": config.secid,
        "start_date": config.start_date,
        "end_date": config.end_date,
    }
    if config.cp_flag_col is not None and config.cp_flag is not None:
        params["cp_flag"] = config.cp_flag
    records: list[dict[str, object]] = []
    for table in config.tables:
        frame = _read_sql(
            db,
            query.format(table=_qualified(config.library, table)),
            params=params,
            date_cols=[config.date_col],
        )
        records.extend(_records_from_frame(frame))
    return records


def fetch_spot_records(db: Any, config: SpotQueryConfig) -> dict[str, float]:
    """Fetch date -> spot mapping from a security-price table."""
    query = _spot_sql(config)
    spots: dict[str, float] = {}
    for table in config.tables:
        frame = _read_sql(
            db,
            query.format(table=_qualified(config.library, table)),
            params={
                "secid": config.secid,
                "start_date": config.start_date,
                "end_date": config.end_date,
            },
            date_cols=[config.date_col],
        )
        for record in _records_from_frame(frame):
            date = _date_key(record[config.date_col])
            spot = _finite_float(record[config.spot_col])
            if spot is not None and spot > 0.0:
                spots[date] = spot
    return spots


def standardize_surface_records(
    records: list[dict[str, object]],
    *,
    date_col: str,
    maturity_col: str,
    delta_col: str,
    iv_col: str,
    spot_col: str | None,
    spot_by_date: dict[str, float] | None = None,
    target_deltas: np.ndarray = DEFAULT_DELTA_GRID,
    target_maturities: np.ndarray = DEFAULT_MATURITY_GRID_DAYS,
) -> list[dict[str, object]]:
    """Convert long-form surface rows into project-standard wide rows."""
    grouped: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    surface_spots: dict[str, float] = {}
    for record in records:
        date = _date_key(record[date_col])
        maturity = _finite_float(record[maturity_col])
        delta = _normalise_delta(record[delta_col])
        iv = _normalise_iv(record[iv_col])
        if maturity is None or delta is None or iv is None:
            continue
        grouped[date].append((maturity, delta, iv))
        if spot_col is not None and spot_col in record:
            spot = _finite_float(record[spot_col])
            if spot is not None and spot > 0.0:
                surface_spots[date] = spot

    rows: list[dict[str, object]] = []
    external_spots = spot_by_date or {}
    for date in sorted(grouped):
        spot = surface_spots.get(date, external_spots.get(date))
        if spot is None or not math.isfinite(spot) or spot <= 0.0:
            continue
        ivs = _interpolate_grid(grouped[date], target_deltas, target_maturities)
        if ivs is None:
            continue
        row: dict[str, object] = {"date": date, "spot": spot}
        for maturity_idx, maturity in enumerate(target_maturities):
            for delta_idx, delta in enumerate(target_deltas):
                row[_iv_column_name(float(maturity), float(delta))] = float(
                    ivs[maturity_idx, delta_idx]
                )
        rows.append(row)
    if not rows:
        raise ValueError("no complete standardized surface rows produced")
    return rows


def write_surface_csv(rows: list[dict[str, object]], path: str | Path) -> None:
    """Write project-standard surface rows to CSV."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["date", "spot"] + [
        _iv_column_name(float(maturity), float(delta))
        for maturity in DEFAULT_MATURITY_GRID_DAYS
        for delta in DEFAULT_DELTA_GRID
    ]
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _surface_sql(config: SurfaceQueryConfig, columns: list[str]) -> str:
    cp_active = config.cp_flag_col is not None and config.cp_flag is not None
    identifiers = [
        config.library,
        *config.tables,
        config.date_col,
        config.secid_col,
        config.maturity_col,
        config.delta_col,
        config.iv_col,
        *(column for column in [config.spot_col] if column is not None),
        *([config.cp_flag_col] if cp_active and config.cp_flag_col is not None else []),
    ]
    _validate_identifiers(identifiers)
    selected = ", ".join(columns)
    cp_clause = f"and {config.cp_flag_col} = :cp_flag " if cp_active else ""
    return (
        f"select {selected} from {{table}} "
        f"where {config.secid_col} = :secid "
        f"{cp_clause}"
        f"and {config.date_col} between :start_date and :end_date"
    )


def _spot_sql(config: SpotQueryConfig) -> str:
    _validate_identifiers(
        [
            config.library,
            *config.tables,
            config.date_col,
            config.secid_col,
            config.spot_col,
        ]
    )
    return (
        f"select {config.date_col}, {config.spot_col} from {{table}} "
        f"where {config.secid_col} = :secid "
        f"and {config.date_col} between :start_date and :end_date"
    )


def _qualified(library: str, table: str) -> str:
    _validate_identifiers([library, table])
    return f"{library}.{table}"


def _validate_identifiers(values: list[str]) -> None:
    bad = [value for value in values if not _IDENTIFIER_RE.match(value)]
    if bad:
        raise ValueError(f"invalid SQL identifier(s): {bad}")


def _read_sql(
    db: Any,
    query: str,
    *,
    params: dict[str, object],
    date_cols: list[str],
) -> Any:
    """Run a parameterized query against WRDS.

    `wrds.Connection.raw_sql` is unusable on the current sqlalchemy/pandas stack:
    pandas forwards the query through `exec_driver_sql`, which neither binds
    `:name` parameters nor accepts pandas' empty `immutabledict`. Wrapping the SQL
    in `sqlalchemy.text()` and executing against the connection's engine restores
    proper named-parameter binding. Fall back to `raw_sql` only if no engine is
    exposed (e.g. a test double).
    """
    engine = getattr(db, "engine", None)
    if engine is None:
        return db.raw_sql(query, params=params, date_cols=date_cols)
    sa = import_module("sqlalchemy")
    pd = import_module("pandas")
    with engine.connect() as conn:
        return pd.read_sql_query(sa.text(query), conn, params=params, parse_dates=date_cols)


def _records_from_frame(frame: Any) -> list[dict[str, object]]:
    records = frame.to_dict("records")
    return [{str(key): value for key, value in record.items()} for record in records]


def _date_key(value: object) -> str:
    return str(value)[:10]


def _finite_float(value: object) -> float | None:
    try:
        number = float(cast(Any, value))
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _normalise_delta(value: object) -> float | None:
    delta = _finite_float(value)
    if delta is None:
        return None
    delta = abs(delta)
    if delta > 1.0:
        delta /= 100.0
    if not 0.0 < delta < 1.0:
        return None
    return delta


def _normalise_iv(value: object) -> float | None:
    iv = _finite_float(value)
    if iv is None:
        return None
    if iv > 3.0:
        iv /= 100.0
    return iv if iv > 0.0 else None


def _interpolate_grid(
    points: list[tuple[float, float, float]],
    target_deltas: np.ndarray,
    target_maturities: np.ndarray,
) -> np.ndarray | None:
    buckets: dict[tuple[float, float], list[float]] = defaultdict(list)
    for maturity, delta, iv in points:
        buckets[(maturity, delta)].append(iv)
    maturities = np.asarray(sorted({key[0] for key in buckets}), dtype=np.float64)
    deltas = np.asarray(sorted({key[1] for key in buckets}), dtype=np.float64)
    if maturities.size == 0 or deltas.size == 0:
        return None

    matrix = np.full((maturities.size, deltas.size), np.nan, dtype=np.float64)
    maturity_index = {float(value): idx for idx, value in enumerate(maturities)}
    delta_index = {float(value): idx for idx, value in enumerate(deltas)}
    for key, values in buckets.items():
        matrix[maturity_index[float(key[0])], delta_index[float(key[1])]] = float(np.mean(values))

    by_delta = np.full((maturities.size, target_deltas.size), np.nan, dtype=np.float64)
    for row_idx in range(maturities.size):
        valid = np.isfinite(matrix[row_idx])
        if not np.any(valid):
            continue
        by_delta[row_idx] = np.interp(target_deltas, deltas[valid], matrix[row_idx, valid])

    out = np.full((target_maturities.size, target_deltas.size), np.nan, dtype=np.float64)
    for delta_idx in range(target_deltas.size):
        valid = np.isfinite(by_delta[:, delta_idx])
        if not np.any(valid):
            return None
        out[:, delta_idx] = np.interp(
            target_maturities,
            maturities[valid],
            by_delta[valid, delta_idx],
        )
    if np.any(~np.isfinite(out)) or np.any(out <= 0.0):
        return None
    return out


def _iv_column_name(maturity: float, delta: float) -> str:
    return f"iv_{int(maturity)}_{round(delta * 100)}"
