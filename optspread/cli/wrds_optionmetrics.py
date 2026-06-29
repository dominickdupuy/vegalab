"""WRDS OptionMetrics discovery/export CLI.

This command is intended for a normal local shell with WRDS network access. The
Codex sandbox may not be able to reach WRDS even after local authentication.
"""

from __future__ import annotations

import argparse
import json
from importlib import import_module
from pathlib import Path
from typing import Any

from optspread.data.wrds_client import (
    connect_wrds,
    credentials_from_env_file,
    redact_secret_text,
)
from optspread.data.wrds_optionmetrics import (
    SpotQueryConfig,
    SurfaceQueryConfig,
    annual_tables,
    describe_candidate_tables,
    fetch_spot_records,
    fetch_surface_records,
    standardize_surface_records,
    write_surface_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover/export WRDS OptionMetrics data")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--username-key", default="username")
    parser.add_argument("--password-key", default="pwrd")
    parser.add_argument("--pgpass", action="store_true", help="Use pgpass/default WRDS auth")
    subparsers = parser.add_subparsers(dest="command", required=True)

    discover = subparsers.add_parser("discover", help="List likely OptionMetrics tables/columns")
    discover.add_argument(
        "--libraries",
        nargs="+",
        default=["optionm", "optionm_all", "optionmsamp_us"],
    )
    discover.add_argument("--out", type=Path)

    export = subparsers.add_parser("export-surface", help="Export standardized surface CSV")
    export.add_argument("--library", required=True)
    export.add_argument("--surface-table", action="append", default=[])
    export.add_argument("--surface-table-pattern")
    export.add_argument("--secid", type=int, required=True)
    export.add_argument("--start", required=True, help="YYYY-MM-DD")
    export.add_argument("--end", required=True, help="YYYY-MM-DD")
    export.add_argument("--out", type=Path, required=True)
    export.add_argument("--date-col", default="date")
    export.add_argument("--secid-col", default="secid")
    export.add_argument("--maturity-col", default="days")
    export.add_argument("--delta-col", default="delta")
    export.add_argument("--iv-col", default="impl_volatility")
    export.add_argument("--cp-flag-col", default="cp_flag")
    export.add_argument(
        "--cp-flag",
        default="C",
        help="Option type to keep ('C' or 'P'); calls and puts must not be mixed.",
    )
    export.add_argument("--surface-spot-col")
    export.add_argument("--spot-library")
    export.add_argument("--spot-table", action="append", default=[])
    export.add_argument("--spot-table-pattern")
    export.add_argument("--spot-date-col", default="date")
    export.add_argument("--spot-secid-col", default="secid")
    export.add_argument("--spot-col", default="close")

    args = parser.parse_args()
    db = _connect(args)
    try:
        if args.command == "discover":
            _discover(args, db)
        elif args.command == "export-surface":
            _export_surface(args, db)
        else:
            raise ValueError(f"unsupported command: {args.command}")
    finally:
        db.close()


def _connect(args: argparse.Namespace) -> Any:
    if args.pgpass:
        wrds = import_module("wrds")
        return wrds.Connection(autoconnect=True, verbose=False)
    credentials = credentials_from_env_file(
        args.env_file,
        username_key=args.username_key,
        password_key=args.password_key,
    )
    try:
        return connect_wrds(credentials)
    except Exception as exc:
        message = redact_secret_text(str(exc), credentials)
        raise SystemExit(f"WRDS_CONNECT_FAILED\n{type(exc).__name__}\n{message}") from exc


def _discover(args: argparse.Namespace, db: Any) -> None:
    payload = describe_candidate_tables(db, tuple(args.libraries))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("WRDS_DISCOVERY_OK")
    print("candidate_tables", len(payload))
    for item in payload[:20]:
        columns = item["columns"]
        preview = columns[:12] if isinstance(columns, list) else columns
        print(f"{item['library']}.{item['table']}: {preview}")


def _export_surface(args: argparse.Namespace, db: Any) -> None:
    surface_tables = _resolve_tables(args.surface_table, args.surface_table_pattern, args)
    if not surface_tables:
        raise SystemExit("--surface-table or --surface-table-pattern is required")
    surface_config = SurfaceQueryConfig(
        library=args.library,
        tables=surface_tables,
        secid=args.secid,
        start_date=args.start,
        end_date=args.end,
        date_col=args.date_col,
        secid_col=args.secid_col,
        maturity_col=args.maturity_col,
        delta_col=args.delta_col,
        iv_col=args.iv_col,
        spot_col=args.surface_spot_col,
        cp_flag_col=args.cp_flag_col or None,
        cp_flag=args.cp_flag or None,
    )
    surface_records = fetch_surface_records(db, surface_config)

    spot_by_date = None
    spot_tables = _resolve_tables(args.spot_table, args.spot_table_pattern, args)
    if spot_tables:
        spot_config = SpotQueryConfig(
            library=args.spot_library or args.library,
            tables=spot_tables,
            secid=args.secid,
            start_date=args.start,
            end_date=args.end,
            date_col=args.spot_date_col,
            secid_col=args.spot_secid_col,
            spot_col=args.spot_col,
        )
        spot_by_date = fetch_spot_records(db, spot_config)

    rows = standardize_surface_records(
        surface_records,
        date_col=args.date_col,
        maturity_col=args.maturity_col,
        delta_col=args.delta_col,
        iv_col=args.iv_col,
        spot_col=args.surface_spot_col,
        spot_by_date=spot_by_date,
    )
    write_surface_csv(rows, args.out)
    print("WRDS_SURFACE_EXPORT_OK")
    print("surface_records", len(surface_records))
    print("standardized_rows", len(rows))
    print("out", args.out)


def _resolve_tables(
    explicit: list[str],
    pattern: str | None,
    args: argparse.Namespace,
) -> tuple[str, ...]:
    tables = list(explicit)
    if pattern is not None:
        tables.extend(annual_tables(pattern, start_date=args.start, end_date=args.end))
    if not tables and pattern is None:
        return ()
    if not tables:
        raise SystemExit("at least one table or table pattern is required")
    return tuple(tables)


if __name__ == "__main__":
    main()
