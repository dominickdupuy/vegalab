"""Build the thesis exhibit manifest."""

from __future__ import annotations

from optspread.reporting.exhibits import standard_exhibits
from optspread.reporting.manifest import validate_manifest


def main() -> None:
    exhibits = standard_exhibits()
    if not validate_manifest(exhibits):
        raise SystemExit("invalid exhibit manifest")
    for exhibit in exhibits:
        print(
            f"{exhibit.exhibit_id}: {exhibit.section} — "
            f"{exhibit.title} [{exhibit.synthetic_or_real}]"
        )


if __name__ == "__main__":
    main()
