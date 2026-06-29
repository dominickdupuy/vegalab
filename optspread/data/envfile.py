"""Small `.env` reader for local data credentials.

This avoids adding a dependency and, more importantly, keeps secrets out of logs:
callers decide which keys to read and never need to print values.
"""

from __future__ import annotations

from pathlib import Path


def read_env_file(path: str | Path) -> dict[str, str]:
    """Parse simple KEY=VALUE lines from an env file."""
    env_path = Path(path)
    if not env_path.exists():
        raise FileNotFoundError(f"env file not found: {env_path}")
    values: dict[str, str] = {}
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        cleaned = value.strip().strip('"').strip("'")
        values[key.strip()] = cleaned
    return values
