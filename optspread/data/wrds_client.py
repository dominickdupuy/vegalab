"""WRDS connection helpers that keep credentials out of repo artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

from optspread.data.envfile import read_env_file


@dataclass(frozen=True, slots=True)
class WRDSCredentials:
    username: str
    password: str


def credentials_from_env_file(
    path: str | Path = ".env",
    *,
    username_key: str = "username",
    password_key: str = "pwrd",
) -> WRDSCredentials:
    """Load WRDS credentials from an env file without exposing the values."""
    env = read_env_file(path)
    username = env.get(username_key, "")
    password = env.get(password_key, "")
    if not username:
        raise ValueError(f"missing WRDS username key: {username_key}")
    if not password:
        raise ValueError(f"missing WRDS password key: {password_key}")
    return WRDSCredentials(username=username, password=password)


def connect_wrds(credentials: WRDSCredentials, *, verbose: bool = False) -> Any:
    """Return an authenticated `wrds.Connection`.

    The upstream client falls back to interactive prompts after a failed first
    attempt. For unattended jobs we connect once with the supplied credentials and
    raise the real error instead of prompting.
    """
    wrds = import_module("wrds")
    db = wrds.Connection(
        wrds_username=credentials.username,
        wrds_password=credentials.password,
        autoconnect=False,
        verbose=verbose,
    )
    db._Connection__make_sa_engine_conn(raise_err=True)
    db.load_library_list()
    return db


def redact_secret_text(text: str, credentials: WRDSCredentials) -> str:
    """Remove credential values from exception messages before printing."""
    return text.replace(credentials.username, "<username>").replace(
        credentials.password, "<password>"
    )
