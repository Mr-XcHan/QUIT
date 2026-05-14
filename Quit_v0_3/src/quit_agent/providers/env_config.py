from __future__ import annotations

import os
from pathlib import Path


def load_env_file(path: str | Path, override: bool = False) -> dict[str, str]:
    """Load simple dotenv-style KEY=value pairs into os.environ."""

    env_path = Path(path).expanduser()
    if not env_path.exists():
        return {}
    loaded: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = _strip_inline_comment(value.strip())
        value = _strip_quotes(value)
        if not key:
            continue
        loaded[key] = value
        if override or key not in os.environ:
            os.environ[key] = value
    return loaded


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _strip_inline_comment(value: str) -> str:
    quote: str | None = None
    for index, char in enumerate(value):
        if char in {"'", '"'}:
            quote = None if quote == char else char
        if char == "#" and quote is None:
            before = value[:index]
            if not before or before[-1].isspace():
                return before.strip()
    return value


def env_value(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value


def env_float(name: str, default: float | None = None) -> float | None:
    value = env_value(name)
    return default if value is None else float(value)


def env_int(name: str, default: int | None = None) -> int | None:
    value = env_value(name)
    return default if value is None else int(value)


def env_bool(name: str, default: bool = False) -> bool:
    value = env_value(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}
