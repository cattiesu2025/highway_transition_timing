"""Small conversion helpers for CSV-like rows."""

from __future__ import annotations

import ast
import json
from collections.abc import Iterable, Mapping
from typing import Any


def get_str(row: Mapping[str, Any], key: str, default: str = "") -> str:
    value = row.get(key, default)
    if value is None:
        return default
    return str(value)


def get_int(row: Mapping[str, Any], key: str, default: int = 0) -> int:
    value = row.get(key, default)
    if value in (None, ""):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(float(value))


def get_float(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def get_bool(row: Mapping[str, Any], key: str, default: bool = False) -> bool:
    value = row.get(key, default)
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    return default


def parse_number_list(value: Any) -> list[float]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            try:
                value = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                value = [part.strip() for part in text.split(",")]
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, Mapping)):
        numbers: list[float] = []
        for item in value:
            try:
                numbers.append(float(item))
            except (TypeError, ValueError):
                continue
        return numbers
    try:
        return [float(value)]
    except (TypeError, ValueError):
        return []


def json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def serialize_cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json_dumps(value)
    if value is None:
        return ""
    return value


def median(values: list[float]) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    n = len(xs)
    mid = n // 2
    if n % 2:
        return xs[mid]
    return (xs[mid - 1] + xs[mid]) / 2.0


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if q <= 0:
        return min(values)
    if q >= 1:
        return max(values)
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    low = int(pos)
    high = min(low + 1, len(xs) - 1)
    frac = pos - low
    return xs[low] * (1 - frac) + xs[high] * frac
