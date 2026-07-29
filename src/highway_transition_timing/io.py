"""CSV input/output helpers."""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .utils import serialize_cell


def read_csv_rows(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = collect_fieldnames(rows)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serialize_cell(row.get(key, "")) for key in fieldnames})


def collect_fieldnames(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    seen: set[str] = set()
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    return fieldnames
