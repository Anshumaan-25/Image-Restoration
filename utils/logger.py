"""Detached scalar logs and strict JSON output."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import csv
import json
import math


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    """Write strict JSON, encoding infinite PSNR explicitly."""
    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                if isinstance(item, float) and not math.isfinite(item):
                    if item == float("inf") and "psnr" in key:
                        result[key], result[key + "_is_infinite"] = None, True
                    else:
                        raise ValueError(f"Nonfinite report value: {key}")
                else:
                    result[key] = clean(item)
            if "psnr" in value:
                result.setdefault("psnr_is_infinite", False)
            return result
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        return value
    Path(path).write_text(json.dumps(clean(data), indent=2, allow_nan=False) + "\n")


def log_row(path: str | Path, row: dict[str, Any]) -> None:
    """Append plain scalars only; never retain a tensor computation graph."""
    destination = Path(path)
    if any(not isinstance(v, (int, float, str, type(None))) for v in row.values()):
        raise ValueError("Logs accept detached scalar values only")
    exists = destination.exists()
    with destination.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
