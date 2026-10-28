"""Atomic versioned run state and strict experiment compatibility."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import Iterator
from contextlib import contextmanager
import hashlib
import json
import os
import torch

from utils.config import Config


@contextmanager
def run_lock(directory: str | Path) -> Iterator[None]:
    """Prevent simultaneous writers; OS releases the lock after a process crash."""
    with (Path(directory) / ".run.lock").open("a+b") as handle:
        handle.seek(0)
        if handle.read(1) == b"":
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(f"Run is already locked by another writer: {directory}") from error
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def compatibility_hash(cfg: Config, datasets: dict[str, str]) -> str:
    """Resume can change wall-time/output location, not the scientific schedule."""
    raw = cfg.to_dict()
    raw.pop("output")
    raw["training"].pop("max_seconds")
    raw["training"].pop("device")
    return hashlib.sha256(json.dumps({"config": raw, "datasets": datasets}, sort_keys=True).encode()).hexdigest()


def save_checkpoint(path: str | Path, state: dict[str, Any]) -> None:
    """Replace one run's checkpoint atomically after fully writing the new state."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as handle:
        torch.save({**state, "schema_version": 1}, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load tensor/plain-value project state without arbitrary Python objects."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or state.get("schema_version") != 1:
        raise ValueError("Unsupported checkpoint schema")
    required = {"model", "config", "step", "cursor", "compatibility", "datasets", "rng", "environment", "source"}
    if not required <= state.keys():
        raise ValueError("Checkpoint is missing resumable metadata")
    return state
