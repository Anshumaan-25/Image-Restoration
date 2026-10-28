"""Stable sample keys, RNG snapshots and truthful source identity."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import hashlib
import json
import random
import subprocess
import numpy as np
import torch


def stable_seed(*parts: object) -> int:
    """SHA256 UTF-8 JSON keys; independent of Python's salted hash."""
    raw = json.dumps(parts, ensure_ascii=True, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % (2**63 - 1)


def seed_everything(seed: int) -> None:
    """Initialize global RNGs; sample randomness uses independent CPU generators."""
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng() -> dict[str, Any]:
    """Use tensor/plain-value encoding compatible with weights_only loading."""
    name, keys, pos, has_gauss, cached = np.random.get_state()
    return {"python": random.getstate(), "numpy": [name, keys.tolist(), pos, has_gauss, cached],
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "mps": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None}


def restore_rng(state: dict[str, Any]) -> None:
    """Restore states only on the backend where they are available."""
    random.setstate(state["python"])
    name, keys, pos, has_gauss, cached = state["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), pos, has_gauss, cached))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    if state["mps"] is not None and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"].cpu())


def source_identity() -> dict[str, Any]:
    """Record unborn Git honestly and fingerprint relevant uncommitted sources."""
    root = Path(__file__).resolve().parents[1]
    def git(*args: str) -> str | None:
        result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    hashes = {}
    for directory in (root / "models", root / "losses", root / "utils", root / "scripts", root / "tests", root / "configs"):
        for path in sorted(directory.rglob("*")):
            if path.suffix in {".py", ".yaml"}:
                hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ("train.py", "eval.py", "inference.py", "requirements.txt"):
        path = root / name
        if path.is_file():
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"git_commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain")), "files": hashes}
