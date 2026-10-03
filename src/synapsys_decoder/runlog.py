"""Reproducible run artifacts: every experiment writes its config, environment,
aggregate metrics, and per-trial rows under ``results/<run_id>/``.
"""

from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


def make_run_id(name: str, *, now: datetime | None = None) -> str:
    """``YYYYmmdd-HHMMSS_<name>`` in UTC."""
    now = now or datetime.now(UTC)
    return f"{now:%Y%m%d-%H%M%S}_{name}"


def git_revision(repo: Path | None = None) -> str | None:
    """Current commit hash with ``-dirty`` suffix if the tree has changes; None outside git."""
    cwd = repo or Path(__file__).resolve().parents[2]
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return f"{rev}-dirty" if dirty else rev


def environment_info() -> dict[str, Any]:
    import matplotlib
    import numpy
    import scipy

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "git_revision": git_revision(),
    }


def create_run_dir(results_root: str | Path, name: str) -> Path:
    run_dir = Path(results_root) / make_run_id(name)
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "tolist"):  # numpy scalars/arrays
        return obj.tolist()
    return obj


def save_run(
    run_dir: Path,
    *,
    config: Mapping[str, Any],
    metrics: Mapping[str, Any],
    trials: Sequence[Mapping[str, Any]] = (),
    command: Sequence[str] | None = None,
) -> None:
    """Write ``config.yaml``, ``run.json`` (env + command + metrics) and ``trials.csv``."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(_jsonable(config), sort_keys=False))
    meta = {
        "run_id": run_dir.name,
        "command": list(command) if command is not None else sys.argv,
        "environment": environment_info(),
        "metrics": _jsonable(metrics),
    }
    (run_dir / "run.json").write_text(json.dumps(meta, indent=2))
    if trials:
        rows = [_jsonable(r) for r in trials]
        fields: list[str] = []
        for r in rows:
            fields.extend(k for k in r if k not in fields)
        with (run_dir / "trials.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
