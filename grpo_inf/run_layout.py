from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUN_SUBDIRS = ("config", "checkpoints", "adapter", "logs", "generations", "eval", "visualizations")


def default_run_id(prefix: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{stamp}"


def create_run_layout(output_root: str | Path, run_id: str) -> dict[str, Path]:
    root = Path(output_root) / "runs" / run_id
    paths = {"root": root}
    for subdir in RUN_SUBDIRS:
        path = root / subdir
        path.mkdir(parents=True, exist_ok=True)
        paths[subdir] = path
    return paths


def write_run_manifest(paths: dict[str, Path], config: dict[str, Any], command: str) -> None:
    manifest = {
        "run_id": paths["root"].name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "config": config,
        "environment": {
            "cwd": os.getcwd(),
            "python": os.environ.get("PYTHONPATH", ""),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        },
    }
    (paths["config"] / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
