from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from grpo_inf.run_layout import create_run_layout, default_run_id, write_run_manifest
from grpo_inf.training.config import validate_training_config


def prepare_training_run(config: dict[str, Any], kind: str, run_id: str | None, command: str) -> tuple[str, dict[str, Path]]:
    errors = validate_training_config(config, kind)
    if errors:
        raise ValueError("; ".join(errors))
    resolved_run_id = run_id or default_run_id(kind)
    paths = create_run_layout(config.get("output_root", "outputs"), resolved_run_id)
    write_run_manifest(paths, config, command)
    (paths["config"] / f"{kind}_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return resolved_run_id, paths


def dry_run_summary(config: dict[str, Any], kind: str, run_id: str, paths: dict[str, Path]) -> dict[str, Any]:
    data_path = Path(str(config.get("data_path", "")))
    return {
        "mode": "dry_run",
        "kind": kind,
        "run_id": run_id,
        "model_id": config.get("model_id"),
        "data_path": str(data_path),
        "data_path_exists": data_path.exists(),
        "output_root": str(paths["root"]),
        "execute_hint": f"rerun train-{kind} with --execute on CSD3 to start model loading/training",
    }
