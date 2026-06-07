from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if config_path.suffix.lower() != ".json":
        raise ValueError("Training configs are JSON in this repo to keep parsing dependency-free.")
    return _expand_env(json.loads(config_path.read_text(encoding="utf-8")))


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        expanded = re.sub(r"\$\{([^}]+)\}", lambda match: os.environ.get(match.group(1), match.group(0)), value)
        return os.path.expandvars(expanded)
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    return value


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def validate_training_config(config: dict[str, Any], kind: str) -> list[str]:
    errors: list[str] = []
    if not config.get("model_id"):
        errors.append("model_id is required")
    if not config.get("data_path"):
        errors.append("data_path is required")
    if not config.get("output_root"):
        errors.append("output_root is required")
    if kind == "grpo":
        grpo = config.get("grpo", {})
        if int(grpo.get("num_generations", 0)) < 2:
            errors.append("grpo.num_generations must be >= 2")
        if grpo.get("loss_type") not in {"grpo", "bnpo", "dr_grpo", "dapo", "cispo", "sapo", "luspo", "vespo"}:
            errors.append("grpo.loss_type is missing or unsupported")
    if kind == "sft" and not config.get("sft", {}):
        errors.append("sft section is required")
    return errors
