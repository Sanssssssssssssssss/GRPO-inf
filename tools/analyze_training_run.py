from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _trainer_states(run_dir: Path) -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = []
    for path in sorted((run_dir / "checkpoints").glob("checkpoint-*/trainer_state.json")):
        payload = _read_json(path)
        payload["_path"] = str(path)
        states.append(payload)
    direct = run_dir / "checkpoints" / "trainer_state.json"
    if direct.exists():
        payload = _read_json(direct)
        payload["_path"] = str(direct)
        states.append(payload)
    return states


def _numeric(values: list[Any]) -> list[float]:
    out: list[float] = []
    for value in values:
        if isinstance(value, (int, float)):
            out.append(float(value))
    return out


def _summarize_log_history(states: list[dict[str, Any]]) -> dict[str, Any]:
    if not states:
        return {"found": False}
    state = max(states, key=lambda item: int(item.get("global_step") or 0))
    logs = state.get("log_history") if isinstance(state.get("log_history"), list) else []
    step_logs = [row for row in logs if isinstance(row, dict) and "step" in row]
    rewards = _numeric([row.get("reward") for row in step_logs])
    losses = _numeric([row.get("loss") for row in step_logs])
    grad_norms = _numeric([row.get("grad_norm") for row in step_logs])
    clipped = _numeric([row.get("clip_ratio") or row.get("clipped_ratio") for row in step_logs])
    step_times = _numeric([row.get("step_time") for row in step_logs])
    return {
        "found": True,
        "state_path": state.get("_path"),
        "global_step": state.get("global_step"),
        "log_count": len(step_logs),
        "last_log": step_logs[-1] if step_logs else None,
        "reward_mean": sum(rewards) / len(rewards) if rewards else None,
        "loss_last": losses[-1] if losses else None,
        "grad_norm_positive_logs": sum(1 for value in grad_norms if value > 0),
        "clipped_ratio_last": clipped[-1] if clipped else None,
        "step_time_mean": sum(step_times) / len(step_times) if step_times else None,
    }


def _summarize_completions(run_dir: Path) -> dict[str, Any]:
    paths = sorted((run_dir / "checkpoints" / "completions").glob("*.parquet"))
    if not paths:
        return {"found": False, "files": 0}
    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover - optional dependency in tiny envs
        return {"found": True, "files": len(paths), "error": f"pandas_unavailable:{exc}"}
    frames = [pd.read_parquet(path) for path in paths]
    df = pd.concat(frames, ignore_index=True)
    summary: dict[str, Any] = {"found": True, "files": len(paths), "rows": int(len(df)), "columns": list(df.columns)}
    for column in (
        "reward_total",
        "reward_schema_valid",
        "reward_contract_valid",
        "reward_truncated_completion",
        "reward_eos_terminated",
        "reward_completion_tokens",
    ):
        if column in df.columns:
            values = pd.to_numeric(df[column], errors="coerce").dropna()
            if not values.empty:
                summary[f"{column}_mean"] = float(values.mean())
                summary[f"{column}_min"] = float(values.min())
                summary[f"{column}_max"] = float(values.max())
    return summary


def analyze(run_dir: Path) -> dict[str, Any]:
    states = _trainer_states(run_dir)
    return {
        "run_dir": str(run_dir),
        "exists": run_dir.exists(),
        "trainer_state": _summarize_log_history(states),
        "completions": _summarize_completions(run_dir),
        "has_adapter": (run_dir / "adapter" / "adapter_model.safetensors").exists(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize SFT/GRPO canary run outputs.")
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.run_dir), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
