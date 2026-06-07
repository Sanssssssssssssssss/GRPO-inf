from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "grpo_inf.cli", *args],
        check=True,
        text=True,
        capture_output=True,
    )


def test_eval_and_visualize_cli(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "smoke"
    summary = run_dir / "eval" / "summary.json"
    scored = run_dir / "eval" / "scored.jsonl"
    result = run_cli(
        "eval-reviewer",
        "--samples",
        "examples/tiny_dataset/grpo/prompts_test_locked.jsonl",
        "--outputs",
        "examples/tiny_dataset/outputs/model_outputs.jsonl",
        "--summary-out",
        str(summary),
        "--scored-out",
        str(scored),
    )
    payload = json.loads(result.stdout)
    assert payload["json_valid_rate"] == 1.0
    assert summary.exists()
    assert scored.exists()

    viz = run_cli("visualize-run", "--run-dir", str(run_dir))
    viz_payload = json.loads(viz.stdout)
    assert Path(viz_payload["dashboard"]).exists()


def test_training_dry_run_cli(tmp_path: Path) -> None:
    config = {
        "model_id": "google/gemma-4-31B-it",
        "data_path": "examples/tiny_dataset/grpo/prompts_train.jsonl",
        "output_root": str(tmp_path / "outputs"),
        "bf16": True,
        "grpo": {
            "num_generations": 4,
            "max_prompt_length": 1024,
            "max_completion_length": 256,
            "learning_rate": 0.000001,
            "loss_type": "dr_grpo",
            "scale_rewards": "batch",
        },
    }
    config_path = tmp_path / "grpo.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    result = run_cli("train-grpo", "--config", str(config_path), "--run-id", "dry")
    payload = json.loads(result.stdout)
    assert payload["mode"] == "dry_run"
    assert payload["data_path_exists"] is True
    assert (tmp_path / "outputs" / "runs" / "dry" / "config" / "grpo_config.json").exists()


def test_print_schema_cli(tmp_path: Path) -> None:
    schema_path = tmp_path / "schema.json"
    run_cli("print-schema", "--out", str(schema_path))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert "decision" in schema["properties"]
