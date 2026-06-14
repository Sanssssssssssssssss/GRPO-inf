from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from grpo_inf.training.grpo import _grpo_config_args, _set_tokenizer_eos_token_id


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
    sample_path = Path("examples/tiny_dataset/grpo/prompts_test_locked.jsonl")
    sample = json.loads(sample_path.read_text(encoding="utf-8").splitlines()[0])
    outputs_path = tmp_path / "model_outputs.jsonl"
    outputs_path.write_text(
        json.dumps({"case_id": sample["case_id"], "completion": json.dumps(sample["gold"], ensure_ascii=False)}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    result = run_cli(
        "eval-reviewer",
        "--samples",
        str(sample_path),
        "--outputs",
        str(outputs_path),
        "--summary-out",
        str(summary),
        "--scored-out",
        str(scored),
    )
    payload = json.loads(result.stdout)
    assert payload["json_valid_rate"] == 1.0
    assert payload["schema_valid_rate"] == 1.0
    assert payload["mode_accuracy"] == 1.0
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


def test_default_grpo_config_has_no_repo_only_keys_in_grpo_block() -> None:
    config = json.loads(Path("configs/training/gemma4_31b_grpo.json").read_text(encoding="utf-8"))
    assert "notes" in config
    assert "notes" not in config["grpo"]


def test_grpo_execute_filters_non_grpo_config_keys() -> None:
    @dataclass
    class DummyGRPOConfig:
        max_steps: int = 1
        generation_kwargs: dict | None = None

    assert _grpo_config_args(
        {
            "max_steps": 1,
            "notes": "local",
            "unknown": True,
            "generation_kwargs": {"eos_token_id": [1, 106]},
        },
        DummyGRPOConfig,
    ) == {"max_steps": 1, "generation_kwargs": {"eos_token_id": [1, 106]}}




def test_grpo_tokenizer_eos_override_uses_eot_token() -> None:
    class DummyTokenizer:
        eos_token = "<eos>"
        eos_token_id = 1
        unk_token = "<unk>"

        def convert_ids_to_tokens(self, token_id: int) -> str:
            return {1: "<eos>", 106: "<turn|>"}[token_id]

    class DummyProcessor:
        tokenizer = DummyTokenizer()

    processor = DummyProcessor()
    _set_tokenizer_eos_token_id(processor, 106)
    assert processor.tokenizer.eos_token == "<turn|>"
    assert processor.tokenizer.eos_token_id == 106


def test_print_schema_cli(tmp_path: Path) -> None:
    schema_path = tmp_path / "schema.json"
    run_cli("print-schema", "--out", str(schema_path))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert "mode" in schema["properties"]
    assert "decision" not in schema["properties"]


def test_build_dataset_zip_smoke_when_available(tmp_path: Path) -> None:
    source_zip = Path("C:/Users/X/Downloads/invoice_reviewer_public_review_500_v2.zip")
    if not source_zip.exists():
        return
    out = tmp_path / "public_review_smoke"
    result = run_cli("build-dataset", "--source", "zip-smoke", "--input-zip", str(source_zip), "--out", str(out))
    payload = json.loads(result.stdout)
    assert payload["audit"]["total_cases"] == 500
    assert payload["audit"]["not_for_final_training"] is True
    assert (out / "sft" / "reviewer_train.jsonl").exists()
    assert (out / "grpo" / "prompts_train.jsonl").exists()
    assert (out / "eval" / "locked_cases.jsonl").exists()
    first = json.loads((out / "grpo" / "prompts_train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert isinstance(first["input"], dict)
    assert first["input"]["mode"] in {"extract", "review"}


def test_fatura_build_requires_explicit_pipeline_zip(tmp_path: Path) -> None:
    env = {key: value for key, value in os.environ.items() if key != "PUBLIC_INVOICE_PIPELINE_ZIP"}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "grpo_inf.cli",
            "build-dataset",
            "--source",
            "fatura",
            "--repo-root",
            str(tmp_path / "workbench"),
            "--out",
            str(tmp_path / "out"),
            "--no-download",
        ],
        text=True,
        capture_output=True,
        env=env,
    )
    assert result.returncode != 0
    assert "PUBLIC_INVOICE_PIPELINE_ZIP" in result.stderr
