from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

from grpo_inf.schema import schema_valid


def _workbench_repo_root() -> Path | None:
    candidates = []
    if os.environ.get("WORKBENCH_REPO_ROOT"):
        candidates.append(Path(os.environ["WORKBENCH_REPO_ROOT"]))
    here = Path(__file__).resolve()
    candidates.extend(
        [
            here.parents[2] / "invoice-case-workbench-openai-sdk",
            here.parents[2] / "Software_for_Gemini_Ultra" / "invoice-case-workbench-openai-sdk",
        ]
    )
    for candidate in candidates:
        if (candidate / "backend" / "app" / "state" / "schemas.py").exists():
            return candidate
    return None


def test_local_evidence_review_result_schema_accepts_tiny_fixture() -> None:
    row = json.loads(Path("examples/tiny_dataset/grpo/prompts_train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert schema_valid(row["gold"])


def test_workbench_evidence_review_result_pydantic_accepts_grpo_output() -> None:
    repo_root = _workbench_repo_root()
    if not repo_root:
        pytest.skip("set WORKBENCH_REPO_ROOT to run workbench integration contract test")
    schema_path = Path(repo_root) / "backend" / "app" / "state" / "schemas.py"
    sys.path.insert(0, str(Path(repo_root)))
    spec = importlib.util.spec_from_file_location("workbench_schemas", schema_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    row = json.loads(Path("examples/tiny_dataset/grpo/prompts_test_locked.jsonl").read_text(encoding="utf-8").splitlines()[0])
    output = row["gold"]
    parsed = module.EvidenceReviewResult.model_validate(output)
    assert parsed.mode == "review"
    assert parsed.source_doc_id
    assert parsed.extracted_fields
    assert parsed.suggested_patch is not None
    normalizer_path = Path(repo_root) / "backend" / "app" / "runtime" / "patch_normalizer.py"
    if not normalizer_path.exists():
        pytest.skip("workbench patch_normalizer.py not found")
    normalizer_spec = importlib.util.spec_from_file_location("workbench_patch_normalizer", normalizer_path)
    assert normalizer_spec and normalizer_spec.loader
    normalizer = importlib.util.module_from_spec(normalizer_spec)
    normalizer_spec.loader.exec_module(normalizer)
    patch = parsed.suggested_patch.model_dump(mode="json")
    compact = normalizer.compact_case_patch_for_write(patch)
    assert isinstance(compact, dict)
