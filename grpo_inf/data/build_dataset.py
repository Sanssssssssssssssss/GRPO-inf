from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Iterable

from grpo_inf.data.audit import audit_dataset
from grpo_inf.io import DatasetReader, read_jsonl, write_json, write_jsonl


DEFAULT_PIPELINE_ZIP = Path("C:/Users/X/Downloads/invoice_reviewer_public_invoice_pipeline_v2.zip")


def _case_id(row: dict[str, Any]) -> str:
    return str(row.get("case_id") or row.get("id") or "")


def _read_suffix(reader: DatasetReader, suffix: str) -> list[dict[str, Any]]:
    return reader.read_jsonl_suffix(suffix)


def _split_rows(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result = {"train": [], "dev": [], "test_locked": []}
    for row in rows:
        split = str(row.get("split") or "train")
        if split in result:
            result[split].append(row)
    return result


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _canonical_sft_row(row: dict[str, Any]) -> dict[str, Any]:
    return {"case_id": _case_id(row), "messages": row.get("messages", [])}


def _canonical_grpo_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": _case_id(row),
        "split": row.get("split"),
        "prompt": row.get("prompt"),
        "gold": row.get("gold") or row.get("answer") or row.get("expected_answer"),
        "reward_metadata": row.get("reward_metadata") or row.get("source") or {},
    }


def build_from_public_review_zip(input_zip: str | Path, out_dir: str | Path, smoke_seed: bool = True) -> dict[str, Any]:
    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    for subdir in ("sft", "grpo", "eval", "system_call", "answers", "case_inputs", "validation", "manifests"):
        (out / subdir).mkdir(parents=True, exist_ok=True)

    with DatasetReader(input_zip) as reader:
        payloads = _read_suffix(reader, "case_inputs/reviewer_payloads.jsonl")
        answers = _read_suffix(reader, "answers/evidence_review_result_expected.jsonl")
        sft_rows = _read_suffix(reader, "sft/reviewer_sft_messages.jsonl")
        grpo_rows = _read_suffix(reader, "grpo/prompts.jsonl")
        direct_rows = _read_suffix(reader, "system_call/reviewer_direct_call_eval.jsonl")
        locked_rows = _read_suffix(reader, "splits/test_locked.jsonl")
        quality = reader.read_text(reader.resolve("validation/quality_report.json")) if reader.resolve("validation/quality_report.json") else "{}"
        manifest = reader.read_text(reader.resolve("manifests/dataset_manifest.json")) if reader.resolve("manifests/dataset_manifest.json") else "{}"

    split_grpo = _split_rows(grpo_rows)
    split_sft = _split_rows(sft_rows)
    write_jsonl(out / "sft/reviewer_train.jsonl", (_canonical_sft_row(row) for row in split_sft["train"] or sft_rows))
    if split_sft["dev"]:
        write_jsonl(out / "sft/reviewer_dev.jsonl", (_canonical_sft_row(row) for row in split_sft["dev"]))
    write_jsonl(out / "grpo/prompts_train.jsonl", (_canonical_grpo_row(row) for row in split_grpo["train"]))
    write_jsonl(out / "grpo/prompts_dev.jsonl", (_canonical_grpo_row(row) for row in split_grpo["dev"]))
    write_jsonl(out / "grpo/prompts_test_locked.jsonl", (_canonical_grpo_row(row) for row in split_grpo["test_locked"]))
    write_jsonl(out / "eval/locked_cases.jsonl", locked_rows)
    write_jsonl(out / "system_call/reviewer_extract_payloads.jsonl", direct_rows or payloads)
    write_jsonl(out / "case_inputs/reviewer_payloads.jsonl", payloads)
    write_jsonl(out / "answers/evidence_review_result_answers.jsonl", answers)

    quality_obj = json.loads(quality)
    quality_obj["imported_as_smoke_seed"] = bool(smoke_seed)
    quality_obj["not_for_final_training"] = bool(smoke_seed)
    quality_obj["reason_not_final"] = "source documents repeat across splits in public_review_500_v2" if smoke_seed else None
    write_json(out / "validation/quality_report.json", quality_obj)
    write_json(out / "manifests/dataset_manifest.json", json.loads(manifest))

    audit = audit_dataset(out, out / "validation/audit_report.json", smoke_seed=smoke_seed)
    return {"out": str(out), "source": "zip-smoke", "audit": audit}


def _find_pipeline_zip(pipeline_zip: str | Path | None) -> Path:
    if pipeline_zip:
        path = Path(pipeline_zip)
    elif os.environ.get("PUBLIC_INVOICE_PIPELINE_ZIP"):
        path = Path(os.environ["PUBLIC_INVOICE_PIPELINE_ZIP"])
    else:
        path = DEFAULT_PIPELINE_ZIP
    if not path.exists():
        raise FileNotFoundError(
            f"public invoice pipeline zip not found: {path}. Set PUBLIC_INVOICE_PIPELINE_ZIP or pass --pipeline-zip."
        )
    return path


def _run_pipeline_zip(
    pipeline_zip: Path,
    repo_root: str | Path,
    out_dir: str | Path,
    target_cases: int,
    download: bool,
    fatura_zip: str | Path | None,
) -> None:
    out = Path(out_dir)
    with tempfile.TemporaryDirectory(prefix="grpo_inf_public_pipeline_") as tmp:
        tmp_path = Path(tmp)
        with zipfile.ZipFile(pipeline_zip) as zf:
            zf.extractall(tmp_path)
        script_matches = list(tmp_path.rglob("scripts/build_public_invoice_reviewer_dataset.py"))
        if not script_matches:
            raise FileNotFoundError("pipeline zip does not contain scripts/build_public_invoice_reviewer_dataset.py")
        script = script_matches[0]
        cmd = [
            sys.executable,
            str(script),
            "--repo-root",
            str(repo_root),
            "--dataset",
            "fatura",
            "--target-cases",
            str(target_cases),
            "--out",
            str(out),
            "--use-repo-extractor",
            "--copy-images",
        ]
        if download:
            cmd.append("--download")
        if fatura_zip:
            cmd.extend(["--fatura-zip", str(fatura_zip)])
        subprocess.run(cmd, check=True, cwd=str(script.parent.parent))


def _canonicalize_fatura_output(out_dir: str | Path, target_cases: int) -> dict[str, Any]:
    out = Path(out_dir)
    _copy_if_exists(out / "sft/reviewer_extract_sft.jsonl", out / "sft/reviewer_train.jsonl")
    _copy_if_exists(out / "splits/test_locked.jsonl", out / "eval/locked_cases.jsonl")
    _copy_if_exists(out / "case_inputs/reviewer_extract_payloads.jsonl", out / "system_call/reviewer_extract_payloads.jsonl")
    quality_path = out / "validation/quality_report.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8")) if quality_path.exists() else {}
    quality["not_for_final_training"] = False
    quality["strict_split_source_uniqueness_required"] = True
    write_json(quality_path, quality)
    audit = audit_dataset(
        out,
        out / "validation/audit_report.json",
        strict_split_source_uniqueness=True,
        smoke_seed=False,
        min_cases=target_cases,
    )
    if not audit.get("valid"):
        raise RuntimeError(f"strict audit failed for {out}: {audit['validation_errors_sample'][:5]}")
    return {"out": str(out), "source": "fatura", "audit": audit}


def build_dataset(
    source: str,
    out_dir: str | Path,
    target_cases: int = 500,
    repo_root: str | Path | None = None,
    input_zip: str | Path | None = None,
    pipeline_zip: str | Path | None = None,
    download: bool = True,
    fatura_zip: str | Path | None = None,
    smoke_seed: bool = False,
) -> dict[str, Any]:
    if source == "zip-smoke":
        if not input_zip:
            raise ValueError("--input-zip is required for --source zip-smoke")
        return build_from_public_review_zip(input_zip, out_dir, smoke_seed=True)
    if source != "fatura":
        raise ValueError(f"unsupported source: {source}")
    if not repo_root:
        raise ValueError("--repo-root is required for --source fatura")
    pipeline = _find_pipeline_zip(pipeline_zip)
    _run_pipeline_zip(pipeline, repo_root, out_dir, target_cases, download, fatura_zip)
    return _canonicalize_fatura_output(out_dir, target_cases)
