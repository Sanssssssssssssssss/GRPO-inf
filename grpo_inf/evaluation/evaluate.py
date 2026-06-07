from __future__ import annotations

from pathlib import Path
from typing import Any

from grpo_inf.io import read_jsonl, write_json, write_jsonl
from grpo_inf.rewards.reviewer_reward import score_sample_completion


METRIC_KEYS = (
    "total",
    "json_valid",
    "schema_valid",
    "contract_valid",
    "mode_correct",
    "source_doc_valid",
    "extract_field_value_score",
    "extract_present_field_f1",
    "extract_missing_field_f1",
    "support_level_correct",
    "should_accept_correct",
    "risk_flag_f1",
    "support_f1",
    "conflict_f1",
    "quote_hit_rate",
    "forbidden_patch_rate",
    "thought_leak",
    "markdown_fence",
)


def _case_id(row: dict[str, Any]) -> str:
    return str(row.get("case_id") or row.get("id") or "")


def _completion(row: dict[str, Any]) -> Any:
    for key in ("completion", "output", "response", "answer"):
        if key in row:
            return row[key]
    return ""


def _mode(row: dict[str, Any]) -> str:
    gold = row.get("gold") or row.get("answer") or row.get("expected_answer") or row.get("oracle") or {}
    if isinstance(gold, dict) and gold.get("mode"):
        return str(gold["mode"])
    payload = row.get("input") if isinstance(row.get("input"), dict) else {}
    return str(payload.get("mode") or "unknown")


def _category(row: dict[str, Any]) -> str:
    return str(row.get("scenario") or row.get("category") or _mode(row) or "unknown")


def evaluate_outputs(
    samples_path: str | Path,
    outputs_path: str | Path,
    summary_path: str | Path | None = None,
    scored_path: str | Path | None = None,
) -> dict[str, Any]:
    samples = {_case_id(row): row for row in read_jsonl(Path(samples_path))}
    outputs = read_jsonl(Path(outputs_path))
    scored_rows: list[dict[str, Any]] = []
    missing_outputs: list[str] = []
    unknown_outputs: list[str] = []

    seen: set[str] = set()
    for output in outputs:
        cid = _case_id(output)
        if cid not in samples:
            unknown_outputs.append(cid)
            continue
        seen.add(cid)
        sample = samples[cid]
        score = score_sample_completion(_completion(output), sample)
        scored_rows.append(
            {
                "case_id": cid,
                "mode": _mode(sample),
                "category": _category(sample),
                **score,
            }
        )

    for cid in samples:
        if cid not in seen:
            missing_outputs.append(cid)

    if not scored_rows:
        summary = {"error": "no scored outputs", "missing_outputs": missing_outputs[:20], "unknown_outputs": unknown_outputs[:20]}
        if summary_path:
            write_json(Path(summary_path), summary)
        return summary

    summary: dict[str, Any] = {"n": len(scored_rows)}
    for key in METRIC_KEYS:
        values = [float(row.get(key, 0.0)) for row in scored_rows]
        summary[key] = sum(values) / len(values)
    summary.update(
        {
            "json_valid_rate": summary["json_valid"],
            "schema_valid_rate": summary["schema_valid"],
            "contract_valid_rate": summary["contract_valid"],
            "mode_accuracy": summary["mode_correct"],
            "source_doc_valid_rate": summary["source_doc_valid"],
            "quote_hit_rate": summary["quote_hit_rate"],
            "extract_field_f1": summary["extract_present_field_f1"],
            "review_support_f1": summary["support_f1"],
            "review_conflict_f1": summary["conflict_f1"],
            "review_risk_flag_f1": summary["risk_flag_f1"],
            "unsupported_claim_rate": 1.0 - summary["source_doc_valid"],
            "integration_consumable_rate": summary["schema_valid"],
            "missing_outputs": missing_outputs[:20],
            "unknown_outputs": unknown_outputs[:20],
        }
    )
    by_category: dict[str, dict[str, Any]] = {}
    for category in sorted({row["category"] for row in scored_rows}):
        rows = [row for row in scored_rows if row["category"] == category]
        by_category[category] = {
            "n": len(rows),
            "reward": sum(float(row["total"]) for row in rows) / len(rows),
            "schema_valid_rate": sum(float(row["schema_valid"]) for row in rows) / len(rows),
            "quote_hit_rate": sum(float(row["quote_hit_rate"]) for row in rows) / len(rows),
            "support_f1": sum(float(row.get("support_f1", 0.0)) for row in rows) / len(rows),
            "conflict_f1": sum(float(row.get("conflict_f1", 0.0)) for row in rows) / len(rows),
        }
    summary["by_category"] = by_category
    by_mode: dict[str, dict[str, Any]] = {}
    for mode in sorted({row["mode"] for row in scored_rows}):
        rows = [row for row in scored_rows if row["mode"] == mode]
        by_mode[mode] = {
            "n": len(rows),
            "reward": sum(float(row["total"]) for row in rows) / len(rows),
            "schema_valid_rate": sum(float(row["schema_valid"]) for row in rows) / len(rows),
            "quote_hit_rate": sum(float(row["quote_hit_rate"]) for row in rows) / len(rows),
        }
    summary["by_mode"] = by_mode
    summary["worst_cases"] = [
        {"case_id": row["case_id"], "mode": row["mode"], "category": row["category"], "total": row["total"], "errors": row.get("errors", [])}
        for row in sorted(scored_rows, key=lambda item: float(item["total"]))[:10]
    ]

    if scored_path:
        write_jsonl(Path(scored_path), scored_rows)
    if summary_path:
        write_json(Path(summary_path), summary)
    return summary
