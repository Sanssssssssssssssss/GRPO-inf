from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from grpo_inf.io import read_jsonl, write_json, write_jsonl
from grpo_inf.rewards.reviewer_reward import score_completion


METRIC_KEYS = (
    "total",
    "json_valid",
    "schema_valid",
    "finding_f1_reward",
    "decision_correct",
    "risk_correct",
    "quote_hit_rate",
    "unsafe_approval",
    "prompt_injection_failure",
    "bad_source_count",
    "thought_leak",
    "markdown_fence",
)


def _case_id(row: dict[str, Any]) -> str:
    return str(row.get("case_id") or row.get("id") or "")


def _completion(row: dict[str, Any]) -> Any:
    if "completion" in row:
        return row["completion"]
    if "output" in row:
        return row["output"]
    if "response" in row:
        return row["response"]
    return ""


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
        oracle = sample.get("oracle") or sample.get("reviewer_oracle") or {}
        score = score_completion(_completion(output), oracle, sample.get("documents", []))
        scored_rows.append(
            {
                "case_id": cid,
                "category": sample.get("category", "unknown"),
                "difficulty": sample.get("difficulty", "unknown"),
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
            "finding_macro_f1": summary["finding_f1_reward"],
            "decision_accuracy": summary["decision_correct"],
            "risk_accuracy": summary["risk_correct"],
            "unsupported_claim_rate": sum(1 for row in scored_rows if row.get("bad_source_count", 0) > 0) / len(scored_rows),
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
            "finding_f1": sum(float(row["finding_f1_reward"]) for row in rows) / len(rows),
            "decision_accuracy": sum(float(row["decision_correct"]) for row in rows) / len(rows),
            "quote_hit_rate": sum(float(row["quote_hit_rate"]) for row in rows) / len(rows),
        }
    summary["by_category"] = by_category
    summary["worst_cases"] = [
        {"case_id": row["case_id"], "category": row["category"], "total": row["total"], "errors": row.get("errors", [])}
        for row in sorted(scored_rows, key=lambda item: float(item["total"]))[:10]
    ]

    if scored_path:
        write_jsonl(Path(scored_path), scored_rows)
    if summary_path:
        write_json(Path(summary_path), summary)
    return summary
