from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from grpo_inf.rewards.reviewer_reward import score_sample_completion
from grpo_inf.schema import schema_errors


SPLITS = ("train", "dev", "test_locked")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _stats(values: list[int]) -> dict[str, int | float]:
    ordered = sorted(values)

    def percentile(q: float) -> int:
        return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]

    return {
        "n": len(ordered),
        "min": ordered[0],
        "p50": statistics.median(ordered),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": ordered[-1],
    }


def _gold_answer(row: dict[str, Any]) -> str:
    gold = row.get("gold")
    if not isinstance(gold, dict):
        raise ValueError(f"{row.get('case_id', '<unknown>')} is missing dict gold")
    errors = schema_errors(gold)
    if errors:
        raise ValueError(f"{row.get('case_id', '<unknown>')} has schema-invalid gold: {errors[:3]}")
    return json.dumps(gold, ensure_ascii=False, separators=(",", ":"))


def _messages_from_row(row: dict[str, Any], answer: str) -> dict[str, Any]:
    prompt = row.get("prompt")
    if not isinstance(prompt, list) or not all(isinstance(item, dict) for item in prompt):
        raise ValueError(f"{row.get('case_id', '<unknown>')} is not a conversational prompt row")
    return {
        "case_id": row.get("case_id"),
        "split": row.get("split"),
        "scenario": row.get("scenario"),
        "prompt": prompt,
        "completion": [{"role": "assistant", "content": answer}],
        "gold": row.get("gold"),
        "documents": row.get("documents", []),
        "input": row.get("input", {}),
        "reward_metadata": row.get("reward_metadata", {}),
    }


def build(source_root: Path, output_root: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "format": "trl_conversational_prompt_completion",
        "splits": {},
    }
    all_answer_chars: list[int] = []
    all_scores: list[float] = []

    for split in SPLITS:
        source = source_root / "grpo" / f"prompts_{split}.jsonl"
        if not source.exists():
            continue
        rows = _read_jsonl(source)
        out_rows: list[dict[str, Any]] = []
        answer_chars: list[int] = []
        self_scores: list[float] = []
        schema_valid = 0
        quote_hits = 0
        mode_counts: dict[str, int] = {}
        for row in rows:
            answer = _gold_answer(row)
            out_rows.append(_messages_from_row(row, answer))
            answer_chars.append(len(answer))
            score = score_sample_completion(answer, row)
            self_scores.append(float(score["total"]))
            schema_valid += int(score.get("schema_valid") == 1.0)
            quote_hits += int(score.get("quote_hit_rate") == 1.0)
            mode = str(row.get("gold", {}).get("mode"))
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
        _write_jsonl(output_root / "sft" / f"reviewer_{split}.jsonl", out_rows)
        split_report = {
            "cases": len(rows),
            "mode_counts": mode_counts,
            "answer_chars": _stats(answer_chars),
            "exact_gold_total_mean": sum(self_scores) / len(self_scores),
            "exact_gold_total_min": min(self_scores),
            "exact_gold_schema_valid_rate": schema_valid / len(rows),
            "exact_gold_quote_hit_rate": quote_hits / len(rows),
        }
        report["splits"][split] = split_report
        all_answer_chars.extend(answer_chars)
        all_scores.extend(self_scores)

    report["overall"] = {
        "cases": len(all_answer_chars),
        "answer_chars": _stats(all_answer_chars),
        "exact_gold_total_mean": sum(all_scores) / len(all_scores),
        "exact_gold_total_min": min(all_scores),
    }
    _write_json(output_root / "sft_compact_dataset_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build compact conversational SFT prompt-completion files.")
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.source_root, args.output_root), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
