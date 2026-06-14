from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from transformers import AutoProcessor

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


def _render_prompt(processor: Any, row: dict[str, Any]) -> str:
    prompt = row.get("prompt")
    if not isinstance(prompt, list) or not all(isinstance(item, dict) for item in prompt):
        raise ValueError(f"{row.get('case_id', '<unknown>')} is not a conversational prompt row")
    rendered = processor.apply_chat_template(
        prompt,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not isinstance(rendered, str):
        raise TypeError("Expected rendered chat template string")
    return rendered


def build(source_root: Path, output_root: Path, model_dir: Path) -> dict[str, Any]:
    processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True)
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    report: dict[str, Any] = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "model_dir": str(model_dir),
        "format": "standard_rendered_prompt_completion",
        "completion_suffix": "<turn|>\\n",
        "splits": {},
    }
    all_prompt_tokens: list[int] = []
    all_completion_tokens: list[int] = []
    all_total_tokens: list[int] = []
    all_scores: list[float] = []

    for split in SPLITS:
        source = source_root / "grpo" / f"prompts_{split}.jsonl"
        if not source.exists():
            continue
        rows = _read_jsonl(source)
        out_rows: list[dict[str, Any]] = []
        prompt_tokens: list[int] = []
        completion_tokens: list[int] = []
        total_tokens: list[int] = []
        self_scores: list[float] = []
        mode_counts: dict[str, int] = {}
        for row in rows:
            prompt = _render_prompt(processor, row)
            answer = _gold_answer(row)
            completion = answer + "<turn|>\n"
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            completion_ids = tokenizer.encode(completion, add_special_tokens=False)
            total_ids = tokenizer.encode(prompt + completion, add_special_tokens=False)
            if total_ids[: len(prompt_ids)] != prompt_ids:
                raise ValueError(f"{row.get('case_id', '<unknown>')} rendered prompt is not a token prefix")
            out_rows.append(
                {
                    "case_id": row.get("case_id"),
                    "split": row.get("split"),
                    "scenario": row.get("scenario"),
                    "prompt": prompt,
                    "completion": completion,
                    "gold": row.get("gold"),
                    "documents": row.get("documents", []),
                    "input": row.get("input", {}),
                    "reward_metadata": row.get("reward_metadata", {}),
                }
            )
            prompt_tokens.append(len(prompt_ids))
            completion_tokens.append(len(completion_ids))
            total_tokens.append(len(total_ids))
            score = score_sample_completion(answer, row)
            self_scores.append(float(score["total"]))
            mode = str(row.get("gold", {}).get("mode"))
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
        _write_jsonl(output_root / "sft" / f"reviewer_{split}.jsonl", out_rows)
        split_report = {
            "cases": len(rows),
            "mode_counts": mode_counts,
            "prompt_tokens": _stats(prompt_tokens),
            "completion_tokens": _stats(completion_tokens),
            "total_tokens": _stats(total_tokens),
            "exact_gold_total_mean": sum(self_scores) / len(self_scores),
            "exact_gold_total_min": min(self_scores),
        }
        report["splits"][split] = split_report
        all_prompt_tokens.extend(prompt_tokens)
        all_completion_tokens.extend(completion_tokens)
        all_total_tokens.extend(total_tokens)
        all_scores.extend(self_scores)

    report["overall"] = {
        "cases": len(all_total_tokens),
        "prompt_tokens": _stats(all_prompt_tokens),
        "completion_tokens": _stats(all_completion_tokens),
        "total_tokens": _stats(all_total_tokens),
        "exact_gold_total_mean": sum(all_scores) / len(all_scores),
        "exact_gold_total_min": min(all_scores),
    }
    _write_json(output_root / "sft_rendered_dataset_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build rendered standard SFT prompt-completion files for Gemma4.")
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.source_root, args.output_root, args.model_dir), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
