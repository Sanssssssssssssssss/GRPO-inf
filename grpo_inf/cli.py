from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from grpo_inf.data.audit import audit_dataset
from grpo_inf.evaluation.evaluate import evaluate_outputs
from grpo_inf.schema import REVIEWER_ANSWER_SCHEMA
from grpo_inf.training.grpo import run_grpo
from grpo_inf.training.sft import run_sft
from grpo_inf.viz.dashboard import render_dashboard


def _print_json(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="grpo-inf", description="Invoice reviewer GRPO infrastructure")
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit-dataset", help="Audit reviewer JSONL dataset or zip")
    audit.add_argument("--data", required=True, help="Dataset directory or zip")
    audit.add_argument("--out", help="Optional JSON report path")

    eval_cmd = sub.add_parser("eval-reviewer", help="Score model outputs against reviewer oracle")
    eval_cmd.add_argument("--samples", required=True, help="GRPO samples JSONL")
    eval_cmd.add_argument("--outputs", required=True, help="Outputs JSONL with case_id and completion")
    eval_cmd.add_argument("--summary-out", help="Summary JSON path")
    eval_cmd.add_argument("--scored-out", help="Per-case scored JSONL path")

    viz = sub.add_parser("visualize-run", help="Render static run dashboard")
    viz.add_argument("--run-dir", required=True, help="Run directory")
    viz.add_argument("--out", help="Optional HTML output path")

    sft = sub.add_parser("train-sft", help="Validate or execute SFT warmup")
    sft.add_argument("--config", required=True)
    sft.add_argument("--run-id")
    sft.add_argument("--execute", action="store_true", help="Actually import training stack and train")

    grpo = sub.add_parser("train-grpo", help="Validate or execute GRPO training")
    grpo.add_argument("--config", required=True)
    grpo.add_argument("--run-id")
    grpo.add_argument("--execute", action="store_true", help="Actually import training stack and train")

    schema = sub.add_parser("print-schema", help="Print reviewer answer JSON schema")
    schema.add_argument("--out", help="Optional schema output path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "audit-dataset":
        _print_json(audit_dataset(args.data, args.out))
    elif args.command == "eval-reviewer":
        _print_json(evaluate_outputs(args.samples, args.outputs, args.summary_out, args.scored_out))
    elif args.command == "visualize-run":
        path = render_dashboard(args.run_dir, args.out)
        _print_json({"dashboard": str(path)})
    elif args.command == "train-sft":
        _print_json(run_sft(args.config, args.run_id, args.execute))
    elif args.command == "train-grpo":
        _print_json(run_grpo(args.config, args.run_id, args.execute))
    elif args.command == "print-schema":
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(REVIEWER_ANSWER_SCHEMA, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            _print_json({"schema": args.out})
        else:
            _print_json(REVIEWER_ANSWER_SCHEMA)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
