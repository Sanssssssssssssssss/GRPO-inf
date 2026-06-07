from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from grpo_inf.data.audit import audit_dataset
from grpo_inf.data.build_dataset import build_dataset
from grpo_inf.evaluation.evaluate import evaluate_outputs
from grpo_inf.schema import REVIEWER_ANSWER_SCHEMA, write_schema
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
    audit.add_argument("--schema", default="evidence_review_result", choices=["evidence_review_result", "ap_risk_ablation"])
    audit.add_argument("--strict-split-source-uniqueness", action="store_true")
    audit.add_argument("--smoke-seed", action="store_true")
    audit.add_argument("--min-cases", type=int)

    build = sub.add_parser("build-dataset", help="Build or normalize EvidenceReviewResult reviewer datasets")
    build.add_argument("--source", required=True, choices=["fatura", "zip-smoke"])
    build.add_argument("--target-cases", type=int, default=500)
    build.add_argument("--repo-root", help="invoice-case-workbench-openai-sdk checkout for Fatura builds")
    build.add_argument("--out", required=True)
    build.add_argument("--input-zip", help="public_review_500_v2 zip for zip-smoke imports")
    build.add_argument("--pipeline-zip", help="public invoice pipeline zip; required unless PUBLIC_INVOICE_PIPELINE_ZIP is set")
    build.add_argument("--fatura-zip", help="Existing FATURA.zip path")
    build.add_argument("--no-download", action="store_true", help="Do not download FATURA inside the pipeline")

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
        _print_json(
            audit_dataset(
                args.data,
                args.out,
                schema_name=args.schema,
                strict_split_source_uniqueness=args.strict_split_source_uniqueness,
                smoke_seed=args.smoke_seed,
                min_cases=args.min_cases,
            )
        )
    elif args.command == "build-dataset":
        _print_json(
            build_dataset(
                source=args.source,
                out_dir=args.out,
                target_cases=args.target_cases,
                repo_root=args.repo_root,
                input_zip=args.input_zip,
                pipeline_zip=args.pipeline_zip,
                download=not args.no_download,
                fatura_zip=args.fatura_zip,
            )
        )
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
            write_schema(args.out)
            _print_json({"schema": args.out})
        else:
            _print_json(REVIEWER_ANSWER_SCHEMA)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
