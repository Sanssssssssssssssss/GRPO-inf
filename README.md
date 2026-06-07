# GRPO-inf

Clean infrastructure for training the `invoice-case-workbench-openai-sdk`
`evidence_reviewer` with Gemma 4 31B-it + LoRA/QLoRA + TRL GRPO on CSD3 Slurm.

This repo is infra-first. It does not download large models or train locally by
default. It provides the EvidenceReviewResult schema contract, dataset builder,
dataset audit, reward functions, offline evaluation, static Plotly dashboards,
training entrypoints, Slurm launch scripts, and vLLM serving profiles.

## What This Trains

The main output contract is the workbench `EvidenceReviewResult`, not a separate
AP-risk schema. The reviewer must return strict JSON with fields such as:

- `mode`: `extract`, `review`, or `repair`
- `source_doc_id`, `evidence_type`, `credibility`, `source_traceability`
- `extracted_fields` and `extraction_result`
- `support_level`, `risk_flags`, `should_accept`
- `supports`, `conflicts`, `evidence_cards`
- `suggested_patch`, `reply_to_user`

The old `decision/risk_level/findings` schema is retained only as
`ap_risk_ablation`; it is not the default training, reward, eval, or serving
route.

## Dataset Policy

`invoice_reviewer_public_review_500_v2.zip` is useful for schema, reward, and
pipeline smoke tests. It is not final training data because its source documents
repeat across splits. Smoke audits mark it with `not_for_final_training=true`.

Formal public invoice training data must be built with strict split source
uniqueness. If public source coverage is insufficient, the builder fails closed
instead of fabricating cases.

Strict audit requires every GRPO row to carry
`reward_metadata.source.stable_source_id`, `source_dataset`, and
`source_image_sha256`; split uniqueness is checked from those stable identifiers,
not temporary paths, attachment IDs, or filenames.

This public repo commits only code, schemas, configs, and tiny fixtures. Keep
full datasets, zips, checkpoints, adapters, model caches, Slurm logs, and
`outputs/` out of git.

## Quick Start

```powershell
py -3 -m pip install -e ".[dev]"
py -3 -m pytest
```

Print the active schema:

```powershell
py -3 -m grpo_inf.cli print-schema --out schemas/evidence_review_result.schema.json
```

Smoke-import the attached public review package:

```powershell
py -3 -m grpo_inf.cli build-dataset `
  --source zip-smoke `
  --input-zip C:\Users\X\Downloads\invoice_reviewer_public_review_500_v2.zip `
  --out outputs\tmp_public_review_smoke
```

Build the strict public invoice dataset through the pipeline zip:

```powershell
py -3 -m pip install -e ".[data]"
py -3 -m grpo_inf.cli build-dataset `
  --source fatura `
  --target-cases 500 `
  --repo-root ..\invoice-case-workbench-openai-sdk `
  --pipeline-zip C:\path\to\invoice_reviewer_public_invoice_pipeline_v2.zip `
  --out data\invoice_reviewer_public_500
```

Audit modes:

```powershell
# Smoke/dev packages may warn on source overlap.
py -3 -m grpo_inf.cli audit-dataset `
  --data C:\Users\X\Downloads\invoice_reviewer_public_review_500_v2.zip `
  --smoke-seed `
  --min-cases 500

# Formal training/locked eval packages must pass strict source uniqueness.
py -3 -m grpo_inf.cli audit-dataset `
  --data data\invoice_reviewer_public_500 `
  --strict-split-source-uniqueness `
  --min-cases 500
```

Run offline eval and visualization smoke tests:

```powershell
py -3 -m grpo_inf.cli eval-reviewer `
  --samples examples\tiny_dataset\grpo\prompts_test_locked.jsonl `
  --outputs examples\tiny_dataset\outputs\model_outputs.jsonl `
  --summary-out outputs\runs\smoke\eval\summary.json `
  --scored-out outputs\runs\smoke\eval\scored.jsonl

py -3 -m grpo_inf.cli visualize-run --run-dir outputs\runs\smoke
```

Dry-run training configs. These create run folders and manifests but do not load
TRL or a model:

```powershell
py -3 -m grpo_inf.cli train-sft --config configs\training\gemma4_31b_sft.json
py -3 -m grpo_inf.cli train-grpo --config configs\training\gemma4_31b_grpo.json
```

Opt-in tiny training smoke tests may download a tiny HF test model:

```powershell
py -3 -m pip install -e ".[train]"
py -3 -m grpo_inf.cli train-sft --config configs\training\tiny_sft_smoke.json --execute
py -3 -m grpo_inf.cli train-grpo --config configs\training\tiny_grpo_smoke.json --execute
```

## CSD3

Set `DATA_ROOT` to a strict-audited `invoice_reviewer_public_500` dataset on
CSD3. Then submit:

```bash
sbatch infra/slurm/train_sft_csd3_2xa10080.sh
sbatch infra/slurm/train_grpo_csd3_2xa10080.sh
```

The Slurm files have editable `#SBATCH --account` and `#SBATCH --partition`
headers. They use the fixed output layout:

```text
outputs/runs/<run_id>/
  config/
  checkpoints/
  adapter/
  logs/
  generations/
  eval/
  visualizations/
```

For a later high-throughput rollout setup, start a separate vLLM server and use
`configs/training/gemma4_31b_grpo_vllm_server.json`. The default 31B GRPO config
keeps `use_vllm=false` for first CSD3 bring-up on 2x A100.

## Model Routes

- Main training: `google/gemma-4-31B-it` with LoRA/QLoRA and TRL GRPO.
- Low-cost fallback/ablation: `google/gemma-4-12B-it` config only.
- Inference showcase: `google/gemma-4-26B-A4B-it` vLLM TP2/MTP profiles,
  isolated from reviewer GRPO training.

## References

- TRL GRPO Trainer: <https://huggingface.co/docs/trl/grpo_trainer>
- vLLM supported models: <https://docs.vllm.ai/en/stable/models/supported_models/>
- vLLM LoRA serving: <https://docs.vllm.ai/en/latest/features/lora/>
- vLLM Gemma 4 MTP: <https://github.com/vllm-project/vllm/blob/main/docs/features/speculative_decoding/mtp.md>
- vLLM parallelism: <https://docs.vllm.ai/en/latest/serving/parallelism_scaling/>
- Gemma 4 31B model card: <https://huggingface.co/google/gemma-4-31B>
