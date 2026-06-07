# GRPO-inf

Clean infrastructure for training an accounts-payable evidence reviewer with
Gemma 4 31B-it + LoRA/QLoRA + TRL GRPO on CSD3 Slurm.

The first version is intentionally infra-first: it does not download models or
train locally by default. It provides the reward function, dataset audit,
offline evaluation, run visualization, training entrypoints, Slurm scripts, and
serving profiles needed to move the job to CSD3.

## What This Trains

The reviewer reads OCR evidence and returns strict JSON only:

- `decision`: `approve`, `hold`, `reject`, or `escalate`
- `risk_level`: `low`, `medium`, `high`, or `critical`
- `findings`: typed findings with valid `source_ids` and exact
  `evidence_quotes`
- `missing_evidence`, `unsupported_items`, `confidence`

The reward is verifiable. It scores JSON/schema validity, finding F1,
decision/risk accuracy, source grounding, exact quote hits, action consistency,
and concision. It penalizes hallucinated evidence, bad source IDs, unsafe
approvals, prompt-injection following, markdown fences, and thought leakage.

## Dataset Policy

The seed zip `invoice_reviewer_grpo_dataset_v0.zip` is useful for v0 smoke
training and reward development, but it is not a final locked evaluation set.
The audit detects vendor/template overlap across splits in that zip, so scores
from it should be treated as development signals only.

This public repo commits only small fixtures, schemas, scripts, and reports.
Keep full datasets, checkpoints, adapters, and model caches outside git.

## Quick Start

```powershell
py -3 -m pip install -e ".[dev]"
py -3 -m pytest
```

Print the output schema:

```powershell
py -3 -m grpo_inf.cli print-schema --out schemas/reviewer_answer.schema.json
```

Audit the attached zip without extracting it into the repo:

```powershell
$env:INVOICE_REVIEWER_DATASET_ZIP = "D:\path\to\invoice_reviewer_grpo_dataset_v0.zip"
py -3 -m grpo_inf.cli audit-dataset `
  --data $env:INVOICE_REVIEWER_DATASET_ZIP `
  --out outputs/audit/invoice_reviewer_grpo_dataset_v0.audit.json
```

Run an offline eval smoke test:

```powershell
py -3 -m grpo_inf.cli eval-reviewer `
  --samples examples\tiny_dataset\grpo\prompts_test_locked.jsonl `
  --outputs examples\tiny_dataset\outputs\model_outputs.jsonl `
  --summary-out outputs\runs\smoke\eval\summary.json `
  --scored-out outputs\runs\smoke\eval\scored.jsonl

py -3 -m grpo_inf.cli visualize-run --run-dir outputs\runs\smoke
```

Dry-run training configs. These create run folders and config manifests, but do
not import TRL or load a model:

```powershell
py -3 -m grpo_inf.cli train-sft --config configs\training\gemma4_31b_sft.json
py -3 -m grpo_inf.cli train-grpo --config configs\training\gemma4_31b_grpo.json
```

## CSD3

Set `DATA_ROOT` to the extracted external dataset directory on CSD3. Then submit:

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

## Model Routes

- Main training: `google/gemma-4-31B-it` with LoRA/QLoRA and TRL GRPO.
- Low-cost fallback/ablation: `google/gemma-4-12B-it` config only.
- Inference showcase: `google/gemma-4-26B-A4B-it` vLLM TP2/MTP profiles, isolated
  from reviewer GRPO training.

## References

- TRL GRPO Trainer: <https://huggingface.co/docs/trl/grpo_trainer>
- vLLM supported models: <https://docs.vllm.ai/en/stable/models/supported_models/>
- vLLM LoRA serving: <https://docs.vllm.ai/en/latest/features/lora/>
- vLLM Gemma 4 MTP: <https://github.com/vllm-project/vllm/blob/main/docs/features/speculative_decoding/mtp.md>
- vLLM parallelism: <https://docs.vllm.ai/en/latest/serving/parallelism_scaling/>
- Gemma 4 31B model card: <https://huggingface.co/google/gemma-4-31B>
