#!/usr/bin/env bash
#SBATCH --account=mphil-dis-sl2-gpu
#SBATCH --partition=ampere
#SBATCH --job-name=gemma4-12b-sft-audit
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=/rds-d6/user/cx272/hpc-work/outputs/grpo-inf/slurm_logs/%x-%j.out

set -euo pipefail

module purge >/dev/null 2>&1 || true
module load python/3.11.0-icl
module load cuda/12.1

export WORK_ROOT="${WORK_ROOT:-/rds-d6/user/cx272/hpc-work}"
export PROJECT_ROOT="${PROJECT_ROOT:-/home/cx272/final_project/GRPO-inf}"
export VENV="${VENV:-$WORK_ROOT/envs/grpo-inf-py311}"
export HF_HOME="${HF_HOME:-$WORK_ROOT/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_XET_CACHE="${HF_XET_CACHE:-$HF_HOME/xet}"
export MODEL_DIR="${MODEL_DIR:-$WORK_ROOT/models/gemma-4-12B-it}"
export COMPACT_DATA_ROOT="${COMPACT_DATA_ROOT:-$WORK_ROOT/datasets/invoice_reviewer_grpo_strict_evr_720_v1/compact_chat_v1}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$WORK_ROOT/outputs/grpo-inf}"
export TMPDIR="${TMPDIR:-$WORK_ROOT/tmp}"
export CUDA_HOME="${CUDA_HOME:-${CUDA_PATH:-/usr/local/software/cuda/12.1}}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$WORK_ROOT/tmp/triton-cache}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$WORK_ROOT/tmp/torch-extensions}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

if [ -z "${INIT_ADAPTER_PATH:-}" ] && [ -n "${SFT_RUN_ID:-}" ]; then
  export INIT_ADAPTER_PATH="$OUTPUT_ROOT/runs/$SFT_RUN_ID/adapter"
fi
if [ -z "${INIT_ADAPTER_PATH:-}" ]; then
  echo "ERROR: set INIT_ADAPTER_PATH or SFT_RUN_ID." >&2
  exit 2
fi
if [ ! -f "$INIT_ADAPTER_PATH/adapter_model.safetensors" ]; then
  echo "ERROR: missing adapter_model.safetensors under $INIT_ADAPTER_PATH" >&2
  exit 2
fi

RUN_ID="${RUN_ID:-gemma4_12b_sft_audit_${SLURM_JOB_ID:-manual}}"
RUN_ROOT="$OUTPUT_ROOT/runs/$RUN_ID"
METADATA_DIR="$RUN_ROOT/config/external_metadata"
mkdir -p "$METADATA_DIR" "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$OUTPUT_ROOT/slurm_logs"

source "$VENV/bin/activate"
NVIDIA_LIB_PATHS="$(find "$VENV/lib" "$VENV/lib64" -path '*/site-packages/nvidia/*/lib' -type d 2>/dev/null | paste -sd: -)"
export LD_LIBRARY_PATH="${NVIDIA_LIB_PATHS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "run_id=$RUN_ID"
echo "model_dir=$MODEL_DIR"
echo "compact_data_root=$COMPACT_DATA_ROOT"
echo "init_adapter_path=$INIT_ADAPTER_PATH"
date -u
nvidia-smi | tee "$METADATA_DIR/nvidia_smi.txt"
env | sort | grep -Ev '(TOKEN|SECRET|PASSWORD|KEY)=' > "$METADATA_DIR/env_redacted.txt"
git rev-parse HEAD > "$METADATA_DIR/git_revision.txt" || true
git status --short > "$METADATA_DIR/git_status_short.txt" || true
git diff -- . > "$METADATA_DIR/git_diff.patch" || true
cp "$0" "$METADATA_DIR/slurm_script.sh" || true
find "$INIT_ADAPTER_PATH" -maxdepth 1 -type f -printf '%f\t%s\n' | sort > "$METADATA_DIR/init_adapter_files.tsv"

python tools/audit_adapter_generation.py \
  --model-dir "$MODEL_DIR" \
  --adapter-dir "$INIT_ADAPTER_PATH" \
  --data-path "$COMPACT_DATA_ROOT/grpo/prompts_dev.jsonl" \
  --output-jsonl "$RUN_ROOT/eval/dev_generation_outputs.jsonl" \
  --summary-out "$RUN_ROOT/eval/dev_generation_summary.json" \
  --limit "${AUDIT_LIMIT:-16}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-4096}" \
  --min-schema-valid-rate "${MIN_SCHEMA_VALID_RATE:-0.0}" \
  --min-contract-valid-rate "${MIN_CONTRACT_VALID_RATE:-0.0}" \
  --max-clipped-rate "${MAX_CLIPPED_RATE:-1.0}" | tee "$METADATA_DIR/dev_generation_summary.stdout.json"
