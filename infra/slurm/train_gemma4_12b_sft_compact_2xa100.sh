#!/usr/bin/env bash
#SBATCH --account=mphil-dis-sl2-gpu
#SBATCH --partition=ampere
#SBATCH --job-name=gemma4-12b-sft-compact
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_a100:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=220G
#SBATCH --time=04:00:00
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
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export MODEL_DIR="${MODEL_DIR:-$WORK_ROOT/models/gemma-4-12B-it}"
export DATA_ROOT="${DATA_ROOT:-$WORK_ROOT/datasets/invoice_reviewer_grpo_strict_evr_720_v1/extracted/invoice_reviewer_grpo_strict_evr_720_v1}"
export DATA_ZIP="${DATA_ZIP:-$WORK_ROOT/datasets/invoice_reviewer_grpo_strict_evr_720_v1/invoice_reviewer_grpo_strict_evr_720_v1.zip}"
export COMPACT_DATA_ROOT="${COMPACT_DATA_ROOT:-$WORK_ROOT/datasets/invoice_reviewer_grpo_strict_evr_720_v1/compact_chat_v1}"
export COMPACT_SFT_DATA_ROOT="${COMPACT_SFT_DATA_ROOT:-$WORK_ROOT/datasets/invoice_reviewer_grpo_strict_evr_720_v1/compact_sft_chat_v1}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$WORK_ROOT/outputs/grpo-inf}"
export TMPDIR="${TMPDIR:-$WORK_ROOT/tmp}"
export CUDA_HOME="${CUDA_HOME:-${CUDA_PATH:-/usr/local/software/cuda/12.1}}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$WORK_ROOT/tmp/triton-cache}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$WORK_ROOT/tmp/torch-extensions}"
export DS_SKIP_CUDA_CHECK="${DS_SKIP_CUDA_CHECK:-1}"
export DS_ACCELERATOR="${DS_ACCELERATOR:-cuda}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export LORA_PREFLIGHT_TIMEOUT_SECONDS="${LORA_PREFLIGHT_TIMEOUT_SECONDS:-600}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

RUN_ID="${RUN_ID:-gemma4_12b_sft_compact_canary_${SLURM_JOB_ID:-manual}}"
CONFIG_PATH="${CONFIG_PATH:-configs/training/gemma4_12b_sft_compact_canary.json}"
RUN_ROOT="$OUTPUT_ROOT/runs/$RUN_ID"
METADATA_DIR="$RUN_ROOT/config/external_metadata"

mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_XET_CACHE" "$HF_DATASETS_CACHE" \
  "$TRANSFORMERS_CACHE" "$OUTPUT_ROOT/slurm_logs" "$TMPDIR" "$TRITON_CACHE_DIR" \
  "$TORCH_EXTENSIONS_DIR" "$METADATA_DIR" "$COMPACT_SFT_DATA_ROOT"

source "$VENV/bin/activate"
NVIDIA_LIB_PATHS="$(find "$VENV/lib" "$VENV/lib64" -path '*/site-packages/nvidia/*/lib' -type d 2>/dev/null | paste -sd: -)"
export LD_LIBRARY_PATH="${NVIDIA_LIB_PATHS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "run_id=$RUN_ID"
echo "host=$(hostname)"
echo "project_root=$PROJECT_ROOT"
echo "model_dir=$MODEL_DIR"
echo "compact_data_root=$COMPACT_DATA_ROOT"
echo "compact_sft_data_root=$COMPACT_SFT_DATA_ROOT"
echo "output_root=$OUTPUT_ROOT"
echo "metadata_dir=$METADATA_DIR"
date -u

python -V | tee "$METADATA_DIR/python_version.txt"
python -m pip --version | tee "$METADATA_DIR/pip_version.txt"
python -m pip check | tee "$METADATA_DIR/pip_check.txt"
python -m pip freeze > "$METADATA_DIR/pip_freeze.txt"
nvidia-smi | tee "$METADATA_DIR/nvidia_smi.txt"
env | sort | grep -Ev '(TOKEN|SECRET|PASSWORD|KEY)=' > "$METADATA_DIR/env_redacted.txt"

git rev-parse HEAD > "$METADATA_DIR/git_revision.txt" || true
git status --short > "$METADATA_DIR/git_status_short.txt" || true
git diff -- . > "$METADATA_DIR/git_diff.patch" || true
git ls-files --others --exclude-standard > "$METADATA_DIR/git_untracked_files.txt" || true
cp "$CONFIG_PATH" "$METADATA_DIR/train_config.json"
cp "$0" "$METADATA_DIR/slurm_script.sh" || true
cp infra/slurm/accelerate_zero3_2xa100.yaml "$METADATA_DIR/accelerate_zero3_2xa100.yaml"

sha256sum "$DATA_ZIP" | tee "$METADATA_DIR/dataset_zip.sha256"
cp "$DATA_ROOT/README.md" "$METADATA_DIR/dataset_README.md"
cp "$DATA_ROOT/manifests/dataset_manifest.json" "$METADATA_DIR/dataset_manifest.json"
cp "$DATA_ROOT/validation/audit_report.json" "$METADATA_DIR/release_audit_report.json"
cp "$DATA_ROOT/validation/gold_self_score_report.json" "$METADATA_DIR/release_gold_self_score_report.json"
find "$MODEL_DIR" -maxdepth 1 -type f -printf '%f\t%s\n' | sort > "$METADATA_DIR/model_files.tsv"

python tools/build_compact_sft_dataset.py \
  --source-root "$COMPACT_DATA_ROOT" \
  --output-root "$COMPACT_SFT_DATA_ROOT" | tee "$METADATA_DIR/compact_sft_build_report.json"
cp "$COMPACT_SFT_DATA_ROOT/sft_compact_dataset_report.json" "$METADATA_DIR/sft_compact_dataset_report.json"
find "$COMPACT_SFT_DATA_ROOT/sft" -maxdepth 1 -type f -printf '%f\t%s\n' | sort > "$METADATA_DIR/compact_sft_files.tsv"

python -m grpo_inf.cli audit-dataset \
  --data "$COMPACT_DATA_ROOT" \
  --out "$METADATA_DIR/repo_audit_compact.json" \
  --min-cases 500 | tee "$METADATA_DIR/repo_audit_compact.stdout.json"
timeout "$LORA_PREFLIGHT_TIMEOUT_SECONDS" python tools/validate_gemma4_lora_targets.py \
  --config "$CONFIG_PATH" \
  --model-dir "$MODEL_DIR" | tee "$METADATA_DIR/gemma4_lora_preflight.json"

python -m grpo_inf.cli train-sft \
  --config "$CONFIG_PATH" \
  --run-id "$RUN_ID" | tee "$METADATA_DIR/dry_run_summary.json"

accelerate launch \
  --config_file infra/slurm/accelerate_zero3_2xa100.yaml \
  grpo_inf/cli.py train-sft \
  --config "$CONFIG_PATH" \
  --run-id "$RUN_ID" \
  --execute
