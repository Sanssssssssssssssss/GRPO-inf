from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _load_trainer_df(run_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    state_path = run_dir / "checkpoints" / "checkpoint-50" / "trainer_state.json"
    if not state_path.exists():
        candidates = sorted((run_dir / "checkpoints").glob("checkpoint-*/trainer_state.json"))
        if not candidates:
            raise FileNotFoundError(f"No trainer_state.json found under {run_dir / 'checkpoints'}")
        state_path = candidates[-1]
    state = json.loads(state_path.read_text(encoding="utf-8"))
    rows = [row for row in state.get("log_history", []) if "step" in row]
    return pd.DataFrame(rows).sort_values("step"), state


def _load_completions(run_dir: Path) -> tuple[pd.DataFrame, int]:
    paths = sorted((run_dir / "checkpoints" / "completions").glob("*.parquet"))
    if not paths:
        return pd.DataFrame(), 0
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True), len(paths)


def _add_trainer_line(fig: go.Figure, trainer_df: pd.DataFrame, field: str, name: str, row: int, col: int, secondary_y: bool = False) -> None:
    if field in trainer_df:
        fig.add_trace(go.Scatter(x=trainer_df["step"], y=trainer_df[field], name=name, mode="lines+markers"), row=row, col=col, secondary_y=secondary_y)


def render(run_dir: Path, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer_df, state = _load_trainer_df(run_dir)
    completions, completion_file_count = _load_completions(run_dir)
    trainer_df.to_csv(out_dir / "trainer_metrics_by_step.csv", index=False)
    if not completions.empty:
        completions.to_csv(out_dir / "completions_scored_sample.csv", index=False)

    numeric_cols = [
        "reward_total", "reward_schema_valid", "reward_contract_valid", "reward_strict_json_valid",
        "reward_json_valid", "reward_quote_hit_rate", "reward_eos_terminated", "reward_truncated_completion",
        "reward_completion_tokens", "reward_length_penalty", "reward_format_score", "advantage", "_trl_reward",
    ]
    existing_numeric = [col for col in numeric_cols if col in completions.columns]
    if not completions.empty:
        step_summary = completions.groupby("step")[existing_numeric].agg(["mean", "min", "max", "std"])
        step_summary.columns = ["__".join(col).strip("_") for col in step_summary.columns.to_flat_index()]
        step_summary = step_summary.reset_index()
    else:
        step_summary = pd.DataFrame({"step": trainer_df["step"]})
    step_summary.to_csv(out_dir / "completion_reward_summary_by_step.csv", index=False)

    merge_cols = ["step"]
    for col in [
        "reward_total__mean", "reward_schema_valid__mean", "reward_contract_valid__mean",
        "reward_quote_hit_rate__mean", "reward_eos_terminated__mean", "reward_truncated_completion__mean",
        "reward_completion_tokens__mean", "reward_length_penalty__mean", "reward_total__std",
    ]:
        if col in step_summary.columns:
            merge_cols.append(col)
    combined = trainer_df.merge(step_summary[merge_cols], on="step", how="left")
    combined.to_csv(out_dir / "combined_step_metrics.csv", index=False)

    fig = make_subplots(
        rows=4, cols=2,
        specs=[[{"secondary_y": True}, {"secondary_y": False}], [{"secondary_y": True}, {"secondary_y": False}], [{"secondary_y": True}, {"secondary_y": True}], [{"secondary_y": True}, {"secondary_y": False}]],
        subplot_titles=("Reward and reward std", "Contract / schema / quote metrics", "KL and entropy", "Loss and grad norm", "Completion length and clipping", "Completion-level reward means", "Step time and cumulative wall clock", "Reward distribution by step"),
        vertical_spacing=0.08,
    )
    _add_trainer_line(fig, trainer_df, "reward", "trainer reward", 1, 1)
    _add_trainer_line(fig, trainer_df, "reward_std", "reward std", 1, 1, secondary_y=True)
    for field, name in [("reward/schema_valid", "schema valid"), ("reward/contract_valid", "contract valid"), ("reward/quote_hit_rate", "quote hit"), ("reward/eos_terminated", "eos terminated"), ("reward/truncated_completion", "truncated")]:
        _add_trainer_line(fig, trainer_df, field, name, 1, 2)
    _add_trainer_line(fig, trainer_df, "kl", "KL", 2, 1)
    _add_trainer_line(fig, trainer_df, "entropy", "entropy", 2, 1, secondary_y=True)
    _add_trainer_line(fig, trainer_df, "loss", "loss", 2, 2)
    _add_trainer_line(fig, trainer_df, "grad_norm", "grad norm", 2, 2)
    _add_trainer_line(fig, trainer_df, "completions/mean_length", "mean completion length", 3, 1)
    _add_trainer_line(fig, trainer_df, "completions/clipped_ratio", "clipped ratio", 3, 1, secondary_y=True)
    if "reward_total__mean" in combined:
        fig.add_trace(go.Scatter(x=combined["step"], y=combined["reward_total__mean"], name="completion reward mean", mode="lines+markers"), row=3, col=2)
    if "reward_schema_valid__mean" in combined:
        fig.add_trace(go.Scatter(x=combined["step"], y=combined["reward_schema_valid__mean"], name="completion schema mean", mode="lines+markers"), row=3, col=2)
    if "reward_truncated_completion__mean" in combined:
        fig.add_trace(go.Scatter(x=combined["step"], y=combined["reward_truncated_completion__mean"], name="completion truncated mean", mode="lines+markers"), row=3, col=2, secondary_y=True)
    if "step_time" in trainer_df:
        fig.add_trace(go.Scatter(x=trainer_df["step"], y=trainer_df["step_time"], name="step time sec", mode="lines+markers"), row=4, col=1)
        fig.add_trace(go.Scatter(x=trainer_df["step"], y=trainer_df["step_time"].fillna(0).cumsum() / 3600.0, name="cumulative hours", mode="lines+markers"), row=4, col=1, secondary_y=True)
    if not completions.empty and "reward_total" in completions:
        fig.add_trace(go.Box(x=completions["step"], y=completions["reward_total"], name="completion reward", boxpoints="outliers"), row=4, col=2)
    fig.update_layout(title=f"Gemma4 12B GRPO Training Diagnostics: {run_dir.name}", template="plotly_white", height=1600, width=1500, hovermode="x unified")
    fig.write_html(out_dir / "training_diagnostics.html", include_plotlyjs="cdn", full_html=True)

    anomaly_fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=("abs(loss)", "grad_norm", "abs(KL)"))
    for row, field, name in [(1, "loss", "abs(loss)"), (2, "grad_norm", "grad_norm"), (3, "kl", "abs(KL)")]:
        if field in trainer_df:
            values = trainer_df[field].abs() if field in {"loss", "kl"} else trainer_df[field]
            anomaly_fig.add_trace(go.Scatter(x=trainer_df["step"], y=values, mode="lines+markers", name=name), row=row, col=1)
            anomaly_fig.update_yaxes(type="log", row=row, col=1)
    anomaly_fig.update_layout(title="Numerical Anomaly Scan (log scale)", template="plotly_white", height=900, width=1200)
    anomaly_fig.write_html(out_dir / "numerical_anomaly_logscale.html", include_plotlyjs="cdn", full_html=True)

    anomaly_steps = [int(row["step"]) for _, row in trainer_df.iterrows() if abs(float(row.get("loss", 0) or 0)) > 1e6 or abs(float(row.get("kl", 0) or 0)) > 1e6 or abs(float(row.get("grad_norm", 0) or 0)) > 1e6]
    clipped_steps = [int(row["step"]) for _, row in trainer_df.iterrows() if float(row.get("completions/clipped_ratio", 0) or 0) > 0]
    summary = {
        "run_dir": str(run_dir),
        "output_dir": str(out_dir),
        "global_step": int(state.get("global_step", 0)),
        "log_rows": int(len(trainer_df)),
        "completion_rows": int(len(completions)),
        "completion_files": int(completion_file_count),
        "reward_mean_trainer": float(trainer_df["reward"].mean()) if "reward" in trainer_df else None,
        "reward_last": float(trainer_df["reward"].iloc[-1]) if "reward" in trainer_df and len(trainer_df) else None,
        "schema_valid_mean_trainer": float(trainer_df["reward/schema_valid"].mean()) if "reward/schema_valid" in trainer_df else None,
        "contract_valid_mean_trainer": float(trainer_df["reward/contract_valid"].mean()) if "reward/contract_valid" in trainer_df else None,
        "quote_hit_mean_trainer": float(trainer_df["reward/quote_hit_rate"].mean()) if "reward/quote_hit_rate" in trainer_df else None,
        "clipped_ratio_mean_trainer": float(trainer_df["completions/clipped_ratio"].mean()) if "completions/clipped_ratio" in trainer_df else None,
        "truncated_completion_mean_trainer": float(trainer_df["reward/truncated_completion"].mean()) if "reward/truncated_completion" in trainer_df else None,
        "avg_step_time_sec": float(trainer_df["step_time"].mean()) if "step_time" in trainer_df else None,
        "anomaly_steps": anomaly_steps,
        "clipped_nonzero_steps": clipped_steps,
        "completion_reward_mean": float(completions["reward_total"].mean()) if "reward_total" in completions else None,
        "completion_schema_valid_mean": float(completions["reward_schema_valid"].mean()) if "reward_schema_valid" in completions else None,
        "completion_contract_valid_mean": float(completions["reward_contract_valid"].mean()) if "reward_contract_valid" in completions else None,
        "completion_eos_terminated_mean": float(completions["reward_eos_terminated"].mean()) if "reward_eos_terminated" in completions else None,
        "completion_truncated_mean": float(completions["reward_truncated_completion"].mean()) if "reward_truncated_completion" in completions else None,
        "completion_length_penalty_mean": float(completions["reward_length_penalty"].mean()) if "reward_length_penalty" in completions else None,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "README.md").write_text("\n".join(["# Gemma4 12B GRPO Training Diagnostics", "", f"Run: `{run_dir.name}`", "", "Files:", "", "- `training_diagnostics.html`: reward, contract metrics, KL, loss, grad norm, clipping, lengths, and step time.", "- `numerical_anomaly_logscale.html`: log-scale scan for loss, grad norm, and KL.", "- `trainer_metrics_by_step.csv`: raw trainer log history.", "- `completion_reward_summary_by_step.csv`: per-step completion-level reward aggregates.", "- `combined_step_metrics.csv`: joined trainer and completion summary.", "- `completions_scored_sample.csv`: flattened scored completions from parquet.", "- `summary.json`: machine-readable summary.", "", f"anomaly_steps: `{anomaly_steps}`", f"clipped_nonzero_steps: `{clipped_steps}`", f"completion_reward_mean: `{summary['completion_reward_mean']}`", f"completion_schema_valid_mean: `{summary['completion_schema_valid_mean']}`", f"completion_eos_terminated_mean: `{summary['completion_eos_terminated_mean']}`", ""]) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Render GRPO training diagnostics from saved trainer/completion logs.")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    out_dir = args.out_dir or args.run_dir / "visualizations" / "training_diagnostics_50step"
    print(json.dumps(render(args.run_dir, out_dir), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
