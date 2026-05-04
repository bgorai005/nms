"""
Plot paper-style figures from a predictions or anomaly report CSV.

Usage:
    python -m src.plot --input reports/test_predictions.csv
    python -m src.plot --input reports/anomaly_report.csv

Outputs (saved next to the CSV):
    fig5_actual_vs_predicted.png  — predicted vs actual latency
    fig5_zscore_anomalies.png     — z-score anomaly plot
    fig6_shap_proxy.png           — feature-importance proxy
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

# ── Matplotlib style ────────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "figure.dpi": 150,
    }
)

COLORS = {
    "actual": "#1f77b4",
    "predicted": "#d62728",
    "normal": "#aec7e8",
    "anomaly": "#ff0000",
    "bar_pos": "#d62728",
    "bar_neg": "#1f77b4",
}


# ── Z-score (Eq. 7 from paper) ───────────────────────────────────────────────
def robust_zscore(errors: np.ndarray) -> np.ndarray:
    median = np.median(errors)
    mad = np.median(np.abs(errors - median))
    mad = max(mad, 1e-8)
    return 0.6745 * (errors - median) / mad


def attach_prediction_columns(df: pd.DataFrame, input_path: Path) -> pd.DataFrame:
    if "predicted_latency_ms" in df.columns and "actual_latency_ms" in df.columns:
        return df

    predictions_path = input_path.parent / "test_predictions.csv"
    if not predictions_path.exists():
        return df

    predictions_df = pd.read_csv(predictions_path)
    merge_keys = ["timestamp", "slice_id", "end_to_end_latency"]
    missing_keys = [col for col in merge_keys if col not in df.columns or col not in predictions_df.columns]
    if missing_keys:
        return df

    prediction_cols = [col for col in ["predicted_latency_ms", "actual_latency_ms", "squared_error_norm"] if col in predictions_df.columns]
    if not prediction_cols:
        return df

    merged = df.merge(
        predictions_df[merge_keys + prediction_cols],
        on=merge_keys,
        how="left",
    )
    return merged


def ensure_latency_columns(df: pd.DataFrame, input_path: Path | None = None) -> pd.DataFrame:
    df = df.copy()
    if input_path is not None:
        df = attach_prediction_columns(df, input_path)
    if "actual_latency_ms" not in df.columns and "end_to_end_latency" in df.columns:
        df["actual_latency_ms"] = df["end_to_end_latency"]
    if "predicted_latency_ms" not in df.columns:
        raise ValueError(
            "Input CSV must contain 'predicted_latency_ms'. "
            "Run src.train first, and if using anomaly_report.csv rerun src.detect_probabilistic afterward."
        )
    if "actual_latency_ms" not in df.columns:
        raise ValueError("Input CSV must contain either 'actual_latency_ms' or 'end_to_end_latency'.")
    return df


def compute_anomaly_mask(df: pd.DataFrame, z_threshold: float) -> tuple[np.ndarray, np.ndarray]:
    errors = df["squared_error_norm"].values if "squared_error_norm" in df.columns else (
        (df["predicted_latency_ms"].values - df["actual_latency_ms"].values) ** 2
    )
    zscores = robust_zscore(errors)

    if "combined_flag" in df.columns:
        anomaly_mask = df["combined_flag"].fillna(0).astype(int).values.astype(bool)
    elif "anomaly_label" in df.columns:
        anomaly_mask = (df["anomaly_label"].astype(str).str.lower() == "anomalous").values
    else:
        anomaly_mask = zscores > z_threshold

    return anomaly_mask, zscores


# ── Figure 5 — Actual vs Predicted ──────────────────────────────────────────
def plot_actual_vs_predicted(df: pd.DataFrame, save_path: Path) -> None:
    df = ensure_latency_columns(df, save_path.parent / "anomaly_report.csv")
    actual = df["actual_latency_ms"].values
    predicted = df["predicted_latency_ms"].values
    x = np.arange(len(actual))

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x, actual, label="Actual", color=COLORS["actual"], linewidth=0.9, alpha=0.85)
    ax.plot(x, predicted, label="Predicted", color=COLORS["predicted"], linewidth=0.9,
            linestyle="--", alpha=0.85)
    ax.set_xlabel("Data Points")
    ax.set_ylabel("Slice Latency (ms)")
    ax.set_title("Graph GCN with GRU — Actual vs Predicted Slice Latency")
    ax.legend(loc="upper right", framealpha=0.85)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {save_path}")


# ── Figure 5 (extended) — Z-score anomaly plot ──────────────────────────────
def plot_zscore_anomalies(
    df: pd.DataFrame, save_path: Path, z_threshold: float = 5.0
) -> np.ndarray:
    df = ensure_latency_columns(df, save_path.parent / "anomaly_report.csv")
    anomaly_mask, zscores = compute_anomaly_mask(df, z_threshold)
    x = np.arange(len(zscores))

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                              gridspec_kw={"height_ratios": [2, 1]})

    # Top: actual vs predicted with anomalies highlighted
    ax0 = axes[0]
    ax0.plot(x, df["actual_latency_ms"].values, label="Actual",
             color=COLORS["actual"], linewidth=0.8, alpha=0.8)
    ax0.plot(x, df["predicted_latency_ms"].values, label="Predicted",
             color=COLORS["predicted"], linewidth=0.8, linestyle="--", alpha=0.8)
    if anomaly_mask.any():
        ax0.scatter(x[anomaly_mask], df["actual_latency_ms"].values[anomaly_mask],
                    color=COLORS["anomaly"], s=18, zorder=5, label="Detected anomaly")
    ax0.set_ylabel("Latency (ms)")
    ax0.set_title("Anomaly Detection — Predicted vs Actual with Z-score Flags")
    ax0.legend(loc="upper right", framealpha=0.85)

    # Bottom: Z-score bar
    ax1 = axes[1]
    bar_colors = [COLORS["anomaly"] if z > z_threshold else COLORS["normal"] for z in zscores]
    ax1.bar(x, zscores, color=bar_colors, width=1.0, linewidth=0)
    ax1.axhline(z_threshold, color="black", linewidth=1.2, linestyle="--",
                label=f"Threshold = {z_threshold}")
    ax1.set_xlabel("Data Points")
    ax1.set_ylabel("Robust Z-score")
    ax1.legend(loc="upper right", framealpha=0.85)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {save_path}")
    print(f"Anomalies detected: {anomaly_mask.sum()} / {len(anomaly_mask)} "
          f"({anomaly_mask.mean() * 100:.2f}%)")
    return anomaly_mask


# ── Figure 6 — Feature importance proxy ─────────────────────────────────────
def plot_feature_importance_proxy(
    df: pd.DataFrame, anomaly_mask: np.ndarray, save_path: Path
) -> None:
    """
    Approximates Fig. 6 SHAP bar chart by computing mean absolute
    deviation per domain KPI between anomalous and normal samples.
    Only uses columns available in the predictions CSV.
    """
    domain_cols = [c for c in df.columns if any(
        c.endswith(f"_{kpi}") for kpi in ["cpu", "memory", "disk", "bandwidth"]
    )]

    if len(domain_cols) == 0:
        print("No domain KPI columns found in CSV — skipping Fig. 6 proxy plot.")
        print("(Columns present:", list(df.columns), ")")
        return

    normal_df = df.loc[~anomaly_mask, domain_cols]
    anomaly_df = df.loc[anomaly_mask, domain_cols]

    if anomaly_df.empty:
        print("No anomalies found at current threshold — skipping Fig. 6 proxy plot.")
        return

    importance = (anomaly_df.mean() - normal_df.mean()).abs().sort_values(ascending=False)
    top = importance.head(10)

    values = top.values
    names = top.index.tolist()
    colors = [COLORS["bar_pos"] if v >= 0 else COLORS["bar_neg"]
              for v in (anomaly_df[top.index].mean() - normal_df[top.index].mean()).values]

    fig, ax = plt.subplots(figsize=(10, max(3, len(names) * 0.45)))
    ax.barh(names[::-1], values[::-1], color=colors[::-1],
            edgecolor="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=1.0)
    ax.set_xlabel("Mean deviation (anomaly − normal)")
    ax.set_title("Top KPI Contributors to Anomalies (SHAP proxy)\nsliceLatency KPI")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {save_path}")


# ── Multi-panel summary (all 3 in one figure) ────────────────────────────────
def plot_summary_panel(df: pd.DataFrame, anomaly_mask: np.ndarray, save_path: Path) -> None:
    df = ensure_latency_columns(df, save_path.parent / "anomaly_report.csv")
    actual = df["actual_latency_ms"].values
    predicted = df["predicted_latency_ms"].values
    _, zscores = compute_anomaly_mask(df, z_threshold=5.0)
    x = np.arange(len(actual))

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(3, 1, hspace=0.45)

    # Panel 1: Actual vs Predicted
    ax0 = fig.add_subplot(gs[0])
    ax0.plot(x, actual, label="Actual", color=COLORS["actual"], linewidth=0.8)
    ax0.plot(x, predicted, label="Predicted", color=COLORS["predicted"],
             linewidth=0.8, linestyle="--")
    ax0.set_ylabel("Latency (ms)")
    ax0.set_title("Graph GCN + GRU: Actual vs Predicted sliceLatency KPI")
    ax0.legend(loc="upper right", fontsize=8)

    # Panel 2: Anomaly overlay
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax1.plot(x, actual, color=COLORS["actual"], linewidth=0.8, alpha=0.7)
    ax1.plot(x, predicted, color=COLORS["predicted"], linewidth=0.8,
             linestyle="--", alpha=0.7)
    if anomaly_mask.any():
        ax1.scatter(x[anomaly_mask], actual[anomaly_mask],
                    color=COLORS["anomaly"], s=16, zorder=5, label="Detected anomaly")
        ax1.legend(loc="upper right", fontsize=8)
    ax1.set_ylabel("Latency (ms)")
    ax1.set_title("Candidate Anomaly Detection")

    # Panel 3: Z-score
    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    bar_colors = [COLORS["anomaly"] if z > 5 else COLORS["normal"] for z in zscores]
    ax2.bar(x, zscores, color=bar_colors, width=1.0, linewidth=0)
    ax2.axhline(5.0, color="black", linewidth=1.2, linestyle="--", label="Z = 5")
    ax2.set_xlabel("Data Points")
    ax2.set_ylabel("Robust Z-score")
    ax2.set_title("Z-score of Prediction Errors (Anomaly Flags)")
    ax2.legend(loc="upper right", fontsize=8)

    fig.suptitle("G5IAD Framework — Evaluation Results", fontsize=13, fontweight="bold", y=1.01)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ── Entry point ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Generate predicted-vs-actual and z-score anomaly plots.")
    parser.add_argument("--input", type=Path, required=True,
                        help="Path to test_predictions.csv or anomaly_report.csv")
    parser.add_argument("--z-threshold", type=float, default=5.0,
                        help="Robust Z-score threshold for anomaly flag (paper default: 5)")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df = ensure_latency_columns(df, args.input)
    out_dir = args.input.parent
    print(f"Loaded {len(df):,} test samples from {args.input}")
    print(f"Columns: {list(df.columns)}\n")

    # Fig 5a — prediction curve
    plot_actual_vs_predicted(df, out_dir / "fig5_actual_vs_predicted.png")

    # Fig 5b + Z-score panel
    anomaly_mask = plot_zscore_anomalies(
        df, out_dir / "fig5_zscore_anomalies.png", z_threshold=args.z_threshold
    )

    # Fig 6 — feature importance proxy
    plot_feature_importance_proxy(df, anomaly_mask, out_dir / "fig6_shap_proxy.png")

    # All-in-one summary panel
    plot_summary_panel(df, anomaly_mask, out_dir / "summary_panel.png")

    print("\nDone. All figures saved to:", out_dir)


if __name__ == "__main__":
    main()
