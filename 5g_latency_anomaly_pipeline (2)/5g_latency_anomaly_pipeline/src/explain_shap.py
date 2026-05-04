import argparse
import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch

from src.config import (
    ANOMALY_REPORT_PATH,
    ANOMALY_EXPLANATIONS_PATH,
    DETECTOR_HIDDEN,
    DETECTOR_LAYERS,
    DETECTOR_METADATA_PATH,
    DETECTOR_MODEL_PATH,
    FEATURE_COLS,
    FEATURE_CONTRIBUTIONS_PATH,
    FEATURE_SCALER_PATH,
    SHAP_BACKGROUND_SIZE,
    SHAP_FORCE_PLOT_PATH,
    SHAP_SUMMARY_PLOT_PATH,
    TARGET_SCALER_PATH,
)
from src.dataset import prepare_detection_data
from src.model import SequenceAnomalyScoreWrapper, SequenceProbGRU


DEVICE = torch.device("cpu")


def score_anomaly_batches(wrapper, x_array, batch_size=64):
    scores = []
    with torch.no_grad():
        for start in range(0, len(x_array), batch_size):
            batch = torch.tensor(x_array[start : start + batch_size], dtype=torch.float32, device=DEVICE)
            score = wrapper(batch).squeeze(-1).cpu().numpy()
            scores.append(score)
    return np.concatenate(scores)


def flatten_sequence_names(seq_len):
    names = []
    for timestep in range(seq_len):
        lag = seq_len - timestep - 1
        for feature in FEATURE_COLS:
            names.append(f"t-{lag}_{feature}")
    return names


def compute_ablation_importance(wrapper, x_sequences, background_mean):
    baseline_scores = score_anomaly_batches(wrapper, x_sequences)
    records = []

    for feature_idx, feature_name in enumerate(FEATURE_COLS):
        modified = x_sequences.copy()
        modified[:, :, feature_idx] = background_mean[feature_idx]
        modified_scores = score_anomaly_batches(wrapper, modified)
        score_delta = baseline_scores - modified_scores
        score_reduction = float(np.mean(score_delta))
        records.append(
            {
                "feature": feature_name,
                "ablation_score_reduction": score_reduction,
                "ablation_importance": max(score_reduction, 0.0),
                "ablation_abs_effect": float(np.mean(np.abs(score_delta))),
            }
        )

    return pd.DataFrame(records).sort_values("ablation_importance", ascending=False).reset_index(drop=True)


def build_local_explanations(report_rows, shap_values):
    records = []
    for local_rank, (report_index, row) in enumerate(report_rows.iterrows(), start=1):
        local_importance = np.abs(shap_values[local_rank - 1]).sum(axis=0)
        top_feature_idx = np.argsort(local_importance)[::-1]
        top_features = [FEATURE_COLS[idx] for idx in top_feature_idx[:5]]
        top_values = [float(local_importance[idx]) for idx in top_feature_idx[:5]]
        records.append(
            {
                "anomaly_rank": local_rank,
                "report_index": int(report_index),
                "timestamp": row.get("timestamp", ""),
                "probability_score": float(row.get("probability_score", np.nan)),
                "nll_score": float(row.get("nll_score", np.nan)),
                "nll_zscore": float(row.get("nll_zscore", np.nan)),
                "window_has_injected_anomaly": int(row.get("window_has_injected_anomaly", 0)),
                "top_feature": top_features[0] if top_features else "",
                "top_feature_importance": top_values[0] if top_values else 0.0,
                "top_5_features": "|".join(top_features),
                "top_5_importance": "|".join(f"{value:.6f}" for value in top_values),
            }
        )

    return pd.DataFrame(records)


def save_force_plot(expected_value, shap_flat, x_flat, feature_names, save_path):
    plt.figure(figsize=(18, 4))
    shap.force_plot(
        expected_value,
        shap_flat,
        x_flat,
        feature_names=feature_names,
        matplotlib=True,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_summary_plot(shap_flat, x_flat, feature_names, save_path):
    plt.figure(figsize=(12, 7))
    shap.summary_plot(shap_flat, x_flat, feature_names=feature_names, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def compute_shap_values(wrapper, background_tensor, explain_tensor):
    method = "DeepExplainer"
    expected_value = None

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="unrecognized nn.Module: GRU")
        explainer = shap.DeepExplainer(wrapper, background_tensor)

    try:
        shap_values = explainer.shap_values(explain_tensor)
    except AssertionError as exc:
        print(
            "DeepExplainer additivity check failed for the GRU-based model; "
            "retrying with check_additivity=False."
        )
        shap_values = explainer.shap_values(explain_tensor, check_additivity=False)
        method = "DeepExplainer(check_additivity=False)"
    except Exception as exc:
        print(f"DeepExplainer failed ({type(exc).__name__}); falling back to GradientExplainer.")
        method = "GradientExplainer"
        explainer = shap.GradientExplainer(wrapper, background_tensor)
        shap_values = explainer.shap_values(explain_tensor)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.asarray(shap_values)

    expected_value = getattr(explainer, "expected_value", None)
    if expected_value is None:
        with torch.no_grad():
            expected_value = wrapper(background_tensor).mean().item()
    else:
        expected_value = float(np.asarray(expected_value).reshape(-1)[0])

    return shap_values, expected_value, method


def normalize_shap_tensor(shap_values, reference_shape):
    shap_values = np.asarray(shap_values)

    if shap_values.shape == reference_shape:
        return shap_values

    squeezed = np.squeeze(shap_values)
    if squeezed.shape == reference_shape:
        return squeezed

    if shap_values.ndim == len(reference_shape) + 1 and shap_values.shape[-1] == 1:
        squeezed = shap_values[..., 0]
        if squeezed.shape == reference_shape:
            return squeezed

    if shap_values.ndim == len(reference_shape) + 1 and shap_values.shape[1] == 1:
        squeezed = shap_values[:, 0, ...]
        if squeezed.shape == reference_shape:
            return squeezed

    raise ValueError(
        f"Unexpected SHAP output shape {shap_values.shape}; expected something compatible with {reference_shape}."
    )


def main():
    parser = argparse.ArgumentParser(description="Explain anomalous KPI sequences with SHAP and feature ablation.")
    parser.add_argument("--data", type=Path, required=True, help="Same CSV used for training and detection.")
    parser.add_argument("--top-n", type=int, default=10, help="Number of anomalous sequences to explain.")
    args = parser.parse_args()

    if not ANOMALY_REPORT_PATH.exists():
        raise FileNotFoundError(f"Missing anomaly report at {ANOMALY_REPORT_PATH}. Run src.detect_probabilistic first.")

    metadata = {}
    if DETECTOR_METADATA_PATH.exists():
        with DETECTOR_METADATA_PATH.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

    target_scaler = joblib.load(TARGET_SCALER_PATH)
    feature_scaler = joblib.load(FEATURE_SCALER_PATH)
    _, arrays, _ = prepare_detection_data(
        args.data,
        seq_len=int(metadata.get("sequence_length", 10)),
        fit_scalers=False,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
    )

    report = pd.read_csv(ANOMALY_REPORT_PATH)
    if len(report) != len(arrays["x_test"]):
        raise ValueError("Anomaly report length does not match the current test window count.")

    anomalous_report = report.loc[report["anomaly_label"] == "Anomalous"].sort_values("nll_score", ascending=False)
    if anomalous_report.empty:
        print("No anomalous sequences were detected, so there is nothing to explain.")
        return

    explain_count = min(args.top_n, len(anomalous_report))
    top_indices = anomalous_report.index[:explain_count].to_numpy()
    x_to_explain = arrays["x_test"][top_indices]

    normal_train = arrays["x_train_normal"]
    background_size = min(SHAP_BACKGROUND_SIZE, len(normal_train))
    rng = np.random.default_rng(42)
    background_idx = rng.choice(len(normal_train), size=background_size, replace=False)
    background = normal_train[background_idx]
    background_mean = background.mean(axis=(0, 1))

    detector = SequenceProbGRU(
        input_dim=len(FEATURE_COLS),
        hidden_dim=DETECTOR_HIDDEN,
        num_layers=DETECTOR_LAYERS,
    ).to(DEVICE)
    detector.load_state_dict(torch.load(DETECTOR_MODEL_PATH, map_location=DEVICE))
    detector.eval()

    wrapper = SequenceAnomalyScoreWrapper(detector).to(DEVICE).eval()
    background_tensor = torch.tensor(background, dtype=torch.float32, device=DEVICE)
    explain_tensor = torch.tensor(x_to_explain, dtype=torch.float32, device=DEVICE)

    print(f"Explaining top {explain_count} anomalous sequences with DeepExplainer...")
    shap_values, expected_value, explanation_method = compute_shap_values(
        wrapper=wrapper,
        background_tensor=background_tensor,
        explain_tensor=explain_tensor,
    )
    shap_values = normalize_shap_tensor(shap_values, x_to_explain.shape)

    base_feature_importance = np.abs(shap_values).sum(axis=1).mean(axis=0)
    shap_df = pd.DataFrame(
        {
            "feature": FEATURE_COLS,
            "shap_importance": base_feature_importance,
        }
    ).sort_values("shap_importance", ascending=False)

    ablation_df = compute_ablation_importance(wrapper, x_to_explain, background_mean)
    feature_contributions = shap_df.merge(ablation_df, on="feature", how="inner")
    feature_contributions["shap_rank"] = feature_contributions["shap_importance"].rank(
        ascending=False, method="dense"
    )
    feature_contributions["ablation_rank"] = feature_contributions["ablation_importance"].rank(
        ascending=False, method="dense"
    )
    feature_contributions["combined_rank"] = feature_contributions["shap_rank"] + feature_contributions["ablation_rank"]
    feature_contributions = feature_contributions.sort_values(
        ["combined_rank", "shap_importance", "ablation_importance"], ascending=[True, False, False]
    ).reset_index(drop=True)

    FEATURE_CONTRIBUTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    feature_contributions.to_csv(FEATURE_CONTRIBUTIONS_PATH, index=False)

    seq_len = x_to_explain.shape[1]
    flat_feature_names = flatten_sequence_names(seq_len)
    shap_flat = shap_values.reshape(len(x_to_explain), -1)
    x_flat = x_to_explain.reshape(len(x_to_explain), -1)

    save_force_plot(
        expected_value=expected_value,
        shap_flat=shap_flat[0],
        x_flat=x_flat[0],
        feature_names=flat_feature_names,
        save_path=SHAP_FORCE_PLOT_PATH,
    )
    save_summary_plot(
        shap_flat=shap_flat,
        x_flat=x_flat,
        feature_names=flat_feature_names,
        save_path=SHAP_SUMMARY_PLOT_PATH,
    )

    top_local = np.abs(shap_values[0]).sum(axis=0)
    local_root_causes = pd.DataFrame({"feature": FEATURE_COLS, "local_shap_importance": top_local}).sort_values(
        "local_shap_importance", ascending=False
    )
    local_explanations = build_local_explanations(anomalous_report.iloc[:explain_count], shap_values)
    local_explanations.to_csv(ANOMALY_EXPLANATIONS_PATH, index=False)

    print(f"Saved ranked feature contributions: {FEATURE_CONTRIBUTIONS_PATH}")
    print(f"Saved anomaly explanations     : {ANOMALY_EXPLANATIONS_PATH}")
    print(f"Saved SHAP force plot          : {SHAP_FORCE_PLOT_PATH}")
    print(f"Saved SHAP summary plot        : {SHAP_SUMMARY_PLOT_PATH}")
    print(f"Explanation method used       : {explanation_method}")
    print("\nTop features explaining anomalies:")
    print(feature_contributions.head(10).to_string(index=False))
    print("\nTop features for the most anomalous sequence:")
    print(local_root_causes.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
