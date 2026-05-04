import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from src.config import (
    ANOMALY_REPORT_PATH,
    BATCH_SIZE,
    DETECTOR_DROPOUT,
    DETECTOR_HIDDEN,
    DETECTOR_LAYERS,
    DETECTOR_METADATA_PATH,
    DETECTOR_MODEL_PATH,
    DETECTOR_PROB_QUANTILE,
    DETECTOR_Z_THRESHOLD,
    EPOCHS,
    FEATURE_COLS,
    LEARNING_RATE,
    PREDICTIONS_PATH,
    SEQ_LEN,
    WEIGHT_DECAY,
)
from src.dataset import prepare_detection_data
from src.model import SequenceProbGRU


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def robust_zscore(values, reference_values=None):
    reference = np.asarray(reference_values if reference_values is not None else values, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    median = np.median(reference)
    mad = np.median(np.abs(reference - median))
    mad = max(mad, 1e-8)
    return 0.6745 * (values - median) / mad


def concat_nonempty(arrays):
    arrays = [np.asarray(arr) for arr in arrays if len(arr) > 0]
    if not arrays:
        return np.array([], dtype=np.float32)
    return np.concatenate(arrays)


def binary_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(len(y_true), 1)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "predicted_positive_rate": float(np.mean(y_pred)) if len(y_pred) else 0.0,
    }


def combine_flags(low_probability_flag, robust_zscore_flag, decision_rule):
    if decision_rule == "probability_only":
        return low_probability_flag.astype(int)
    if decision_rule == "zscore_only":
        return robust_zscore_flag.astype(int)
    if decision_rule == "either":
        return (low_probability_flag | robust_zscore_flag).astype(int)
    if decision_rule == "both":
        return (low_probability_flag & robust_zscore_flag).astype(int)
    raise ValueError(f"Unsupported decision rule: {decision_rule}")


def calibrate_thresholds(reference_scores, val_scores, val_labels, default_prob_quantile, default_z_threshold):
    reference_log_prob = reference_scores["log_prob"]
    reference_nll = reference_scores["nll_score"]
    val_nll_zscore = robust_zscore(val_scores["nll_score"], reference_values=reference_nll)

    quantiles = sorted({0.001, 0.005, 0.01, 0.02, default_prob_quantile, 0.05, 0.1})
    z_thresholds = [None, 2.5, 3.0, default_z_threshold, 4.0, 5.0, 6.0]
    decision_rules = ["probability_only", "both", "either", "zscore_only"]

    if len(val_labels) and np.unique(val_labels).size > 1:
        best = None
        for quantile in quantiles:
            log_prob_threshold = float(np.quantile(reference_log_prob, quantile))
            low_prob_flag = (val_scores["log_prob"] < log_prob_threshold).astype(int)

            for z_threshold in z_thresholds:
                z_flag = (
                    (val_nll_zscore > z_threshold).astype(int)
                    if z_threshold is not None
                    else np.zeros_like(low_prob_flag)
                )
                for decision_rule in decision_rules:
                    if z_threshold is None and decision_rule != "probability_only":
                        continue

                    pred = combine_flags(low_prob_flag, z_flag, decision_rule)
                    metrics = binary_metrics(val_labels, pred)
                    candidate = {
                        "probability_quantile": float(quantile),
                        "log_probability_threshold": log_prob_threshold,
                        "robust_zscore_threshold": None if z_threshold is None else float(z_threshold),
                        "decision_rule": decision_rule,
                        "metrics": metrics,
                    }

                    ranking = (
                        metrics["f1"],
                        metrics["precision"],
                        metrics["recall"],
                        -abs(metrics["predicted_positive_rate"] - float(np.mean(val_labels))),
                    )
                    if best is None or ranking > best["ranking"]:
                        best = {"ranking": ranking, **candidate}
        return best

    return {
        "probability_quantile": float(default_prob_quantile),
        "log_probability_threshold": float(np.quantile(reference_log_prob, default_prob_quantile)),
        "robust_zscore_threshold": float(default_z_threshold),
        "decision_rule": "both",
        "metrics": None,
    }


def run_epoch(model, loader, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    total_nll = 0.0
    total_count = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for (x_batch,) in loader:
            x_batch = x_batch.to(DEVICE, non_blocking=True)
            if train:
                optimizer.zero_grad()

            nll, _ = model.negative_log_likelihood(x_batch)
            loss = nll.mean()

            if train:
                loss.backward()
                optimizer.step()

            batch_size = x_batch.size(0)
            total_nll += float(loss.item()) * batch_size
            total_count += batch_size

    return total_nll / max(total_count, 1)


def score_sequences(model, x_array, batch_size=256):
    model.eval()
    if len(x_array) == 0:
        return {
            "log_prob": np.array([], dtype=np.float32),
            "probability_score": np.array([], dtype=np.float32),
            "nll_score": np.array([], dtype=np.float32),
        }

    log_probs = []
    probability_scores = []
    nll_scores = []

    with torch.no_grad():
        for start in range(0, len(x_array), batch_size):
            batch = torch.tensor(x_array[start : start + batch_size], dtype=torch.float32, device=DEVICE)
            batch_log_prob = model.sequence_log_probability(batch)
            batch_prob = model.probability_score(batch)
            log_probs.append(batch_log_prob.cpu().numpy())
            probability_scores.append(batch_prob.cpu().numpy())
            nll_scores.append((-batch_log_prob).cpu().numpy())

    return {
        "log_prob": np.concatenate(log_probs),
        "probability_score": np.concatenate(probability_scores),
        "nll_score": np.concatenate(nll_scores),
    }


def build_report(
    df_split,
    source_index,
    window_labels,
    scores,
    log_prob_threshold,
    z_threshold,
    nll_reference,
    decision_rule,
):
    report = df_split.iloc[source_index].copy().reset_index(drop=True)
    report["sequence_log_probability"] = scores["log_prob"]
    report["probability_score"] = scores["probability_score"]
    report["nll_score"] = scores["nll_score"]
    report["nll_zscore"] = robust_zscore(scores["nll_score"], reference_values=nll_reference)
    report["low_probability_flag"] = (report["sequence_log_probability"] < log_prob_threshold).astype(int)
    if z_threshold is None:
        report["robust_zscore_flag"] = 0
    else:
        report["robust_zscore_flag"] = (report["nll_zscore"] > z_threshold).astype(int)
    report["window_has_injected_anomaly"] = window_labels.astype(int)
    combined_flag = combine_flags(
        report["low_probability_flag"].to_numpy(dtype=int),
        report["robust_zscore_flag"].to_numpy(dtype=int),
        decision_rule,
    )
    report["combined_flag"] = combined_flag
    report["decision_rule"] = decision_rule
    report["anomaly_label"] = np.where(report["combined_flag"] == 1, "Anomalous", "Normal")
    return report


def attach_prediction_columns(report):
    if not PREDICTIONS_PATH.exists():
        return report

    predictions_df = pd.read_csv(PREDICTIONS_PATH)
    if predictions_df.empty:
        return report

    merge_keys = ["timestamp", "slice_id", "end_to_end_latency"]
    missing_keys = [col for col in merge_keys if col not in report.columns or col not in predictions_df.columns]
    if missing_keys:
        return report

    prediction_cols = [col for col in ["predicted_latency_ms", "actual_latency_ms", "squared_error_norm"] if col in predictions_df.columns]
    if not prediction_cols:
        return report

    merged = report.merge(
        predictions_df[merge_keys + prediction_cols],
        on=merge_keys,
        how="left",
    )
    return merged


def main():
    parser = argparse.ArgumentParser(description="Train a probabilistic GRU on normal KPI sequences and detect anomalies.")
    parser.add_argument("--data", type=Path, required=True, help="CSV path generated by src.data_generation.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--prob-quantile", type=float, default=DETECTOR_PROB_QUANTILE)
    parser.add_argument("--z-threshold", type=float, default=DETECTOR_Z_THRESHOLD)
    args = parser.parse_args()

    loaders, arrays, _ = prepare_detection_data(args.data, seq_len=SEQ_LEN, batch_size=args.batch_size, fit_scalers=True)
    input_dim = len(FEATURE_COLS)

    print(f"Device: {DEVICE}")
    print(f"Normal train windows: {len(arrays['x_train_normal'])}")
    print(f"Validation windows  : {len(arrays['x_val'])}")
    print(f"Test windows        : {len(arrays['x_test'])}")

    model = SequenceProbGRU(
        input_dim=input_dim,
        hidden_dim=DETECTOR_HIDDEN,
        num_layers=DETECTOR_LAYERS,
        dropout=DETECTOR_DROPOUT,
    ).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_val_nll = float("inf")
    patience_counter = 0
    early_stop_patience = 10
    DETECTOR_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    val_loader_for_early_stop = loaders["val_normal"] if len(arrays["x_val_normal"]) > 0 else loaders["val"]

    print(f"\n{'Epoch':>5} {'Train NLL':>12} {'Val NLL':>12}")
    print("-" * 33)
    for epoch in range(1, args.epochs + 1):
        train_nll = run_epoch(model, loaders["train_normal"], optimizer=optimizer)
        val_nll = run_epoch(model, val_loader_for_early_stop, optimizer=None)
        print(f"{epoch:>5} {train_nll:>12.4f} {val_nll:>12.4f}")

        if val_nll < best_val_nll:
            best_val_nll = val_nll
            patience_counter = 0
            torch.save(model.state_dict(), DETECTOR_MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    print("\nLoading best detector checkpoint...")
    model.load_state_dict(torch.load(DETECTOR_MODEL_PATH, map_location=DEVICE))

    train_scores = score_sequences(model, arrays["x_train_normal"])
    val_normal_scores = score_sequences(model, arrays["x_val_normal"])
    val_scores = score_sequences(model, arrays["x_val"])
    test_scores = score_sequences(model, arrays["x_test"])
    reference_scores = {
        "log_prob": concat_nonempty([train_scores["log_prob"], val_normal_scores["log_prob"]]),
        "nll_score": concat_nonempty([train_scores["nll_score"], val_normal_scores["nll_score"]]),
    }
    calibration = calibrate_thresholds(
        reference_scores=reference_scores,
        val_scores=val_scores,
        val_labels=arrays["val_window_labels"],
        default_prob_quantile=args.prob_quantile,
        default_z_threshold=args.z_threshold,
    )

    report = build_report(
        arrays["df_test"],
        arrays["test_source_index"],
        arrays["test_window_labels"],
        test_scores,
        log_prob_threshold=calibration["log_probability_threshold"],
        z_threshold=calibration["robust_zscore_threshold"],
        nll_reference=reference_scores["nll_score"],
        decision_rule=calibration["decision_rule"],
    )
    report = attach_prediction_columns(report)

    ANOMALY_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(ANOMALY_REPORT_PATH, index=False)

    test_metrics = None
    if len(arrays["test_window_labels"]) and np.unique(arrays["test_window_labels"]).size > 1:
        test_metrics = binary_metrics(
            arrays["test_window_labels"],
            (report["anomaly_label"] == "Anomalous").astype(int).to_numpy(),
        )

    metadata = {
        "probability_quantile": calibration["probability_quantile"],
        "log_probability_threshold": calibration["log_probability_threshold"],
        "robust_zscore_threshold": calibration["robust_zscore_threshold"],
        "decision_rule": calibration["decision_rule"],
        "normal_train_windows": int(len(arrays["x_train_normal"])),
        "normal_val_windows": int(len(arrays["x_val_normal"])),
        "input_dim": input_dim,
        "sequence_length": int(arrays["x_test"].shape[1]),
        "validation_metrics": calibration["metrics"],
        "test_metrics": test_metrics,
    }
    with DETECTOR_METADATA_PATH.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    anomalous = report.loc[report["anomaly_label"] == "Anomalous"].sort_values("nll_score", ascending=False)

    print(f"\nSaved anomaly report: {ANOMALY_REPORT_PATH}")
    print(f"Saved detector meta : {DETECTOR_METADATA_PATH}")
    print(f"Detected anomalies  : {len(anomalous)} / {len(report)}")
    print(
        "Calibration         : "
        f"rule={calibration['decision_rule']}, "
        f"prob_q={calibration['probability_quantile']}, "
        f"z={calibration['robust_zscore_threshold']}"
    )

    if calibration["metrics"] is not None:
        print(
            "Validation metrics  : "
            f"precision={calibration['metrics']['precision']:.3f}, "
            f"recall={calibration['metrics']['recall']:.3f}, "
            f"f1={calibration['metrics']['f1']:.3f}"
        )
    if test_metrics is not None:
        print(
            "Test metrics        : "
            f"precision={test_metrics['precision']:.3f}, "
            f"recall={test_metrics['recall']:.3f}, "
            f"f1={test_metrics['f1']:.3f}"
        )

    if not anomalous.empty:
        preview_cols = [
            "timestamp",
            "probability_score",
            "sequence_log_probability",
            "nll_zscore",
            "combined_flag",
            "window_has_injected_anomaly",
            "anomaly_label",
        ]
        print("\nTop anomalous sequences:")
        print(anomalous[preview_cols].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
