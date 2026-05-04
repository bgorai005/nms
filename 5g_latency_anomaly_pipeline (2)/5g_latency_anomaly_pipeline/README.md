# 5G Slice Latency Anomaly Detection Pipeline

This project builds an end-to-end pipeline for synthetic 5G slice latency data generation, graph-based latency prediction, probabilistic sequence anomaly detection, and SHAP-based KPI root-cause analysis.

## Project Structure

```text
5g_latency_anomaly_pipeline/
├── data/
│   ├── raw/
│   └── processed/
├── artifacts/
├── reports/
├── src/
│   ├── config.py
│   ├── data_generation.py
│   ├── dataset.py
│   ├── detect_probabilistic.py
│   ├── explain_shap.py
│   ├── model.py
│   └── train.py
├── requirements.txt
└── README.md
```

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

PyTorch Geometric sometimes needs a version-specific install command. If `torch-geometric` fails to import, install it using the official PyTorch Geometric wheel command for your local PyTorch/CUDA version.

## Generate Data

Run one of the three generation strategies:

```bash
python -m src.data_generation --strategy 1 --days 10
python -m src.data_generation --strategy 2 --days 10
python -m src.data_generation --strategy 3 --days 10
```

The generated CSV files are saved in `data/raw/`.

Each dataset contains:

- KPI features for all six domains: `RAN1`, `Core1`, `Edge1`, `RAN2`, `Core2`, `Edge2`
- `end_to_end_latency`
- `is_anomaly`
- `anomaly_domain`

## Part A: Train the Graph-Based Predictor

Train on any generated dataset:

```bash
python -m src.train --data data/raw/strategy1_data.csv --epochs 10
python -m src.train --data data/raw/strategy2_data.csv --epochs 10
python -m src.train --data data/raw/strategy3_data.csv --epochs 10
```

Artifacts are saved in `artifacts/`:

- `best_model.pt`
- `feature_scaler.pkl`
- `target_scaler.pkl`

Test predictions are saved in `reports/test_predictions.csv`.

## Part B: Train the Probabilistic Detector

Train the sequence detector on historical normal behavior and score the held-out test windows:

```bash
python -m src.detect_probabilistic --data data/raw/strategy3_data.csv --epochs 10
```

Outputs:

- `artifacts/sequence_prob_gru.pt`
- `artifacts/sequence_prob_metadata.json`
- `reports/anomaly_report.csv`

The anomaly report includes:

- `sequence_log_probability`
- `probability_score`
- `nll_zscore`
- `anomaly_label`

Detection follows the framework:

- GRU learns `P(x_i | x_1:i-1)` over normalized KPI sequences
- the detector is trained only on windows labeled normal in the training split
- low sequence probability marks anomalous windows
- robust median/MAD z-scores are included as an optional second flag

## Part C: Explain Anomalies

After running the detector, explain the most anomalous sequences:

```bash
python -m src.explain_shap --data data/raw/strategy3_data.csv
```

Outputs:

- `reports/feature_contributions.csv`
- `reports/shap_force_top_anomaly.png`
- `reports/shap_summary_top_anomalies.png`

## Notes

- The predictor uses `Input -> GCN -> GRU -> Fully Connected -> latency`.
- The probabilistic detector uses a separate GRU with Gaussian conditional outputs to estimate sequence likelihoods.
- SHAP explanations are computed against the anomaly score itself, and feature ablation is used alongside SHAP to rank root-cause KPIs.
