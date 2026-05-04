from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
REPORTS_DIR = ROOT_DIR / "reports"

DOMAINS = ["RAN1", "Core1", "Edge1", "RAN2", "Core2", "Edge2"]
METRICS = ["cpu", "memory", "disk", "bandwidth"]
FEATURE_COLS = [f"{domain}_{metric}" for domain in DOMAINS for metric in METRICS]
TARGET_COL = "end_to_end_latency"

NUM_NODES = len(DOMAINS)
NODE_FEATURES = len(METRICS)
SEQ_LEN = 10
BATCH_SIZE = 64
EPOCHS =    100
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 5e-4
GCN_HIDDEN = 12
GRU_HIDDEN = 200
GRU_LAYERS = 1
GCN_DROPOUT = 0.2
FC_DROPOUT = 0.3
GRAD_CLIP_NORM = 1.0
SCHEDULER_PATIENCE = 2
SCHEDULER_FACTOR = 0.5
EARLY_STOP_PATIENCE = 10
DETECTOR_HIDDEN = 128
DETECTOR_LAYERS = 2
DETECTOR_DROPOUT = 0.2
DETECTOR_PROB_QUANTILE = 0.05
DETECTOR_Z_THRESHOLD = 3.5
SHAP_BACKGROUND_SIZE = 64

MODEL_PATH = ARTIFACTS_DIR / "best_model.pt"
DETECTOR_MODEL_PATH = ARTIFACTS_DIR / "sequence_prob_gru.pt"
DETECTOR_METADATA_PATH = ARTIFACTS_DIR / "sequence_prob_metadata.json"
FEATURE_SCALER_PATH = ARTIFACTS_DIR / "feature_scaler.pkl"
TARGET_SCALER_PATH = ARTIFACTS_DIR / "target_scaler.pkl"
PREDICTIONS_PATH = REPORTS_DIR / "test_predictions.csv"
TRAIN_HISTORY_PATH = REPORTS_DIR / "train_history.csv"
ANOMALY_REPORT_PATH = REPORTS_DIR / "anomaly_report.csv"
FEATURE_CONTRIBUTIONS_PATH = REPORTS_DIR / "feature_contributions.csv"
ANOMALY_EXPLANATIONS_PATH = REPORTS_DIR / "anomaly_explanations.csv"
SHAP_FORCE_PLOT_PATH = REPORTS_DIR / "shap_force_top_anomaly.png"
SHAP_SUMMARY_PLOT_PATH = REPORTS_DIR / "shap_summary_top_anomalies.png"
