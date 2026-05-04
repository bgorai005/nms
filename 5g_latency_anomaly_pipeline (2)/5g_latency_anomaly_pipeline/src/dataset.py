import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.config import (
    BATCH_SIZE,
    FEATURE_COLS,
    FEATURE_SCALER_PATH,
    TARGET_COL,
    TARGET_SCALER_PATH,
)


def validate_columns(df):
    missing = [col for col in FEATURE_COLS + [TARGET_COL] if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")


def split_time_series(df, train_ratio=0.70, val_ratio=0.15):
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return (
        df.iloc[:train_end].reset_index(drop=True),
        df.iloc[train_end:val_end].reset_index(drop=True),
        df.iloc[val_end:].reset_index(drop=True),
    )


def load_and_split_dataframe(data_path, train_ratio=0.70, val_ratio=0.15):
    df = pd.read_csv(data_path)
    validate_columns(df)
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL]).reset_index(drop=True)
    df_train, df_val, df_test = split_time_series(df, train_ratio=train_ratio, val_ratio=val_ratio)
    return df, df_train, df_val, df_test


def fit_and_save_scalers(df_train):
    FEATURE_SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    feature_scaler.fit(df_train[FEATURE_COLS].values)
    target_scaler.fit(df_train[[TARGET_COL]].values)
    joblib.dump(feature_scaler, FEATURE_SCALER_PATH)
    joblib.dump(target_scaler, TARGET_SCALER_PATH)
    return feature_scaler, target_scaler


def normalize_df(df_split, feature_scaler, target_scaler):
    df_out = df_split.copy()
    df_out[FEATURE_COLS] = feature_scaler.transform(df_split[FEATURE_COLS].values)
    df_out[TARGET_COL] = target_scaler.transform(df_split[[TARGET_COL]].values)
    return df_out


def row_to_graph(row, domains):
    return np.array(
        [
            [
                row[f"{domain}_cpu"],
                row[f"{domain}_memory"],
                row[f"{domain}_disk"],
                row[f"{domain}_bandwidth"],
            ]
            for domain in domains
        ],
        dtype=np.float32,
    )


def create_sequences(df_norm, domains, seq_len):
    all_graphs = np.stack([row_to_graph(df_norm.iloc[i], domains) for i in range(len(df_norm))])
    x = np.stack([all_graphs[i : i + seq_len] for i in range(len(df_norm) - seq_len)])
    y = df_norm[TARGET_COL].values[seq_len:].astype(np.float32)
    source_index = np.arange(seq_len, len(df_norm))
    return x, y, source_index


def create_feature_sequences(df_norm, seq_len, feature_cols=FEATURE_COLS):
    features = df_norm[feature_cols].values.astype(np.float32)
    if len(features) < seq_len:
        raise ValueError(f"Need at least {seq_len} rows to create feature sequences.")
    x = np.stack([features[i : i + seq_len] for i in range(len(features) - seq_len + 1)])
    source_index = np.arange(seq_len - 1, len(df_norm))
    return x, source_index


def create_sequence_labels(df_split, seq_len, label_col="is_anomaly"):
    if label_col not in df_split.columns:
        return np.zeros(max(len(df_split) - seq_len + 1, 0), dtype=np.int64)

    labels = df_split[label_col].fillna(0).astype(int).values
    if len(labels) < seq_len:
        return np.zeros(0, dtype=np.int64)
    return np.array([int(labels[i : i + seq_len].max()) for i in range(len(labels) - seq_len + 1)], dtype=np.int64)


def make_loader(x, y, batch_size=BATCH_SIZE, shuffle=False):
    dataset = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=torch.cuda.is_available())


def make_sequence_loader(x, batch_size=BATCH_SIZE, shuffle=False):
    dataset = TensorDataset(torch.tensor(x, dtype=torch.float32))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=torch.cuda.is_available())


def prepare_data(
    data_path,
    domains,
    seq_len,
    batch_size=BATCH_SIZE,
    fit_scalers=True,
    feature_scaler=None,
    target_scaler=None,
):
    _, df_train, df_val, df_test = load_and_split_dataframe(data_path)
    if fit_scalers:
        feature_scaler, target_scaler = fit_and_save_scalers(df_train)
    elif feature_scaler is None or target_scaler is None:
        raise ValueError("feature_scaler and target_scaler are required when fit_scalers=False.")

    train_norm = normalize_df(df_train, feature_scaler, target_scaler)
    val_norm = normalize_df(df_val, feature_scaler, target_scaler)
    test_norm = normalize_df(df_test, feature_scaler, target_scaler)

    x_train, y_train, train_idx = create_sequences(train_norm, domains, seq_len)
    x_val, y_val, val_idx = create_sequences(val_norm, domains, seq_len)
    x_test, y_test, test_idx = create_sequences(test_norm, domains, seq_len)

    loaders = {
        "train": make_loader(x_train, y_train, batch_size, shuffle=True),
        "val": make_loader(x_val, y_val, batch_size, shuffle=False),
        "test": make_loader(x_test, y_test, batch_size, shuffle=False),
    }
    arrays = {
        "df_train": df_train,
        "df_val": df_val,
        "x_test": x_test,
        "y_test": y_test,
        "test_source_index": test_idx,
        "df_test": df_test,
    }
    scalers = {"feature": feature_scaler, "target": target_scaler}
    return loaders, arrays, scalers


def prepare_detection_data(
    data_path,
    seq_len,
    batch_size=BATCH_SIZE,
    fit_scalers=True,
    feature_scaler=None,
    target_scaler=None,
):
    _, df_train, df_val, df_test = load_and_split_dataframe(data_path)

    normal_train_df = df_train
    if "is_anomaly" in df_train.columns:
        candidate = df_train.loc[df_train["is_anomaly"].fillna(0).astype(int) == 0]
        if not candidate.empty:
            normal_train_df = candidate

    if fit_scalers:
        feature_scaler, target_scaler = fit_and_save_scalers(normal_train_df)
    elif feature_scaler is None or target_scaler is None:
        raise ValueError("feature_scaler and target_scaler are required when fit_scalers=False.")

    train_norm = normalize_df(df_train, feature_scaler, target_scaler)
    val_norm = normalize_df(df_val, feature_scaler, target_scaler)
    test_norm = normalize_df(df_test, feature_scaler, target_scaler)

    x_train_all, train_source_idx = create_feature_sequences(train_norm, seq_len)
    x_val, val_source_idx = create_feature_sequences(val_norm, seq_len)
    x_test, test_source_idx = create_feature_sequences(test_norm, seq_len)

    train_window_labels = create_sequence_labels(df_train, seq_len)
    val_window_labels = create_sequence_labels(df_val, seq_len)
    test_window_labels = create_sequence_labels(df_test, seq_len)

    if len(train_window_labels) != len(x_train_all):
        raise ValueError("Training sequence windows and labels are misaligned.")

    normal_window_mask = train_window_labels == 0
    x_train_normal = x_train_all[normal_window_mask] if normal_window_mask.any() else x_train_all
    normal_train_source_idx = train_source_idx[normal_window_mask] if normal_window_mask.any() else train_source_idx
    val_normal_window_mask = val_window_labels == 0
    x_val_normal = x_val[val_normal_window_mask] if val_normal_window_mask.any() else x_val[:0]
    val_normal_source_idx = val_source_idx[val_normal_window_mask] if val_normal_window_mask.any() else val_source_idx[:0]

    loaders = {
        "train_normal": make_sequence_loader(x_train_normal, batch_size=batch_size, shuffle=True),
        "val": make_sequence_loader(x_val, batch_size=batch_size, shuffle=False),
        "val_normal": make_sequence_loader(x_val_normal, batch_size=batch_size, shuffle=False),
        "test": make_sequence_loader(x_test, batch_size=batch_size, shuffle=False),
    }
    arrays = {
        "df_train": df_train,
        "df_val": df_val,
        "df_test": df_test,
        "x_train_normal": x_train_normal,
        "x_train_all": x_train_all,
        "x_val": x_val,
        "x_val_normal": x_val_normal,
        "x_test": x_test,
        "train_source_index": train_source_idx,
        "train_normal_source_index": normal_train_source_idx,
        "val_source_index": val_source_idx,
        "val_normal_source_index": val_normal_source_idx,
        "test_source_index": test_source_idx,
        "train_window_labels": train_window_labels,
        "val_window_labels": val_window_labels,
        "test_window_labels": test_window_labels,
    }
    scalers = {"feature": feature_scaler, "target": target_scaler}
    return loaders, arrays, scalers
