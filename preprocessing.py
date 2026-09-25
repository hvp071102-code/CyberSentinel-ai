"""
preprocessing.py
-----------------
Loads raw CICIDS2017 CSV files (or the synthetic sample), cleans them, and
reduces them to the forward-only / unidirectional feature set defined in
config.py.

Handles the well-known CICIDS2017 data-quality issues:
- Column names have inconsistent leading/trailing whitespace
- Infinity values in Flow Bytes/s, Flow Packets/s (division by ~0 duration)
- NaN rows
- Mixed dtypes when multiple day-CSVs are concatenated
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from config import (
    FORWARD_FEATURES,
    LABEL_COLUMN,
    LABEL_TO_CATEGORY,
    RAW_DATA_DIR,
    RANDOM_STATE,
)


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    return df


def load_raw_csvs(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load and concatenate every CSV in data_dir (defaults to data/raw)."""
    data_dir = Path(data_dir) if data_dir else RAW_DATA_DIR
    csv_files = sorted(Path(data_dir).glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {data_dir}. Download the CICIDS2017 "
            "MachineLearningCSV files from "
            "https://www.unb.ca/cic/datasets/ids-2017.html and place them "
            "there, or use the synthetic sample in data/sample/ for a demo run."
        )
    frames = []
    for f in csv_files:
        df = pd.read_csv(f, low_memory=False, encoding="latin1")
        df = _clean_columns(df)
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    return full


def map_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(str).str.strip()
    df[LABEL_COLUMN] = df[LABEL_COLUMN].map(
        lambda x: LABEL_TO_CATEGORY.get(x, x)
    )
    # Drop any label we don't recognize (defensive, keeps pipeline robust
    # to unexpected/rare label spellings across CICIDS2017 CSVs)
    known = set(LABEL_TO_CATEGORY.values())
    df = df[df[LABEL_COLUMN].isin(known)]
    return df


def select_forward_features(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in FORWARD_FEATURES if c not in df.columns]
    if missing:
        raise KeyError(
            f"Expected forward-direction columns missing from data: {missing}. "
            "Check that the input CSV is CICIDS2017-formatted (CICFlowMeter output)."
        )
    cols = FORWARD_FEATURES + [LABEL_COLUMN]
    return df[cols].copy()


def clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    feature_cols = [c for c in df.columns if c != LABEL_COLUMN]
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feature_cols)
    df = df.drop_duplicates()
    return df


def prepare_dataset(
    raw_df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
):
    """
    Full preprocessing pipeline: label mapping -> feature selection ->
    cleaning -> encode -> scale -> stratified train/val/test split.

    Returns a dict with arrays/objects needed by train.py and detect.py.
    """
    df = map_labels(raw_df)
    df = select_forward_features(df)
    df = clean_numeric(df)

    X = df.drop(columns=[LABEL_COLUMN]).values
    y_raw = df[LABEL_COLUMN].values

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(test_size + val_size),
        random_state=RANDOM_STATE, stratify=y,
    )
    rel_test = test_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=rel_test,
        random_state=RANDOM_STATE, stratify=y_temp,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "scaler": scaler,
        "encoder": encoder,
        "feature_names": [c for c in df.columns if c != LABEL_COLUMN],
    }
