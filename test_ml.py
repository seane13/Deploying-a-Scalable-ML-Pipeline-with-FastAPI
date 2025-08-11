import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from ml.data import process_data
from ml.model import (
    train_model,
    inference,
    compute_model_metrics,
    save_model,
    load_model,
)

LABEL = "salary"
SEED = 42
DATA_PATH = os.getenv("DATA_PATH", "data/census.csv")
CATEGORICAL_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

def _load_sample_df(n=2000, seed=SEED):
    """Load a small, stratified sample for fast, deterministic tests."""
    df = pd.read_csv(DATA_PATH)
    if len(df) > n:
        # Use stratified split instead of groupby.apply to avoid deprecation warnings
        df, _ = train_test_split(df, train_size=n, stratify=df[LABEL], random_state=seed)
    return df.reset_index(drop=True)


def _fit_process_split(df):
    """Train/test split and processing using the project pipeline."""
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=df[LABEL]
    )

    # IMPORTANT: capture lb from training
    X_train, y_train, encoder, lb = process_data(
        train_df,
        categorical_features=CATEGORICAL_FEATURES,
        label=LABEL,
        training=True,
    )

    # IMPORTANT: pass BOTH encoder and lb to the test call
    X_test, y_test, _, _ = process_data(
        test_df,
        categorical_features=CATEGORICAL_FEATURES,
        label=LABEL,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    # Fallback: if y_test is still object dtype, binarize it with the same lb
    if getattr(y_test, "dtype", None) is not None and y_test.dtype == object:
        y_test = lb.transform(np.array(y_test).reshape(-1, 1)).ravel()

    return (X_train, y_train, X_test, y_test, encoder, test_df)


# 1) Algorithm check
def test_model_algorithm_is_random_forest():
    df = _load_sample_df(n=400)  # smaller for speed
    X_train, y_train, *_ = _fit_process_split(df)
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier), "Expected RandomForestClassifier"


# 2) Metrics within reasonable range
def test_metrics_within_expected_range():
    df = _load_sample_df(n=1200)
    X_train, y_train, X_test, y_test, *_ = _fit_process_split(df)

    model = train_model(X_train, y_train)
    preds = inference(model, X_test)
    precision, recall, f1 = compute_model_metrics(y_test, preds)

    # metrics bounded in [0, 1]
    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
    assert 0.0 <= f1 <= 1.0

    # sanity floor for this pipeline/dataset (tune if needed)
    assert f1 >= 0.55, f"F1 too low for this dataset/pipeline: {f1:.3f}"


# 3) Inference shapes and label set
def test_inference_shapes_and_labels():
    df = _load_sample_df(n=600)
    X_train, y_train, X_test, y_test, *_ = _fit_process_split(df)

    model = train_model(X_train, y_train)
    preds = inference(model, X_test)

    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == X_test.shape[0], "Pred length must match X_test rows"

    # binary labels {0,1}
    uniq = set(np.unique(preds))
    assert uniq.issubset({0, 1}), f"Unexpected labels: {uniq}"


# 4) Processed arrays are numeric with expected rank
def test_processed_arrays_numeric_and_shaped():
    df = _load_sample_df(n=400)
    X_train, y_train, X_test, y_test, *_ = _fit_process_split(df)

    # rank & dtype checks
    assert X_train.ndim == 2 and X_test.ndim == 2
    assert y_train.ndim == 1 and y_test.ndim == 1

    assert X_train.shape[0] > 0 and X_train.shape[1] > 0
    assert X_test.shape[0] > 0 and X_test.shape[1] == X_train.shape[1]

    # numeric dtypes for features and labels
    assert X_train.dtype.kind in ("f", "i", "u")
    assert X_test.dtype.kind in ("f", "i", "u")
    assert y_train.dtype.kind in ("i", "u")
    assert y_test.dtype.kind in ("i", "u")


# 5) Save/load roundtrip preserves predictions
def test_artifact_roundtrip_predictions_identical(tmp_path):
    df = _load_sample_df(n=600)
    X_train, y_train, X_test, y_test, *_ = _fit_process_split(df)

    model = train_model(X_train, y_train)
    preds_before = inference(model, X_test)

    model_path = tmp_path / "model.pkl"
    save_model(model, str(model_path))
    loaded = load_model(str(model_path))
    preds_after = inference(loaded, X_test)

    assert np.array_equal(preds_before, preds_after), "Predictions changed after reload"
