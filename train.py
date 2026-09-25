"""
train.py
--------
Trains and compares two scikit-learn classifiers for multi-class cyber
attack detection on the forward-only (unidirectional) CICIDS2017 feature
set, picks the best on validation macro-F1, evaluates it on the held-out
test set, and saves all artifacts needed for inference (detect.py).

Usage:
    # Demo run on the bundled synthetic sample data
    python src/train.py --data-dir data/sample

    # Real run on downloaded CICIDS2017 CSVs
    python src/train.py --data-dir data/raw
"""

import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)

from config import MODELS_DIR, OUTPUTS_DIR, RAW_DATA_DIR, RANDOM_STATE
from preprocessing import load_raw_csvs, prepare_dataset


def build_candidates():
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.1,
            max_depth=None,
            random_state=RANDOM_STATE,
        ),
    }


def evaluate(model, X, y, class_names):
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    macro_f1 = f1_score(y, preds, average="macro")
    report = classification_report(y, preds, target_names=class_names, digits=3)
    cm = confusion_matrix(y, preds)
    return acc, macro_f1, report, cm, preds


def plot_confusion_matrix(cm, class_names, out_path):
    plt.figure(figsize=(9, 7))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix (row-normalized) — Best Model, Test Set")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_feature_importance(model, feature_names, out_path, top_n=15):
    if not hasattr(model, "feature_importances_"):
        return
    importances = model.feature_importances_
    idx = np.argsort(importances)[-top_n:]
    plt.figure(figsize=(8, 6))
    plt.barh(np.array(feature_names)[idx], importances[idx], color="#3b6ea5")
    plt.xlabel("Importance")
    plt.title(f"Top {top_n} Forward-Direction Features")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=str, default=str(RAW_DATA_DIR))
    args = parser.parse_args()

    print(f"[1/5] Loading data from {args.data_dir} ...")
    raw_df = load_raw_csvs(args.data_dir)
    print(f"      Loaded {len(raw_df):,} raw flow records")

    print("[2/5] Preprocessing (label mapping, forward-feature selection, cleaning, scaling, split) ...")
    data = prepare_dataset(raw_df)
    class_names = list(data["encoder"].classes_)
    print(f"      Classes: {class_names}")
    print(f"      Train/Val/Test sizes: "
          f"{len(data['y_train'])}/{len(data['y_val'])}/{len(data['y_test'])}")

    print("[3/5] Training candidate models ...")
    results = {}
    for name, model in build_candidates().items():
        t0 = time.time()
        model.fit(data["X_train"], data["y_train"])
        elapsed = time.time() - t0
        val_acc, val_f1, _, _, _ = evaluate(model, data["X_val"], data["y_val"], class_names)
        results[name] = {"model": model, "val_acc": val_acc, "val_f1": val_f1, "train_time_s": elapsed}
        print(f"      {name:24s} val_acc={val_acc:.4f}  val_macro_f1={val_f1:.4f}  ({elapsed:.1f}s)")

    best_name = max(results, key=lambda k: results[k]["val_f1"])
    best_model = results[best_name]["model"]
    print(f"[4/5] Best model on validation macro-F1: {best_name}")

    test_acc, test_f1, report, cm, _ = evaluate(
        best_model, data["X_test"], data["y_test"], class_names
    )
    print(f"      Test accuracy={test_acc:.4f}  Test macro-F1={test_f1:.4f}")
    print(report)

    print("[5/5] Saving artifacts ...")
    joblib.dump(best_model, MODELS_DIR / "model.joblib")
    joblib.dump(data["scaler"], MODELS_DIR / "scaler.joblib")
    joblib.dump(data["encoder"], MODELS_DIR / "label_encoder.joblib")
    joblib.dump(data["feature_names"], MODELS_DIR / "feature_names.joblib")

    (OUTPUTS_DIR / "classification_report.txt").write_text(
        f"Best model: {best_name}\n\n{report}"
    )
    metrics = {
        "best_model": best_name,
        "test_accuracy": test_acc,
        "test_macro_f1": test_f1,
        "model_comparison": {
            k: {"val_acc": v["val_acc"], "val_macro_f1": v["val_f1"], "train_time_s": v["train_time_s"]}
            for k, v in results.items()
        },
        "classes": class_names,
        "n_train": len(data["y_train"]),
        "n_val": len(data["y_val"]),
        "n_test": len(data["y_test"]),
    }
    (OUTPUTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    plot_confusion_matrix(cm, class_names, OUTPUTS_DIR / "confusion_matrix.png")
    plot_feature_importance(best_model, data["feature_names"], OUTPUTS_DIR / "feature_importance.png")

    print(f"\nDone. Model + scaler + encoder saved to {MODELS_DIR}")
    print(f"Metrics + plots saved to {OUTPUTS_DIR}")


if __name__ == "__main__":
    main()
