"""
detect.py
---------
The detection module itself: loads the trained model/scaler/encoder and
classifies unidirectional IP flow records into BENIGN or one of the
attack categories (DoS, DDoS, PortScan, Brute Force, Botnet, Web Attack,
Infiltration, Heartbleed), with a severity/alert layer on top.

This is the piece meant to sit downstream of a flow exporter (e.g.
CICFlowMeter, nProbe, a custom sensor on a unidirectional tap/diode) that
emits per-flow forward-direction statistics only.

Usage as a library:
    from detect import UnidirectionalIDS
    ids = UnidirectionalIDS()
    result = ids.predict_flow({...})            # single flow -> dict
    results_df = ids.predict_batch("flows.csv")  # many flows -> DataFrame

Usage as a CLI demo:
    python src/detect.py --input data/sample/sample_flows.csv --limit 20
"""

import argparse
from pathlib import Path
from typing import Dict, List, Union

import joblib
import numpy as np
import pandas as pd

from config import (
    MODELS_DIR,
    FORWARD_FEATURES,
    LABEL_COLUMN,
    CATEGORY_SEVERITY,
)


class UnidirectionalIDS:
    """Loads a trained model and classifies unidirectional traffic flows."""

    def __init__(self, models_dir: Union[str, Path] = MODELS_DIR):
        models_dir = Path(models_dir)
        try:
            self.model = joblib.load(models_dir / "model.joblib")
            self.scaler = joblib.load(models_dir / "scaler.joblib")
            self.encoder = joblib.load(models_dir / "label_encoder.joblib")
            self.feature_names = joblib.load(models_dir / "feature_names.joblib")
        except FileNotFoundError as e:
            raise FileNotFoundError(
                "No trained model found. Run `python src/train.py "
                "--data-dir data/sample` (demo) or `--data-dir data/raw` "
                "(real CICIDS2017 data) first."
            ) from e

    # ------------------------------------------------------------------
    def _vectorize(self, flow: Dict) -> np.ndarray:
        missing = [f for f in self.feature_names if f not in flow]
        if missing:
            raise KeyError(f"Flow record is missing required fields: {missing}")
        row = [flow[f] for f in self.feature_names]
        return np.array(row, dtype=float).reshape(1, -1)

    def predict_flow(self, flow: Dict) -> Dict:
        """Classify a single unidirectional flow record (dict of features)."""
        x = self._vectorize(flow)
        x_scaled = self.scaler.transform(x)
        pred_idx = self.model.predict(x_scaled)[0]
        category = self.encoder.inverse_transform([pred_idx])[0]

        proba = None
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(x_scaled)[0]
            proba = float(probs[pred_idx])

        return {
            "predicted_category": category,
            "confidence": proba,
            "severity": CATEGORY_SEVERITY.get(category, "unknown"),
            "is_attack": category != "BENIGN",
        }

    def predict_batch(self, source: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
        """Classify many flows at once from a CSV path or a DataFrame."""
        if isinstance(source, (str, Path)):
            df = pd.read_csv(source)
            df.columns = [c.strip() for c in df.columns]
        else:
            df = source.copy()

        missing = [f for f in self.feature_names if f not in df.columns]
        if missing:
            raise KeyError(f"Input data is missing required fields: {missing}")

        X = df[self.feature_names].apply(pd.to_numeric, errors="coerce")
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        X_scaled = self.scaler.transform(X.values)

        preds = self.model.predict(X_scaled)
        categories = self.encoder.inverse_transform(preds)

        out = df.copy()
        out["predicted_category"] = categories
        out["severity"] = [CATEGORY_SEVERITY.get(c, "unknown") for c in categories]
        out["is_attack"] = out["predicted_category"] != "BENIGN"

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_scaled)
            out["confidence"] = probs[np.arange(len(preds)), preds]

        return out

    def alerts(self, results_df: pd.DataFrame, min_severity: str = "low") -> pd.DataFrame:
        """Filter batch results down to flows that should raise an alert."""
        order = ["none", "low", "medium", "high", "critical"]
        threshold = order.index(min_severity)
        mask = results_df["severity"].apply(
            lambda s: order.index(s) >= threshold if s in order else False
        )
        cols = [c for c in ["Destination Port", "predicted_category", "severity", "confidence"]
                if c in results_df.columns]
        return results_df.loc[mask, cols].reset_index(drop=True)


def _cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=str, required=True, help="CSV of unidirectional flow records")
    parser.add_argument("--limit", type=int, default=20, help="Rows to display")
    parser.add_argument("--min-severity", type=str, default="low",
                         choices=["none", "low", "medium", "high", "critical"])
    args = parser.parse_args()

    ids = UnidirectionalIDS()
    results = ids.predict_batch(args.input)

    print(f"Classified {len(results)} flows.\n")
    print("Category distribution:")
    print(results["predicted_category"].value_counts().to_string())

    print(f"\nSample predictions (first {args.limit}):")
    show_cols = [c for c in ["Destination Port", "predicted_category", "severity", "confidence"]
                 if c in results.columns]
    print(results[show_cols].head(args.limit).to_string(index=False))

    alerts = ids.alerts(results, min_severity=args.min_severity)
    print(f"\n{len(alerts)} alert(s) at severity >= '{args.min_severity}':")
    print(alerts.head(20).to_string(index=False))


if __name__ == "__main__":
    _cli()
