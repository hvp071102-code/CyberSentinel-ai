"""
generate_sample_data.py
------------------------
Generates a small SYNTHETIC dataset shaped like CICIDS2017's forward-only
(unidirectional) feature set, so the rest of the pipeline (train, evaluate,
detect) can be run and demoed immediately without first downloading the
real ~6 GB CICIDS2017 CSVs.

This is a stand-in for coursework demo / CI purposes only. For the real
project, download the official CICIDS2017 "MachineLearningCSV" files from
https://www.unb.ca/cic/datasets/ids-2017.html, drop them into
data/raw/, and run train.py with --data-dir data/raw instead.

Usage:
    python src/generate_sample_data.py --rows 6000 --out data/sample/sample_flows.csv
"""

import argparse
import numpy as np
import pandas as pd

from config import FORWARD_FEATURES, LABEL_COLUMN, SAMPLE_DATA_DIR, RANDOM_STATE

rng = np.random.default_rng(RANDOM_STATE)

# Rough per-class generative profiles: (mean_scale, noise_scale, packet-rate bias)
# These are illustrative only -- built to give each class a distinct, learnable
# signature, not to reproduce real traffic statistics.
CLASS_PROFILES = {
    "BENIGN":      dict(weight=0.55, scale=1.0,  rate=1.0,  bulk=1.0),
    "DoS":         dict(weight=0.10, scale=0.3,  rate=25.0, bulk=0.4),
    "DDoS":        dict(weight=0.08, scale=0.15, rate=60.0, bulk=0.2),
    "PortScan":    dict(weight=0.10, scale=0.05, rate=40.0, bulk=0.05),
    "Brute Force": dict(weight=0.06, scale=0.5,  rate=8.0,  bulk=0.6),
    "Botnet":      dict(weight=0.04, scale=1.2,  rate=3.0,  bulk=1.3),
    "Web Attack":  dict(weight=0.04, scale=0.8,  rate=5.0,  bulk=0.9),
    "Infiltration": dict(weight=0.02, scale=2.0, rate=0.5,  bulk=2.5),
    "Heartbleed":  dict(weight=0.01, scale=3.0,  rate=0.2,  bulk=4.0),
}


def _make_rows(category: str, n: int, profile: dict) -> pd.DataFrame:
    scale, rate, bulk = profile["scale"], profile["rate"], profile["bulk"]
    base = {}
    base["Destination Port"] = rng.choice(
        [80, 443, 22, 21, 23, 3389, 8080, 53, 3306], size=n
    )
    base["Flow Duration"] = np.abs(rng.normal(2_000_000 * scale, 500_000 * scale, n))
    base["Total Fwd Packets"] = np.abs(rng.poisson(10 * rate, n)) + 1
    base["Total Length of Fwd Packets"] = base["Total Fwd Packets"] * np.abs(
        rng.normal(300 * bulk, 80 * bulk, n)
    )
    base["Fwd Packet Length Max"] = np.abs(rng.normal(500 * bulk, 100, n))
    base["Fwd Packet Length Min"] = np.abs(rng.normal(40, 10, n))
    base["Fwd Packet Length Mean"] = np.abs(rng.normal(250 * bulk, 60, n))
    base["Fwd Packet Length Std"] = np.abs(rng.normal(80, 30, n))
    base["Fwd IAT Total"] = base["Flow Duration"] * np.abs(rng.normal(0.9, 0.1, n))
    base["Fwd IAT Mean"] = base["Fwd IAT Total"] / np.maximum(base["Total Fwd Packets"], 1)
    base["Fwd IAT Std"] = np.abs(rng.normal(1000 / rate, 300 / rate, n))
    base["Fwd IAT Max"] = base["Fwd IAT Mean"] * np.abs(rng.normal(2, 0.5, n))
    base["Fwd IAT Min"] = base["Fwd IAT Mean"] * np.abs(rng.normal(0.2, 0.1, n))
    base["Fwd PSH Flags"] = rng.binomial(1, min(0.3 * scale, 1.0), n)
    base["Fwd URG Flags"] = rng.binomial(1, 0.02, n)
    base["Fwd Header Length"] = base["Total Fwd Packets"] * 20
    base["Fwd Packets/s"] = rate * np.abs(rng.normal(5, 1.5, n))
    base["Min Packet Length"] = np.abs(rng.normal(30, 8, n))
    base["Max Packet Length"] = np.abs(rng.normal(600 * bulk, 120, n))
    base["FIN Flag Count"] = rng.binomial(1, 0.15, n)
    base["SYN Flag Count"] = rng.binomial(1, min(0.2 * rate / 5, 1.0), n)
    base["RST Flag Count"] = rng.binomial(1, min(0.1 * rate / 5, 1.0), n)
    base["PSH Flag Count"] = rng.binomial(1, min(0.3 * scale, 1.0), n)
    base["ACK Flag Count"] = rng.binomial(1, 0.6, n)
    base["URG Flag Count"] = rng.binomial(1, 0.02, n)
    base["Down/Up Ratio"] = np.abs(rng.normal(1.0, 0.5, n))
    base["Avg Fwd Segment Size"] = base["Fwd Packet Length Mean"]
    base["Fwd Avg Bytes/Bulk"] = np.abs(rng.normal(100 * bulk, 40, n))
    base["Fwd Avg Packets/Bulk"] = np.abs(rng.normal(2 * bulk, 1, n))
    base["Fwd Avg Bulk Rate"] = np.abs(rng.normal(50 * bulk, 20, n))
    base["Subflow Fwd Packets"] = base["Total Fwd Packets"]
    base["Subflow Fwd Bytes"] = base["Total Length of Fwd Packets"]
    base["Init_Win_bytes_forward"] = np.abs(rng.normal(8192, 2000, n))
    base["act_data_pkt_fwd"] = np.abs(rng.poisson(5 * rate, n))
    base["min_seg_size_forward"] = np.abs(rng.normal(20, 4, n))
    base["Active Mean"] = np.abs(rng.normal(50000 * scale, 15000, n))
    base["Active Std"] = np.abs(rng.normal(5000, 2000, n))
    base["Active Max"] = base["Active Mean"] * 1.5
    base["Active Min"] = base["Active Mean"] * 0.5
    base["Idle Mean"] = np.abs(rng.normal(200000 * scale, 60000, n))
    base["Idle Std"] = np.abs(rng.normal(20000, 8000, n))
    base["Idle Max"] = base["Idle Mean"] * 1.5
    base["Idle Min"] = base["Idle Mean"] * 0.5

    df = pd.DataFrame(base)
    df = df[FORWARD_FEATURES]
    df[LABEL_COLUMN] = category
    return df


def generate(n_rows: int) -> pd.DataFrame:
    frames = []
    for category, profile in CLASS_PROFILES.items():
        n = max(int(n_rows * profile["weight"]), 20)
        frames.append(_make_rows(category, n, profile))
    df = pd.concat(frames, ignore_index=True)
    df = df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=6000)
    parser.add_argument("--out", type=str, default=str(SAMPLE_DATA_DIR / "sample_flows.csv"))
    args = parser.parse_args()

    df = generate(args.rows)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} synthetic unidirectional flow records to {args.out}")
    print(df[LABEL_COLUMN].value_counts())
