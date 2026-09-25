"""
config.py
---------
Central configuration for the AI-Based Detection of Cyber Threats in
Unidirectional IP Traffic module.

Design note on "unidirectional":
CICIDS2017 flow records (produced by CICFlowMeter) contain both forward
(source -> destination) and backward (destination -> source) statistics.
A sensor that only sees ONE direction of traffic (e.g. a network tap on an
asymmetric-routed link, a port-mirrored uplink, or a unidirectional
diode/sensor used in OT/ICS networks) can never populate the backward
("Bwd"/"Backward") columns or the combined "Flow Bytes/s" / "Flow
Packets/s" / "Flow IAT *" columns, because those require seeing packets
travelling in both directions of the same flow.

FORWARD_FEATURES below is therefore a deliberately restricted feature set:
only columns that can be computed purely from the forward-direction
packet stream. This is what makes the module suitable for unidirectional
capture points, at the cost of discarding some signal the full
bidirectional CICIDS2017 feature set would normally use.
"""

from pathlib import Path

# ---------------------------------------------------------------------
# Paths
#
# Deployment-agnostic: works whether the project keeps its normal nested
# layout (src/, models/, data/sample/ as separate folders) OR everything
# was uploaded flat into one folder (e.g. via GitHub's browser upload,
# which doesn't preserve subfolders unless a path prefix was set). Each
# path below picks whichever location actually has the file/folder.
# ---------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_PARENT = _THIS_DIR.parent if _THIS_DIR.name == "src" else _THIS_DIR


def _first_existing(*candidates: Path) -> Path:
    for c in candidates:
        if c.exists():
            return c
    return candidates[-1]  # fall back to the flat/default location


MODELS_DIR = _first_existing(_PARENT / "models", _THIS_DIR)
SAMPLE_DATA_DIR = _first_existing(_PARENT / "data" / "sample", _THIS_DIR, _PARENT)
RAW_DATA_DIR = _first_existing(_PARENT / "data" / "raw", _PARENT)
OUTPUTS_DIR = _first_existing(_PARENT / "outputs", _PARENT)
DATA_DIR = SAMPLE_DATA_DIR.parent if SAMPLE_DATA_DIR.name == "sample" else _PARENT

for d in (RAW_DATA_DIR, OUTPUTS_DIR):
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass  # read-only deployment filesystem (e.g. Streamlit Cloud) — fine to skip

# ---------------------------------------------------------------------
# Raw CICIDS2017 label -> broader attack category
# (CICIDS2017 "Label" column values, as published by the Canadian
# Institute for Cybersecurity, University of New Brunswick)
# ---------------------------------------------------------------------
LABEL_TO_CATEGORY = {
    "BENIGN": "BENIGN",
    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",
    "DDoS": "DDoS",
    "PortScan": "PortScan",
    "FTP-Patator": "Brute Force",
    "SSH-Patator": "Brute Force",
    "Bot": "Botnet",
    "Web Attack \x96 Brute Force": "Web Attack",
    "Web Attack \x96 XSS": "Web Attack",
    "Web Attack \x96 Sql Injection": "Web Attack",
    "Web Attack – Brute Force": "Web Attack",
    "Web Attack – XSS": "Web Attack",
    "Web Attack – Sql Injection": "Web Attack",
    "Infiltration": "Infiltration",
    "Heartbleed": "Heartbleed",
}

ATTACK_CATEGORIES = [
    "BENIGN", "DoS", "DDoS", "PortScan", "Brute Force",
    "Botnet", "Web Attack", "Infiltration", "Heartbleed",
]

# Severity used by the detection module for alerting/triage
CATEGORY_SEVERITY = {
    "BENIGN": "none",
    "PortScan": "low",
    "Brute Force": "medium",
    "Web Attack": "medium",
    "Botnet": "high",
    "DoS": "high",
    "DDoS": "critical",
    "Infiltration": "critical",
    "Heartbleed": "critical",
}

# ---------------------------------------------------------------------
# Forward-only ("unidirectional-observable") feature set
# Column names match raw CICIDS2017 CSVs (note the leading spaces that
# ship in the official files, handled in preprocessing.py).
# ---------------------------------------------------------------------
FORWARD_FEATURES = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Length of Fwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Fwd PSH Flags",
    "Fwd URG Flags",
    "Fwd Header Length",
    "Fwd Packets/s",
    "Min Packet Length",
    "Max Packet Length",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "Down/Up Ratio",
    "Avg Fwd Segment Size",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Subflow Fwd Packets",
    "Subflow Fwd Bytes",
    "Init_Win_bytes_forward",
    "act_data_pkt_fwd",
    "min_seg_size_forward",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
]

LABEL_COLUMN = "Label"
RANDOM_STATE = 42
