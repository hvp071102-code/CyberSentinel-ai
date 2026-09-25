# CyberSentinel AI — Unidirectional IDS

AI-based detection of cyber threats in unidirectional IP traffic. SIH prototype.

A Streamlit dashboard on top of a real ML detection engine trained on
**CICIDS2017**, restricted to 42 **forward-direction-only** flow features
(so it works off a one-way tap / port-mirror / data-diode sensor that
never sees return traffic). Detects **BENIGN** plus 8 attack categories:
DoS, DDoS, PortScan, Brute Force, Botnet, Web Attack, Infiltration,
Heartbleed.

## Files in this repo

| File | Purpose |
|---|---|
| `app.py` | Streamlit dashboard — run this |
| `config.py` | Feature list, label map, severity map, file paths |
| `preprocessing.py` | CICIDS2017 cleaning / feature selection |
| `train.py` | Trains and saves the model |
| `detect.py` | `UnidirectionalIDS` — loads the model and classifies flows |
| `generate_sample_data.py` | Creates the synthetic demo dataset |
| `model.joblib`, `scaler.joblib`, `label_encoder.joblib`, `feature_names.joblib` | Pre-trained model artifacts (already trained — ready to use) |
| `sample_flows.csv` | Synthetic demo traffic for the dashboard |
| `requirements.txt` | Pinned Python package versions |
| `run_app.bat` | Double-click launcher (Windows, local use) |

Everything sits flat in one folder — `config.py` auto-detects this layout,
so no subfolders are required.

## Run it locally

```bash
python -m venv .venv
.venv\Scripts\activate.bat        # Windows
# source .venv/bin/activate       # Mac/Linux

pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`. A trained model is already included, so
it works immediately — pick **"Demo traffic"** in the sidebar to classify
600 sample flows right away, or **"Upload CSV"** to score your own flow
data (must contain the 42 required forward-direction columns, shown in
the sidebar).

On Windows, once set up once, you can just double-click **`run_app.bat`**
from then on instead of typing commands.

### ⚠️ If you see `ModuleNotFoundError: No module named '_loss'`

The model files were saved with a specific scikit-learn/numpy version.
Retrain locally so the model matches what's installed on your machine:

```bash
python generate_sample_data.py --rows 8000
python train.py --data-dir .
```

This overwrites the `.joblib` files with ones compatible with your setup.

## Deploy it online (Streamlit Community Cloud)

1. Push/upload this repo to GitHub (public).
2. Go to **share.streamlit.io** → Sign in with GitHub → **Create app**.
3. Pick this repo, branch `main`, main file path `app.py`.
4. **Important:** under **Advanced settings**, set **Python version to 3.12**
   (Streamlit Cloud currently defaults new apps to Python 3.14, which
   doesn't yet have stable scikit-learn wheels — this causes the model
   to fail loading with a `ModuleNotFoundError`).
5. Click **Deploy**. You'll get a public link like
   `https://your-app-name.streamlit.app`.

`requirements.txt` pins exact library versions
(`numpy==2.4.4`, `scikit-learn==1.8.0`, `scipy==1.17.1`, `joblib==1.5.3`)
to match what the bundled model was trained with — don't loosen these
unless you also retrain the model against the new versions.

## How the AI/ML module works

1. **Input** — each row is one network flow, described by 42 statistics
   a one-directional sensor can measure (packet counts, byte lengths,
   inter-arrival times, TCP flag counts, header lengths, active/idle
   timers). Every `Bwd`/`Backward` and combined `Flow *` column from the
   original CICIDS2017 dataset is deliberately excluded, since those
   need to see both directions of a conversation.
2. **Training** (`train.py`) — trains Random Forest and
   HistGradientBoosting classifiers on labelled CICIDS2017 flows, keeps
   whichever scores higher on a held-out validation split, evaluates on
   a further held-out test split, and saves the model + a
   `StandardScaler` + a `LabelEncoder` with `joblib`.
3. **Inference** (`detect.py`) — the `UnidirectionalIDS` class loads
   those saved files once, then for each flow: selects the 42 required
   columns → scales them → runs the classifier → decodes the predicted
   class → looks up a severity tier (`none/low/medium/high/critical`).
   `predict_batch()` scores a whole CSV at once (what the dashboard
   uses); `alerts()` filters results down to a chosen severity threshold.
4. **Dashboard** (`app.py`) — loads the model once, lets you pick demo
   or uploaded traffic, runs `predict_batch()`, and shows summary
   metrics, a threat-category chart, a severity chart, a recent-events
   table, a filterable alerts table, and a CSV export button.

### Retraining on the real CICIDS2017 dataset

The bundled model is trained on **synthetic** sample data (safe to ship,
but not a real evaluation). For your actual SIH submission:

```bash
# Download the "MachineLearningCSV" files from
# https://www.unb.ca/cic/datasets/ids-2017.html and put them in this folder
python train.py --data-dir .
```

This overwrites the `.joblib` files with the real-data-trained model — no
other code changes needed.

### Command-line use (no dashboard)

```bash
python detect.py --input sample_flows.csv --min-severity medium
```

## Safety note

Use only on networks/traffic you're authorized to monitor. This is a
defensive detection prototype — it classifies and alerts, it does not
block traffic, exploit anything, or take any active action.
