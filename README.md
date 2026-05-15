# focus_platform_ai

This folder contains the AI components for the Focus platform — data preprocessing, model training, prediction utilities, and a small Flask prediction API.

## Layout

- `createTests.py` — helper to run quick smoke tests
- `predict.py` — convenience script for running an ad-hoc prediction
- `requirements.txt` — Python dependencies
- `api/ai_api.py` — Flask API (endpoints: `/health`, `/predict`)
- `preprocessing/` — data preprocessing scripts
- `training/` — training scripts and helpers
- `prediction/` — production-oriented prediction helpers
- `models/` — trained model artifacts (expected: `evaluate_skill_model_vFINAL.pkl`)
- `data/` — sample and raw data files
- `analysis/` — plotting / evaluation utilities

## Requirements

- Python 3.8+
- See `requirements.txt` for Python packages. Key packages used:
  - `flask`, `joblib`, `lightgbm`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`

### Packages to install

Install all required packages from `requirements.txt` (recommended):

```bash
pip install -r requirements.txt
```

Or install the core packages individually:

```bash
pip install flask joblib lightgbm numpy pandas scikit-learn matplotlib
```

## Quick setup

1. Create and activate a virtual environment (recommended):

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Prepare data

- Place your training / input CSV files under the `data/` directory. A common file used by the pipeline is `final_processed_data.csv` (or the provided sample files).

3. Edit configuration (if needed)

- Review training and preprocessing scripts for any hard-coded paths. Most scripts expect relative paths from the `focus_platform_ai` folder.

## Common workflows

Preprocess data:

```bash
python preprocessing/preprocess.py
```

Build skill snapshots from ordered student answers and skill difficulty values:

```bash
python preprocessing/skill_snapshot_smoke_test.py
```

Train a model (example):

```bash
python training/train.py
```

Run a local ad-hoc prediction using the convenience script:

```bash
python predict.py "path/to/example_input.json"
```

Run the production-style prediction script in `prediction/` (if provided):

```bash
python prediction/predict.py --input data/some_input.csv --output predictions.csv
```

Run tests / quick checks:

```bash
python createTests.py
```

## Flask API (prediction)

The API is a small Flask app in `api/ai_api.py`. It attempts to load a model bundle from `models/` (the code expects a file like `evaluate_skill_model_vFINAL.pkl`).

Start the API (development):

```bash
python api/ai_api.py
```

By default the server listens on port `5000`. Endpoints:

- `GET /health` — returns model load status and feature/threshold info
- `POST /predict` — accepts either prebuilt feature rows or raw ordered student answers plus `skills_difficulty`; returns probability and status per skill

Example `curl` request:

```bash
curl -X POST http://127.0.0.1:5000/predict \
	-H "Content-Type: application/json" \
	-d '{"items":[{"order_id":1,"skill_id":456,"is_correct":1},{"order_id":2,"skill_id":456,"is_correct":0}],"skills_difficulty":{"456":0.35}}'
```

## Model artifacts

- Trained models (joblib bundles) should be placed in `models/` and include at least the serialized model and metadata keys used by `api/ai_api.py` (e.g. `model`, `features`, `threshold`). The API tries `evaluate_skill_model_vFINAL.pkl` by default.

## Notes & troubleshooting

- If the API reports "model load failed", confirm that the expected file exists in `models/` and is a joblib bundle with the expected keys.
- Use `pip install -r requirements.txt` inside the activated virtualenv to avoid system package conflicts.
- For large training runs, prefer running on a machine with sufficient memory and consider using smaller sample data for quick iterations.

## Contributing

- Add new preprocessing transformations under `preprocessing/`.
- Keep models in `models/` and name them clearly with version tags.
