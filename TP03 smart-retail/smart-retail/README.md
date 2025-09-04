# Smart Retail: Customer Purchase Prediction — Codespaces Starter

A from-scratch, Codespaces-friendly starter for an AI-powered recommendation engine that clusters customers (K-Means) and predicts next purchases (RandomForest/XGBoost), with a FastAPI service for real-time recommendations and scaffolding for AWS integration (SageMaker/Lambda/DynamoDB placeholders).

## Quickstart (GitHub Codespaces)

1. Open this repo in **GitHub Codespaces**.
2. Run the setup and generate some sample data:
   ```bash
   make setup
   make data
   ```
3. Train models (K-Means + RandomForest by default):
   ```bash
   make train
   ```
4. Launch the API:
   ```bash
   make api
   ```
   Then open the forwarded port and visit `/docs` for Swagger UI.

## Repo Structure

```
.
├─ .devcontainer/
│  └─ devcontainer.json        # Codespaces environment
├─ .github/workflows/
│  └─ ci.yml                   # Lint & test on push (optional)
├─ api/
│  └─ main.py                  # FastAPI app (inference API)
├─ data/
│  └─ (generated files)        # Synthetic CSVs + trained models
├─ scripts/
│  ├─ generate_synthetic_data.py
│  ├─ run_training.sh
│  └─ start_api.sh
├─ src/
│  ├─ config.py
│  ├─ data_prep.py
│  ├─ features.py
│  ├─ train.py
│  ├─ inference.py
│  └─ models/
├─ tests/
│  └─ test_smoke.py
├─ Makefile
├─ requirements.txt
└─ README.md
```

## What’s included

- **Synthetic data generator** for customers, products, transactions
- **Customer segmentation** via K-Means
- **Next-purchase prediction** via RandomForest (switchable to XGBoost if installed)
- **Top-N recommendations** fallback to popularity when model not trained
- **FastAPI** inference service with `/health` and `/recommend` endpoints
- **Makefile** helpers for common tasks
- **AWS placeholders** in comments where SageMaker/Lambda/DynamoDB would integrate

## Commands

```bash
make setup       # install deps
make data        # generate synthetic data
make train       # train K-Means + classifier
make api         # run FastAPI (uvicorn)
make test        # run basic tests
```

## Environment Variables (optional)

- `MODEL_DIR` (default: `data`)
- `N_CLUSTERS` (default: 5)
- `TOP_K` (default: 5)

## Notes

- By default we use **RandomForest** to avoid native compilers in Codespaces. If you want **XGBoost**, keep it in `requirements.txt` and enable the flag in `src/train.py`.
- For AWS integration, see comments in code where `boto3` could push/pull from S3, register SageMaker endpoints, or use DynamoDB for low-latency customer context.
