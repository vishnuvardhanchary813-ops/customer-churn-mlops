# Customer Churn MLOps

An intermediate-level Python and Scikit-learn project for predicting customer churn.

## Project stages

1. Inspect the IBM Telco Customer Churn dataset in `notebooks/`.
2. Build leakage-safe preprocessing in `src/data/preprocess.py`.
3. Train and evaluate candidate models in `src/models/`, tracking runs with MLflow.
4. Track data and pipeline stages with DVC.
5. Serve predictions with FastAPI and Streamlit.
6. Test with pytest, package with Docker Compose, and build with GitHub Actions.

## Dataset

Place the IBM Telco Customer Churn CSV at `raw/dataset.csv`. Keep the raw data out of Git; DVC configuration will track it. The expected target column is `Churn`.

## Current status

EDA, preprocessing, model training, test evaluation, the FastAPI backend, and the Streamlit frontend are implemented. Training selects the model by validation F1, logs MLflow runs to `mlflow.db`, and saves the fitted pipeline to `artifacts/model/model.joblib`.

## DVC data and pipeline

Install the DVC dependency separately, then initialize DVC and track the raw dataset:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dvc.txt
$env:DVC_SITE_CACHE_DIR = Join-Path (Get-Location).Path 'work\dvc-site-cache'
dvc init
dvc add raw/dataset.csv
```

Commit the generated `.dvc/`, `raw/dataset.csv.dvc`, and updated `.gitignore` files. Reproduce and inspect the pipeline with:

```powershell
dvc repro
dvc status
dvc dag
dvc metrics show
```

`dvc.yaml` defines preprocessing, training, and test evaluation stages. `params.yaml` holds the split and validation settings; MLflow records each model candidate run and its metrics. The model artifact remains in Git for the Docker/EC2 deployment. Without a shared DVC remote, the dataset and DVC cache stay local; a shared project needs an S3 remote before another machine can reproduce preprocessing.

## Dataset EDA notebook

Install notebook dependencies and launch Jupyter from the project root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-eda.txt
.\.venv\Scripts\python.exe -m notebook
```

Open `notebooks/01_data_understanding.ipynb`. It reads `raw/dataset.csv`, checks data quality, and explores churn rates by contract, internet service, payment method, and tenure.

## Run the API

Install dependencies, then start the backend from the project root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` to try the endpoints. The model must be trained first.

## Run the frontend

Keep FastAPI running in one terminal. In another terminal from the project root:

```powershell
.\.venv\Scripts\python.exe -m streamlit run frontend/app.py
```

Open `http://localhost:8501`. Locally the frontend calls `http://127.0.0.1:8000`; set `API_URL` when the backend uses another address.

## Run tests

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -v
```

The API/model tests need the saved `artifacts/model/model.joblib` pipeline.

## Run with Docker Compose

Build and start both containers from the project root:

```powershell
docker compose build
docker compose up -d
docker compose ps
docker compose logs -f
```

Open `http://localhost:8501`. The frontend calls the backend at `http://api:8000` over the Compose network. The API has no host port mapping and is reachable only by services on the Compose network.

Stop the containers with:

```powershell
docker compose down
```

## GitHub Actions

The workflow in `.github/workflows/ci.yml` runs on pushes to `main` and pull requests targeting `main`. It installs the project dependencies with Python 3.11, runs pytest, and builds both Docker images. After pushing the repository to GitHub, check the **Actions** tab for the latest run and open it to view step logs.
