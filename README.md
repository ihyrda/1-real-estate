# Real Estate Price Prediction

A modular rebuild of the `Real_Estate.ipynb` course notebook. The project predicts residential sale price from thirteen property features, compares a linear regression against a random forest, selects the stronger model on held-out data, and serves the result through a Streamlit application. 

## Setup

Requires **Python 3.11**. Run every command from the project root (the folder containing `app.py`).

The virtual environment is not included in the repository, so this creates a fresh one:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Dependencies are listed in `requirements.txt`.

## Running the tests

```powershell
python -m pytest -v
```

## Running the app

```powershell
streamlit run app.py
```

## Project structure

```text
1_real_estate/
├── app.py                 Streamlit application
├── README.md
├── requirements.txt
├── data/
│   └── final.csv          prepared modelling dataset
├── original_notebook/
│   └── Real_Estate.ipynb  preserved course source
├── src/
│   ├── __init__.py
│   ├── data.py            loading, validation, summary
│   ├── preprocessing.py   feature preparation, derived fields
│   ├── modeling.py        split, train, evaluate, select, predict
│   └── logging_config.py  console logging
└── tests/
    └── test_project.py
```

## How it works

`data/final.csv` is already a prepared modelling dataset — fully numeric, no missing values, derived columns present — so no cleaning step is required. Validation confirms the file is the one the project expects rather than repairing anything.

Both models train on the same 80/20 split, seeded at 42 and stratified on `property_type_Condo` so the condo share is matched on both sides. Metrics are recalculated on every run rather than copied from the notebook.

**Selection rule: lowest test MAE.** 

Four features are derived rather than collected, because the dataset does not support entering them independently:

## Links

- GitHub repository: _to be added after upload_
- Deployed Streamlit app: _to be added after deployment_
