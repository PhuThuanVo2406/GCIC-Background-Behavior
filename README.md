# GCIC pipeline: EDA, model comparison and dashboard

Reproducible Python pipeline for the "Background or Behavior?" GCIC poster. It audits the clean OULAD table, explores which features relate to course success, tunes eight classifiers, explains the best one with SHAP, and serves the results in a Streamlit dashboard.

- **Written results:** [REPORT.md](REPORT.md) (key insights after every step) and [EXECUTIVE_SUMMARY.md](EXECUTIVE_SUMMARY.md) (one page).
- **Notebook:** [GCIC_pipeline.ipynb](GCIC_pipeline.ipynb) runs the same step scripts, with each step's figures and insights inline.

## Folder layout

| Path | What it is |
|---|---|
| `config.py` | Paths (relative to this folder), `RANDOM_STATE = 42`, feature lists, labels, colour palette |
| `utils.py` | Data loading with column checks, figure/table saving, shared plot helpers |
| `step1_audit.py` … `step7_results.py` | One script per step, each with a `main()` and a docstring header |
| `run_all.py` | Runs steps 1–7 in order |
| `build_notebook.py` | Builds and executes `GCIC_pipeline.ipynb` from the scripts and `REPORT.md` |
| `dashboard/app.py` | Step 8: the Streamlit dashboard |
| `outputs/figures/` | Every chart as PNG at 200 dpi |
| `outputs/tables/` | Every number behind the charts and the report (CSV/JSON) |
| `outputs/models/` | `best_model.joblib` (preprocessing + model pipeline) and `best_model.json` |
| `outputs/dashboard_screenshots/` | One screenshot per dashboard tab |

## Setup

Python 3.11 or newer (tested on 3.13).

```bash
cd gcic/pipeline
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data

The scripts read two files from the sibling `clean/` folder:

- `../clean/oulad_clean.csv`: one row per registration (32,593 × 45), built by `../code/prep_oulad.py` from the raw OULAD CSVs.
- `../clean/oulad_cutoff_day28.csv`: day-28 behavior features, built by `../code/prep_cutoffs.py`. Only used for the leakage check at the end of step 7.

To point at copies elsewhere, set environment variables instead of editing code:

```bash
export GCIC_DATA=/path/to/oulad_clean.csv
export GCIC_CUTOFF28=/path/to/oulad_cutoff_day28.csv
export GCIC_RAW=/path/to/raw/oulad         # optional, only for step 1's missing-value chart
export GCIC_OUTPUT=/path/to/outputs        # optional, defaults to ./outputs
```

## Run

```bash
python run_all.py                 # steps 1–7; tuning in step 6 is the slow part
python step3_feature_target.py    # or run any single step
python build_notebook.py          # rebuild + execute the notebook (reuses saved step 6 results)
streamlit run dashboard/app.py    # step 8, opens http://localhost:8501
```

Steps 1–5 take under a minute in total. Step 6 took about 10 minutes on a 4-core machine (most of it the random forest, scikit-learn gradient boosting and SVM searches), and step 7 under a minute. Steps 6 and 7 must have run before the dashboard starts, because it loads `outputs/models/best_model.joblib`. Inside the notebook, set `RETUNE = True` in the step 6 cell to redo the search.

## Reproducibility notes

- Every random process uses `random_state=42`: the split, CV folds, random searches, models, samples for plots and SHAP.
- Scaling and encoding sit inside each model's `Pipeline`, so cross-validation never sees validation-fold statistics.
- Column names are checked against the data on load (`utils.load_data`), and the scripts raise an error instead of plotting a column that isn't there.
- Package versions are pinned in `requirements.txt`. Small differences in timings, and in the last decimal of tree-model scores, are expected on other hardware.
