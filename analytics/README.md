# `/analytics` — Zepto Data & AI Platform

Profiles, cleans, and tells a visual story about the Titanic dataset (Part A), then
builds and rigorously evaluates a full classification + regression modeling pipeline
on the same cleaned data (Part B).

## How to run

```bash
cd analytics
pip install -r requirements.txt

python 01_eda.py        # loads titanic ONCE, profiles, cleans, EDA charts + titanic.csv
python 02_modeling.py   # reads the same titanic.csv, models, evaluates, saves pipeline
```

`01_eda.py` needs internet access the first time (Seaborn fetches the dataset from
its GitHub-hosted data repo and caches it locally — see the module spec's note on
this). `02_modeling.py` never re-fetches: it reads the committed `titanic.csv`.

**Outputs:**
- `titanic.csv` — the one committed offline fallback (raw data, saved immediately after
  the single load, per Task 1 — loadable via `pd.read_csv("titanic.csv")` even with no
  network)
- `charts/*.png` — every chart referenced below
- `eda_findings.md` — full Part A write-up (profiling, cleaning justification, IQR
  outliers, bivariate breakdowns, correlation interpretation, all 4 data-story charts
  with interpretations, standardization check)
- `modeling_findings.md` — full Part B write-up (split justification, all evaluation
  tables, imbalance comparison, tuning results, regression metrics, final
  recommendation)
- `model_pipeline.joblib` — the saved, reloadable, end-to-end fitted pipeline (Task 15)

## Design decisions

**One load, one cleaning policy, no re-fetch.** `01_eda.py` is the only place
`sns.load_dataset("titanic")` is called. The Task-2 cleaning logic lives in
`src/cleaning.py` as a single `clean_titanic()` function, imported by *both*
`01_eda.py` and `02_modeling.py` — so `02_modeling.py` reads the committed
`titanic.csv` and re-applies the identical, already-decided cleaning policy rather
than re-fetching data or re-deciding how to clean it.

**Missing-value policy** (percentages measured on the real 891-row dataset):
| Column | Missing % | Decision | Why |
|---|---|---|---|
| `deck` | 77.2% | Drop column | Imputing would fabricate ~3/4 of the column; the partial signal it carries (cabin location as a wealth proxy) is already captured, with far less missingness, by `pclass`/`fare`. |
| `embarked` / `embark_town` | 0.22% | Drop rows (2 rows) | Under the 5% threshold; negligible data loss, avoids inventing a port of embarkation. |
| `age` | 19.87% | Median-impute | In the 5–30% threshold range; median (not mean) used since age is right-skewed by older outliers. |

**Modeling feature set.** `pclass, sex, age, sibsp, parch, fare, embarked` — the
classic 7-feature set. Explicitly excluded: `alive` (identical information to
`survived`, would leak the label directly), and `class`/`who`/`adult_male`/`alone`/
`embark_town` (derived duplicates of features already included, adding redundant/
collinear columns with no new information).

**Regression side-task features.** Deliberately excludes `survived` as a predictor
of `fare`, even though it correlates: fare/class causally influenced survival odds,
not the reverse, so using `survived` to predict `fare` would model the relationship
backwards. Kept to demographic/family features only (`pclass`, `age`, `sibsp`,
`parch`, `sex`, `embarked`).

**Leakage prevention (Task 8).** All preprocessing (imputation, encoding, scaling)
is wrapped in a single `sklearn.Pipeline` (`ColumnTransformer` + estimator) that is
always fit on `(X_train, y_train)` only; `X_test` only ever sees `.predict()` /
`.transform()`, never `.fit()`. This is enforced structurally, not by convention.

**Why the imputers are "no-ops" here, and why they're still included.** By the time
`02_modeling.py` runs, Task 2's cleaning has already resolved missingness in every
feature used (deck dropped, embarked-missing rows dropped, age imputed) — so the
`SimpleImputer` steps inside the Task 8 pipeline don't actually change this
particular dataset. They're included anyway because that's what makes the *saved*
pipeline (Task 15) genuinely safe to run on fresh raw data later that might contain
missing values this training data didn't.

## Final model comparison

**Classification (test set, n=178):**
| model | accuracy | precision | recall | f1 | auc |
|---|---|---|---|---|---|
| Logistic Regression | 0.809 | 0.783 | 0.691 | 0.734 | 0.861 |
| Decision Tree | 0.758 | 0.745 | 0.559 | 0.639 | 0.830 |
| Random Forest | 0.809 | 0.774 | 0.706 | 0.738 | 0.816 |

**Regression (fare prediction):**
| model | MAE | RMSE | R² | Adjusted R² |
|---|---|---|---|---|
| Linear Regression | 21.14 | 41.75 | 0.347 | 0.324 |

Classification and regression metrics are on different scales and are reported as
two separate tables, not a single merged scale.

**Recommendation:** deploy **Random Forest**. It has the highest F1 (0.738) among
the three classifiers, balancing precision (0.774) and recall (0.706) rather than
optimizing one at the other's expense — both false positives and false negatives on
`survived` are costly in different ways here. Its AUC (0.816) confirms it separates
the two classes well across thresholds, not just at the default 0.5 cutoff.

**Imbalance handling:** SMOTE (applied to the training fold only) gave the best F1
(0.761) of the three variants tested (baseline / `class_weight='balanced'` / SMOTE)
— see `modeling_findings.md` for the full table.

**Tuning:** `GridSearchCV` over `n_estimators`/`max_depth`/`max_features` selected
`{'max_depth': None, 'max_features': 'sqrt', 'n_estimators': 100}`, with an OOB score
of 0.809 for the best estimator (`RandomForestClassifier(oob_score=True, ...)`).

Full write-ups, every chart, and every required written interpretation (skew
direction, top-2 correlations, all 4 data-story chart interpretations, the
heteroscedasticity conclusion, etc.) are in `eda_findings.md` and
`modeling_findings.md`.

## A note on how this was tested

This module was built with AI assistance (permitted per the program's guidelines,
provided the implementation is understood and can be explained). Unlike
`/data_pipeline`, both scripts here were run for real end-to-end in the same
environment they were built in — the Titanic dataset is hosted on GitHub, which was
reachable — so every number in this README and in both `*_findings.md` files came
from an actual run, not synthetic/placeholder data.

## Git workflow

The overall repository (not just this module) includes a feature branch created,
committed to at least twice, and merged into `main` — see root README /
`git log --graph --all`.
