# Zepto Data & AI Platform — Capstone Project

One connected platform, three modules, one repository:

| Module | Marks | What it does |
|---|---|---|
| `/data_pipeline` | 25 | Scrapes a product catalog, cleans it, and loads it into a normalized SQLite store |
| `/analytics` | 50 | Profiles and models a customer-style dataset end to end |
| `/support_assistant` | 25 | Grounded GenAI assistant answering policy questions from Zepto's own documents |

> Status: `/data_pipeline` and `/analytics` complete. `/support_assistant` in progress.

## Requirements

Each module has its own `requirements.txt` (chosen over one consolidated file so each
module's dependencies stay isolated and easy to install independently — see each
module's README for why).

## How to run each module

### 1. `/data_pipeline`
See [`data_pipeline/README.md`](data_pipeline/README.md) for full details. Quick start:
```bash
cd data_pipeline
pip install -r requirements.txt
python -m src.pipeline          # scrape -> clean -> convert -> load into SQLite
python -m src.queries           # run the 5 required SQL queries + pandas verification
```

### 2. `/analytics`
See [`analytics/README.md`](analytics/README.md) for full details (design decisions,
model comparison table, final recommendation). Quick start:
```bash
cd analytics
pip install -r requirements.txt
python 01_eda.py         # loads Titanic once, profiles, cleans, EDA charts + titanic.csv
python 02_modeling.py    # reads the same titanic.csv, models, evaluates, saves pipeline
```

### 3. `/support_assistant`
_Coming next — see `support_assistant/README.md` once added._

## Design decisions summary

**Known source-data artifact.** One book title ("Full Moon over Noah's Ark...")
renders with a corrupted apostrophe (`â` instead of `'`) regardless of decoding
strategy. This was confirmed to be baked into books.toscrape.com's own HTML —
independent scrapes of the same site by other people show the identical
corruption — rather than a bug in this pipeline's request/decode handling.
It was left as-is rather than pattern-matched and silently rewritten, since
"fixing" source data invisibly can hide genuine upstream data-quality issues
from downstream consumers.

See each module's own README for detailed reasoning. Short version:
- **`/data_pipeline`**: scrapes [books.toscrape.com](https://books.toscrape.com), a site built
  specifically for scraping practice, as a free/legal stand-in for a Zepto-style product
  catalog. Data is normalized into `categories` + `books` tables sharing a PK/FK
  relationship, with price converted from GBP to INR via a fixed project-defined rate
  (105.50) — see `data_pipeline/README.md` for the full imputation and schema rationale.
- **`/analytics`**: profiles and cleans the Titanic dataset (`deck` dropped at 77%
  missing, `age` median-imputed at 20% missing, `embarked` rows dropped at 0.2%
  missing), then trains and compares Logistic Regression, Decision Tree, and Random
  Forest classifiers plus a fare-prediction regression — all preprocessing fit on the
  training split only. Recommends Random Forest (highest F1) — see
  `analytics/README.md` for the full comparison table and reasoning.

## Git workflow

This repo's history includes a feature branch (`feature/data-pipeline`) created, committed
to at least twice, and merged into `main` — visible via `git log --graph --all`.