# Zepto Data & AI Platform — Capstone Project

One connected platform, three modules, one repository:

| Module | Marks | What it does |
|---|---|---|
| `/data_pipeline` | 25 | Scrapes a product catalog, cleans it, and loads it into a normalized SQLite store |
| `/analytics` | 50 | Profiles and models a customer-style dataset end to end |
| `/support_assistant` | 25 | Grounded GenAI assistant answering policy questions from Zepto's own documents |

> Status: `/data_pipeline` complete. `/analytics` and `/support_assistant` in progress.

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
python -m src.pipeline          # full run
python verify.py                # sanity-check the loaded DB
```

### 2. `/analytics`
_Coming next — see `analytics/README.md` once added._

### 3. `/support_assistant`
_Coming next — see `support_assistant/README.md` once added._

## Design decisions summary

See each module's own README for detailed reasoning. Short version:
- **`/data_pipeline`**: scrapes [books.toscrape.com](https://books.toscrape.com), a site built
  specifically for scraping practice, as a free/legal stand-in for a Zepto-style product
  catalog. Data is normalized into `categories` + `products` tables in SQLite, with a
  `scrape_runs` log table for pipeline observability.

## Git workflow

This repo's history includes a feature branch (`feature/data-pipeline`) created, committed
to at least twice, and merged into `main` — visible via `git log --graph --all`.
