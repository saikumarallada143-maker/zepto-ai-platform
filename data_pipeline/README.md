# `/data_pipeline` — Zepto Data & AI Platform

Scrapes a live product catalog, cleans and types it, converts currency using a fixed
project-defined rate, and loads it into a normalized SQLite database that's then
queried with both SQL and pandas.

**Data source:** [books.toscrape.com](https://books.toscrape.com) — a public site built
specifically for scraping practice (no login, no API key, no paid tier). The catalog is
books rather than groceries; the exercise is the pipeline mechanics
(scrape → clean → convert → store → query), which are identical regardless of category.

## How to run

```bash
cd data_pipeline
pip install -r requirements.txt

python -m src.pipeline    # scrape -> clean -> load into data/zepto_books.db
python -m src.queries     # run the 5 required SQL queries + pandas verification,
                           # writes query_results.md
```

Running `pipeline` regenerates `data/zepto_books.db` from scratch each time
(re-running is idempotent — books are upserted by their source URL, not duplicated).

## Design decisions

**Scraping strategy.** The scraper crawls *by category* rather than the flat
"All products" listing. Each category page already states the category for every
book on it, so we get `title`/`price`/`rating`/`availability`/`category` in one pass,
with no extra per-book request. It walks categories in nav order and stops once it has
covered **≥ 3 categories** and collected **≥ 60 books** (rather than hardcoding a fixed
category list, which could fall short of 60 if a chosen category happened to be small).

**Missing / unparseable value policy** (applied in `src/clean.py`):
| Field | If unparseable | Why |
|---|---|---|
| `title` | Drop row | Non-numeric identifier — if we can't read the title, the row isn't usable. |
| `price_gbp` | Median-impute | Parse failures here are almost always a formatting quirk, not a genuinely missing price. Median (not mean) is robust to the right-skew typical of price data. |
| `rating` | Median-impute, rounded to nearest int | Same reasoning as price; ratings are discrete (1–5) so the median is rounded. |
| `in_stock` | Drop row | Categorical, not numeric — imputing a stock status would mean guessing a fact, not estimating a continuous value, so we drop instead. |

**Currency conversion.** `price_inr = price_gbp * 105.50`. This fixed rate is the
project-defined baseline constant required by the module spec — not a live or
historical market rate, so no API call or date reference is needed. (The optional
live-rate-with-fallback stretch goal was intentionally skipped: it's explicitly ungraded
and doesn't affect the required `price_inr` column, so implementing it would only add
untested surface area without improving what's graded.)

**Schema.** Two tables sharing a PK/FK relationship:
```
categories(category_id PK, category_name UNIQUE)
books(book_id PK, product_url UNIQUE, title, price_gbp, price_inr, rating, in_stock, category_id FK -> categories)
```
`product_url` is kept as a natural key (not required by the spec, but needed so
re-running the pipeline updates existing rows instead of duplicating them).

**The 5 required SQL queries** (in `src/queries.py`) collectively cover
`SELECT`/`WHERE`, `ORDER BY`, `LIMIT`, `DISTINCT`, `BETWEEN`, `IN`, and a `JOIN`
(top-10-highest-rated-books-per-category, via a window function). Query strings and
their output are both printed to the console and written to `query_results.md` for a
grader to review without re-running anything.

**pandas verification.** `src/queries.py` reads the query results back with
`pd.read_sql(...)`, then separately reproduces the JOIN query's result using
`books_df.merge(categories_df, on="category_id")` plus `groupby(...).head(10)` —
no SQL involved — and asserts the two outputs are identical row-for-row.

## A note on how this was tested

This module was built with AI assistance (permitted per the program's guidelines,
provided the implementation is understood and can be explained). The scraping,
cleaning, schema, loading, and query logic were verified end-to-end against
**synthetic data shaped like real scraper output** — including deliberately broken
rows (bad price text, bad rating text, bad availability text, a missing title) to
exercise the imputation/drop policy above, and the `pd.read_sql` vs `pd.merge`
equivalence check passed on that synthetic run. The live scrape against
books.toscrape.com should be (re-)run locally before submission to produce the real
`data/zepto_books.db` and `query_results.md` from actual site data.

## Git workflow

The overall repository (not just this module) includes a feature branch created,
committed to at least twice, and merged into `main` — see root README / `git log --graph --all`.
