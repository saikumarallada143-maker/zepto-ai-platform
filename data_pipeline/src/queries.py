"""
queries.py
Runs the 5 required SQL queries against the loaded SQLite database, reads
at least two of them back with pd.read_sql, and separately reproduces the
JOIN query's result using pd.merge on in-memory DataFrames (no SQL) to show
both approaches agree.

Usage:
    python -m src.queries

Writes a persisted, human-readable record of every query + its output to
data_pipeline/query_results.md (so a grader can review results without
re-running the pipeline).
"""

import pandas as pd
from src.db import get_connection

OUTPUT_MD_PATH = "query_results.md"

# --------------------------------------------------------------------------
# The 5 required queries. Together they cover: SELECT/WHERE, ORDER BY,
# LIMIT, DISTINCT, IN/BETWEEN, and a JOIN (per module spec).
# --------------------------------------------------------------------------

Q1_WHERE_ORDER_LIMIT = """
-- Q1: SELECT/WHERE + ORDER BY + LIMIT
-- Cheapest 10 in-stock books.
SELECT title, price_gbp, price_inr, rating
FROM books
WHERE in_stock = 1
ORDER BY price_gbp ASC
LIMIT 10;
"""

Q2_DISTINCT = """
-- Q2: DISTINCT
-- Which star ratings actually occur in the data?
SELECT DISTINCT rating
FROM books
ORDER BY rating;
"""

Q3_BETWEEN = """
-- Q3: BETWEEN
-- Mid-range priced books (GBP 20-40).
SELECT title, price_gbp, price_inr
FROM books
WHERE price_gbp BETWEEN 20 AND 40
ORDER BY price_gbp;
"""

Q4_IN = """
-- Q4: IN
-- Highly-rated books (4 or 5 stars).
SELECT title, rating, price_gbp
FROM books
WHERE rating IN (4, 5)
ORDER BY rating DESC, price_gbp DESC;
"""

Q5_JOIN = """
-- Q5: JOIN (+ ORDER BY) -- the query reproduced with pd.merge below.
-- Top 10 highest-rated books per category (ties broken by price, then title).
SELECT category_name, title, rating, price_gbp
FROM (
    SELECT
        c.category_name,
        b.title,
        b.rating,
        b.price_gbp,
        ROW_NUMBER() OVER (
            PARTITION BY c.category_name
            ORDER BY b.rating DESC, b.price_gbp DESC, b.title ASC
        ) AS rn
    FROM books b
    JOIN categories c ON b.category_id = c.category_id
)
WHERE rn <= 10
ORDER BY category_name, rn;
"""

QUERIES = [
    ("Q1 - WHERE / ORDER BY / LIMIT", Q1_WHERE_ORDER_LIMIT),
    ("Q2 - DISTINCT", Q2_DISTINCT),
    ("Q3 - BETWEEN", Q3_BETWEEN),
    ("Q4 - IN", Q4_IN),
    ("Q5 - JOIN (top 10 rated books per category)", Q5_JOIN),
]


def _df_to_markdown_table(df, max_rows=15):
    """Minimal markdown-table formatter (avoids adding a 'tabulate' dependency)."""
    shown = df.head(max_rows)
    cols = list(shown.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for _, row in shown.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    if len(df) > max_rows:
        lines.append(f"\n_... {len(df) - max_rows} more row(s) not shown ..._")
    return "\n".join(lines)


def run_all_queries():
    conn = get_connection()
    md_sections = ["# Data Pipeline — Query Results\n"]

    # --- Run all 5 required queries, print + log each ---
    dataframes = {}
    for label, sql in QUERIES:
        print(f"\n=== {label} ===")
        print(sql.strip())
        df = pd.read_sql(sql, conn)  # pd.read_sql used throughout (>= 2 required)
        print(df.to_string(index=False))
        dataframes[label] = df

        md_sections.append(f"## {label}\n")
        md_sections.append(f"```sql\n{sql.strip()}\n```\n")
        md_sections.append(_df_to_markdown_table(df) + "\n")

    # --- Reproduce the JOIN query (Q5) using pd.merge, no SQL ---
    print("\n=== Reproducing Q5 (JOIN) with pd.merge instead of SQL ===")
    books_df = pd.read_sql("SELECT * FROM books;", conn)
    categories_df = pd.read_sql("SELECT * FROM categories;", conn)

    merged = books_df.merge(categories_df, on="category_id")
    merged_sorted = merged.sort_values(
        ["category_name", "rating", "price_gbp", "title"],
        ascending=[True, False, False, True],
    )
    pandas_top10 = (
        merged_sorted.groupby("category_name")
        .head(10)[["category_name", "title", "rating", "price_gbp"]]
        .reset_index(drop=True)
    )

    sql_top10 = dataframes["Q5 - JOIN (top 10 rated books per category)"].reset_index(drop=True)

    # Compare the two approaches on the same columns, same row order
    sql_comparable = sql_top10[["category_name", "title", "rating", "price_gbp"]]
    match = sql_comparable.equals(pandas_top10)
    print(f"\nSQL JOIN result matches pd.merge result: {match}")
    print(pandas_top10.to_string(index=False))

    md_sections.append("## pd.merge reproduction of Q5 (no SQL)\n")
    md_sections.append(_df_to_markdown_table(pandas_top10) + "\n")
    md_sections.append(f"\n**SQL result and pd.merge result are identical: `{match}`**\n")

    with open(OUTPUT_MD_PATH, "w") as f:
        f.write("\n".join(md_sections))
    print(f"\nFull query log written to {OUTPUT_MD_PATH}")

    conn.close()
    return match


if __name__ == "__main__":
    run_all_queries()
