"""
load.py
Loads a cleaned DataFrame into the SQLite store via Python's sqlite3 module
(parameterized inserts). Re-running the pipeline on the same book
(same product_url) updates it instead of creating a duplicate row.
"""


def upsert_category(conn, category_name):
    conn.execute(
        "INSERT OR IGNORE INTO categories (category_name) VALUES (?)",
        (category_name,),
    )
    row = conn.execute(
        "SELECT category_id FROM categories WHERE category_name = ?",
        (category_name,),
    ).fetchone()
    return row[0]


def load_books(conn, cleaned_df):
    loaded = 0
    for _, row in cleaned_df.iterrows():
        category_id = upsert_category(conn, row["category"])
        conn.execute(
            """
            INSERT INTO books
                (product_url, title, price_gbp, price_inr, rating, in_stock, category_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(product_url) DO UPDATE SET
                title=excluded.title,
                price_gbp=excluded.price_gbp,
                price_inr=excluded.price_inr,
                rating=excluded.rating,
                in_stock=excluded.in_stock,
                category_id=excluded.category_id
            """,
            (
                row["product_url"],
                row["title"],
                row["price_gbp"],
                row["price_inr"],
                int(row["rating"]),
                int(row["in_stock"]),
                category_id,
            ),
        )
        loaded += 1
    conn.commit()
    return loaded
