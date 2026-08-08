-- schema.sql
-- Normalized two-table schema, matching the module spec's PK/FK requirement.
-- categories is separated from books so each category name is stored once
-- (3NF-style), not repeated on every book row.

CREATE TABLE IF NOT EXISTS categories (
    category_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS books (
    book_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    product_url  TEXT UNIQUE NOT NULL,   -- natural key from source site; used to dedupe re-runs
    title        TEXT NOT NULL,
    price_gbp    REAL NOT NULL,
    price_inr    REAL NOT NULL,          -- price_gbp * FIXED_GBP_TO_INR_RATE (105.50)
    rating       INTEGER NOT NULL,       -- 1-5
    in_stock     INTEGER NOT NULL,       -- 0/1 boolean
    category_id  INTEGER NOT NULL REFERENCES categories(category_id)
);

CREATE INDEX IF NOT EXISTS idx_books_category ON books(category_id);
