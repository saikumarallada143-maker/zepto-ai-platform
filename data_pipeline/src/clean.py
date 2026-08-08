"""
clean.py
Converts raw scraped rows into typed, analysis-ready records.

Missing/unparseable-value policy (stated here, and in the README, per the
module spec's requirement to "state and justify your choice"):

  - title:      required, non-numeric identifier. If missing/empty, the row
                is unusable (we can't identify what book it is) -> DROP.
  - price_gbp:  numeric. If the raw text doesn't parse to a number, we
                MEDIAN-IMPUTE using the median of all successfully-parsed
                prices in the same scrape batch. Rationale: a parse failure
                here is almost always a formatting quirk, not a genuinely
                missing price, and dropping the row would needlessly lose an
                otherwise-valid book. Median (not mean) is used because
                prices are right-skewed and the median is robust to outliers.
  - rating:     numeric-ordinal (1-5). Same reasoning as price -> MEDIAN-IMPUTE
                (rounded to the nearest integer, since ratings are discrete).
  - in_stock:   categorical, not numeric, so imputation doesn't make sense
                the same way. If availability text doesn't match either known
                pattern, we DROP the row rather than guess a stock status.

FIXED_GBP_TO_INR_RATE is the project-defined constant required by the module
spec (not a live/historical market rate) — see README for the exact value.
"""

import re
import pandas as pd

STAR_WORD_TO_INT = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}

# Required fixed baseline conversion rate (project-defined constant).
FIXED_GBP_TO_INR_RATE = 105.50


def _parse_price(price_raw):
    """'£51.77' -> 51.77 (float). Returns None if unparseable."""
    match = re.search(r"[\d.]+", price_raw or "")
    return float(match.group()) if match else None


def _parse_rating(star_rating):
    """'Three' -> 3 (int). Returns None if unparseable."""
    return STAR_WORD_TO_INT.get(star_rating)


def _parse_in_stock(availability_raw):
    """
    'In stock (22 available)' / 'In stock' -> True
    'Out of stock' -> False
    Anything else -> None (unparseable; row will be dropped).
    """
    text = (availability_raw or "").strip().lower()
    if text.startswith("in stock"):
        return True
    if text.startswith("out of stock"):
        return False
    return None


def clean_books(raw_books):
    """
    raw_books: list[dict] straight from scraper.py.
    Returns a cleaned pandas DataFrame with columns:
        title, category, price_gbp, price_inr, rating, in_stock, product_url
    Never raises on a single bad row — applies the imputation/drop policy above.
    """
    df = pd.DataFrame(raw_books)
    if df.empty:
        return df

    df["title"] = df["title"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.strip()

    # --- Drop rows with no usable title (can't identify the product) ---
    df = df[df["title"].str.len() > 0].copy()

    # --- Parse numeric-ish fields ---
    df["price_gbp"] = df["price_raw"].apply(_parse_price)
    df["rating"] = df["star_rating"].apply(_parse_rating)
    df["in_stock"] = df["availability_raw"].apply(_parse_in_stock)

    # --- Drop rows where in_stock couldn't be determined (categorical -> drop) ---
    n_before = len(df)
    df = df[df["in_stock"].notna()].copy()
    if len(df) < n_before:
        print(f"Dropped {n_before - len(df)} row(s) with unparseable availability text.")

    # --- Median-impute numeric fields (price_gbp, rating) ---
    for col, is_int in (("price_gbp", False), ("rating", True)):
        n_missing = df[col].isna().sum()
        if n_missing:
            median_val = df[col].median()
            if is_int:
                median_val = round(median_val)
            df[col] = df[col].fillna(median_val)
            print(f"Median-imputed {n_missing} missing '{col}' value(s) with {median_val}.")

    df["rating"] = df["rating"].astype(int)
    df["in_stock"] = df["in_stock"].astype(bool)

    # --- Fixed-rate currency conversion (required baseline, no API call) ---
    df["price_inr"] = (df["price_gbp"] * FIXED_GBP_TO_INR_RATE).round(2)

    # --- Drop exact duplicate books (same product page scraped twice) ---
    df = df.drop_duplicates(subset=["product_url"])

    cleaned = df[
        ["product_url", "title", "category", "price_gbp", "price_inr", "rating", "in_stock"]
    ].reset_index(drop=True)

    return cleaned
