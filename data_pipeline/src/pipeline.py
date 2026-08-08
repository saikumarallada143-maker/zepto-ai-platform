"""
pipeline.py
Single entry point for the data_pipeline module: scrape -> clean -> load.

Usage:
    python -m src.pipeline
"""

from src.scraper import scrape_all
from src.clean import clean_books
from src.db import get_connection, init_schema
from src.load import load_books


def run():
    print("Step 1/3: Scraping books.toscrape.com ...")
    raw_books = scrape_all()

    print("\nStep 2/3: Cleaning + converting currency ...")
    cleaned = clean_books(raw_books)
    print(f"{len(cleaned)} clean rows ready to load "
          f"({len(raw_books) - len(cleaned)} dropped during cleaning).")

    print("\nStep 3/3: Loading into SQLite ...")
    conn = get_connection()
    init_schema(conn)
    loaded = load_books(conn, cleaned)
    conn.close()
    print(f"Loaded {loaded} books into data/zepto_books.db")
    print("\nPipeline run complete. Next: python -m src.queries")


if __name__ == "__main__":
    run()
