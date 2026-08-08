"""
scraper.py
Scrapes book listings from books.toscrape.com — a site built specifically
for scraping practice — using requests + BeautifulSoup, per the module spec.

Design decision: we crawl *by category* rather than the flat "All products"
listing. Each category page already tells us the category for every book on
it, so we get title/price/rating/availability/category in one pass with no
extra per-product request. We keep pulling categories (in nav order) until
we've covered >=3 categories AND collected >=60 books, per the acceptance
criteria — instead of hardcoding a fixed category list, which could fall
short of 60 if a chosen category happens to be small.
"""

import time
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://books.toscrape.com/"
REQUEST_DELAY_SECONDS = 0.5  # polite delay between requests
MIN_BOOKS = 60
MIN_CATEGORIES = 3


def get_session():
    session = requests.Session()
    session.headers.update(
        {"User-Agent": "ZeptoCapstoneBot/1.0 (educational scraping project)"}
    )
    return session


def get_categories(session):
    """Return [(category_name, category_url), ...] from the left nav, in page order."""
    resp = session.get(BASE_URL, timeout=10)
    resp.raise_for_status()
    resp.encoding = "utf-8"
    soup = BeautifulSoup(resp.text, "html.parser")
    nav_links = soup.select("div.side_categories ul li ul li a")
    return [
        (a.get_text(strip=True), requests.compat.urljoin(BASE_URL, a["href"]))
        for a in nav_links
    ]


def parse_listing_page(html, category_name):
    """Extract raw book dicts from one catalog listing page (one category, one page)."""
    soup = BeautifulSoup(html, "html.parser")
    books = []
    for pod in soup.select("article.product_pod"):
        title = pod.h3.a["title"]
        price_raw = pod.select_one("p.price_color").get_text(strip=True)
        availability_raw = pod.select_one("p.instock.availability").get_text(strip=True)
        rating_classes = pod.select_one("p.star-rating")["class"]
        star_rating = rating_classes[1] if len(rating_classes) > 1 else None
        product_url = requests.compat.urljoin(
            BASE_URL + "catalogue/", pod.h3.a["href"]
        )
        books.append(
            {
                "title": title,
                "price_raw": price_raw,          # e.g. "£51.77"
                "star_rating": star_rating,       # e.g. "Three"
                "availability_raw": availability_raw,  # e.g. "In stock (22 available)"
                "category": category_name,
                "product_url": product_url,
            }
        )
    return books


def get_next_page_url(html, current_url):
    soup = BeautifulSoup(html, "html.parser")
    next_link = soup.select_one("li.next a")
    return requests.compat.urljoin(current_url, next_link["href"]) if next_link else None


def scrape_category(session, category_name, category_url):
    """Yield raw book dicts for every page in a category, following pagination."""
    url = category_url
    while url:
        resp = session.get(url, timeout=10)
        resp.raise_for_status()
        html = resp.text
        yield from parse_listing_page(html, category_name)
        url = get_next_page_url(html, url)
        time.sleep(REQUEST_DELAY_SECONDS)


def scrape_all(min_books=MIN_BOOKS, min_categories=MIN_CATEGORIES):
    """
    Scrape categories in nav order until we've covered >= min_categories
    categories AND collected >= min_books books. Returns list[dict].
    """
    session = get_session()
    categories = get_categories(session)

    all_books = []
    categories_scraped = 0

    for name, url in categories:
        print(f"Scraping category: {name}")
        all_books.extend(list(scrape_category(session, name, url)))
        categories_scraped += 1

        if len(all_books) >= min_books and categories_scraped >= min_categories:
            break

    print(
        f"Done: {len(all_books)} raw rows across {categories_scraped} categories "
        f"(target was >= {min_books} books, >= {min_categories} categories)."
    )
    return all_books
