# Data Pipeline — Query Results

## Q1 - WHERE / ORDER BY / LIMIT

```sql
-- Q1: SELECT/WHERE + ORDER BY + LIMIT
-- Cheapest 10 in-stock books.
SELECT title, price_gbp, price_inr, rating
FROM books
WHERE in_stock = 1
ORDER BY price_gbp ASC
LIMIT 10;
```

| title | price_gbp | price_inr | rating |
|---|---|---|---|
| Tastes Like Fear (DI Marnie Rome #3) | 10.69 | 1127.79 | 1 |
| Hide Away (Eve Duncan #20) | 11.84 | 1249.12 | 1 |
| The Girl You Lost | 12.29 | 1296.59 | 5 |
| Playing with Fire | 13.71 | 1446.41 | 3 |
| That Darkness (Gardiner and Renner #1) | 13.92 | 1468.56 | 1 |
| The Girl In The Ice (DCI Erika Foster #1) | 15.85 | 1672.18 | 3 |
| The Constant Princess (The Tudor Court #1) | 16.62 | 1753.41 | 3 |
| A Murder in Time | 16.64 | 1755.52 | 1 |
| A Study in Scarlet (Sherlock Holmes #1) | 16.73 | 1765.02 | 2 |
| A Spy's Devotion (The Regency Spies of London #1) | 16.97 | 1790.33 | 5 |

## Q2 - DISTINCT

```sql
-- Q2: DISTINCT
-- Which star ratings actually occur in the data?
SELECT DISTINCT rating
FROM books
ORDER BY rating;
```

| rating |
|---|
| 1 |
| 2 |
| 3 |
| 4 |
| 5 |

## Q3 - BETWEEN

```sql
-- Q3: BETWEEN
-- Mid-range priced books (GBP 20-40).
SELECT title, price_gbp, price_inr
FROM books
WHERE price_gbp BETWEEN 20 AND 40
ORDER BY price_gbp;
```

| title | price_gbp | price_inr |
|---|---|---|
| Blood Defense (Samantha Brinkman #1) | 20.3 | 2141.65 |
| Love, Lies and Spies | 20.55 | 2168.02 |
| Between Shades of Gray | 20.79 | 2193.34 |
| Delivering the Truth (Quaker Midwife Mystery #1) | 20.89 | 2203.9 |
| Voyager (Outlander #3) | 21.07 | 2222.89 |
| The Silkworm (Cormoran Strike #2) | 23.05 | 2431.78 |
| The Road to Little Dribbling: Adventures of an American in Britain (Notes From a Small Island #2) | 23.21 | 2448.66 |
| Career of Evil (Cormoran Strike #3) | 24.72 | 2607.96 |
| The Mysterious Affair at Styles (Hercule Poirot #1) | 24.8 | 2616.4 |
| What Happened on Beale Street (Secrets of the South Mysteries #2) | 25.37 | 2676.54 |
| Extreme Prey (Lucas Davenport #26) | 25.4 | 2679.7 |
| Starlark | 25.83 | 2725.06 |
| 1,000 Places to See Before You Die | 26.08 | 2751.44 |
| Girl With a Pearl Earring | 26.77 | 2824.24 |
| Poisonous (Max Revere Novels #3) | 26.8 | 2827.4 |

_... 18 more row(s) not shown ..._

## Q4 - IN

```sql
-- Q4: IN
-- Highly-rated books (4 or 5 stars).
SELECT title, rating, price_gbp
FROM books
WHERE rating IN (4, 5)
ORDER BY rating DESC, price_gbp DESC;
```

| title | rating | price_gbp |
|---|---|---|
| A Flight of Arrows (The Pathfinders #2) | 5 | 55.53 |
| The Bachelor Girl's Guide to Murder (Herringford and Watts Mysteries #1) | 5 | 52.3 |
| A Time of Torment (Charlie Parker #14) | 5 | 48.35 |
| While You Were Mine | 5 | 41.32 |
| The Red Tent | 5 | 35.66 |
| Mrs. Houdini | 5 | 30.25 |
| The Passion of Dolssa | 5 | 28.32 |
| 1,000 Places to See Before You Die | 5 | 26.08 |
| What Happened on Beale Street (Secrets of the South Mysteries #2) | 5 | 25.37 |
| The Silkworm (Cormoran Strike #2) | 5 | 23.05 |
| Voyager (Outlander #3) | 5 | 21.07 |
| Between Shades of Gray | 5 | 20.79 |
| A Spy's Devotion (The Regency Spies of London #1) | 5 | 16.97 |
| The Girl You Lost | 5 | 12.29 |
| The No. 1 Ladies' Detective Agency (No. 1 Ladies' Detective Agency #1) | 4 | 57.7 |

_... 12 more row(s) not shown ..._

## Q5 - JOIN (top 10 rated books per category)

```sql
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
```

| category_name | title | rating | price_gbp |
|---|---|---|---|
| Historical Fiction | A Flight of Arrows (The Pathfinders #2) | 5 | 55.53 |
| Historical Fiction | While You Were Mine | 5 | 41.32 |
| Historical Fiction | The Red Tent | 5 | 35.66 |
| Historical Fiction | Mrs. Houdini | 5 | 30.25 |
| Historical Fiction | The Passion of Dolssa | 5 | 28.32 |
| Historical Fiction | Voyager (Outlander #3) | 5 | 21.07 |
| Historical Fiction | Between Shades of Gray | 5 | 20.79 |
| Historical Fiction | A Spy's Devotion (The Regency Spies of London #1) | 5 | 16.97 |
| Historical Fiction | A Paris Apartment | 4 | 39.01 |
| Historical Fiction | World Without End (The Pillars of the Earth #2) | 4 | 32.97 |
| Mystery | The Bachelor Girl's Guide to Murder (Herringford and Watts Mysteries #1) | 5 | 52.3 |
| Mystery | A Time of Torment (Charlie Parker #14) | 5 | 48.35 |
| Mystery | What Happened on Beale Street (Secrets of the South Mysteries #2) | 5 | 25.37 |
| Mystery | The Silkworm (Cormoran Strike #2) | 5 | 23.05 |
| Mystery | The Girl You Lost | 5 | 12.29 |

_... 15 more row(s) not shown ..._

## pd.merge reproduction of Q5 (no SQL)

| category_name | title | rating | price_gbp |
|---|---|---|---|
| Historical Fiction | A Flight of Arrows (The Pathfinders #2) | 5 | 55.53 |
| Historical Fiction | While You Were Mine | 5 | 41.32 |
| Historical Fiction | The Red Tent | 5 | 35.66 |
| Historical Fiction | Mrs. Houdini | 5 | 30.25 |
| Historical Fiction | The Passion of Dolssa | 5 | 28.32 |
| Historical Fiction | Voyager (Outlander #3) | 5 | 21.07 |
| Historical Fiction | Between Shades of Gray | 5 | 20.79 |
| Historical Fiction | A Spy's Devotion (The Regency Spies of London #1) | 5 | 16.97 |
| Historical Fiction | A Paris Apartment | 4 | 39.01 |
| Historical Fiction | World Without End (The Pillars of the Earth #2) | 4 | 32.97 |
| Mystery | The Bachelor Girl's Guide to Murder (Herringford and Watts Mysteries #1) | 5 | 52.3 |
| Mystery | A Time of Torment (Charlie Parker #14) | 5 | 48.35 |
| Mystery | What Happened on Beale Street (Secrets of the South Mysteries #2) | 5 | 25.37 |
| Mystery | The Silkworm (Cormoran Strike #2) | 5 | 23.05 |
| Mystery | The Girl You Lost | 5 | 12.29 |

_... 15 more row(s) not shown ..._


**SQL result and pd.merge result are identical: `True`**
