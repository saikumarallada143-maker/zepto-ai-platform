"""
src/cleaning.py
The Task-2 missing-value-handling logic, factored out so both 01_eda.py and
02_modeling.py apply the *identical* cleaning to the *identical* raw data —
satisfying "clean it once" without needing a second, non-required CSV
artifact. 02_modeling.py reads the one committed titanic.csv (raw, per Task 1)
and calls clean_titanic() itself rather than re-deciding the cleaning policy.

Policy (percentages measured on the real seaborn titanic dataset, 891 rows):
  - deck (~77.2% missing):        drop the column (imputing ~3/4 fabricated
                                   values would swamp any real signal; the
                                   partial signal it carries is already
                                   captured, with far less missingness, by
                                   pclass/fare).
  - embarked/embark_town (~0.22%): drop those rows (<5% threshold; negligible
                                   data loss, avoids inventing a port).
  - age (~19.87%):                median-impute (5-30% threshold; median
                                   chosen over mean since age is right-skewed
                                   by older outliers).
"""


def clean_titanic(df):
    """df: raw DataFrame straight from sns.load_dataset('titanic') or
    pd.read_csv('titanic.csv'). Returns the cleaned DataFrame plus a dict of
    the missing-percentages measured (for reporting)."""
    missing_pct = {
        "deck": df["deck"].isna().mean() * 100,
        "embarked": df["embarked"].isna().mean() * 100,
        "age": df["age"].isna().mean() * 100,
    }

    df_clean = df.drop(columns=["deck"])
    df_clean = df_clean.dropna(subset=["embarked", "embark_town"])
    age_median = df_clean["age"].median()
    df_clean["age"] = df_clean["age"].fillna(age_median)

    return df_clean, missing_pct, age_median
