"""
01_eda.py — /analytics Part A (Tasks 1-6)

Loads the Titanic dataset ONCE via seaborn, profiles it, cleans it per the
required percentage-threshold rule, runs univariate/bivariate/multivariate
analysis, and does an EDA-stage standardization sanity check.

Saves:
  - titanic.csv                  (the one committed offline fallback, Task 1)
  - charts/*.png                 (every chart referenced in eda_findings.md)
  - eda_findings.md              (every required written interpretation)

Usage:
    python 01_eda.py
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")  # no display needed, just save PNGs
import matplotlib.pyplot as plt

from src.cleaning import clean_titanic

CHARTS_DIR = "charts"
FINDINGS_PATH = "eda_findings.md"

findings = ["# EDA Findings — /analytics\n"]


def log(md_text, also_print=True):
    findings.append(md_text)
    if also_print:
        print(md_text)


# ---------------------------------------------------------------------------
# Task 1: Load ONCE, profile, save the offline-fallback CSV
# ---------------------------------------------------------------------------
df = sns.load_dataset("titanic")  # the one and only sns.load_dataset call in the module

print("=== df.info() ===")
df.info()
print("\n=== df.describe() ===")
print(df.describe())
print("\n=== df.shape ===", df.shape)

missing_pct = (df.isna().mean() * 100).sort_values(ascending=False)
missing_pct = missing_pct[missing_pct > 0]
print("\n=== Missing value % (columns with any missing) ===")
print(missing_pct)

df.to_csv("titanic.csv", index=False)
print("\nSaved titanic.csv (committed offline fallback) — shape:", df.shape)

log("## Task 1 — Profile\n")
log(f"Shape: `{df.shape}`. Columns with any missing values:\n")
log(missing_pct.to_frame("missing_%").to_markdown())
log("")

# ---------------------------------------------------------------------------
# Task 2: Missing-value handling per the threshold rule
#   <5% missing  -> drop those rows
#   5-30%        -> impute
#   very high (>~30%, unreliable to impute) -> explicit drop-column or
#                   "missing"-category decision, justified in writing
# ---------------------------------------------------------------------------
n_before = len(df)
df_clean, missing_pct, age_median = clean_titanic(df)
n_dropped_embarked = n_before - len(df_clean)
deck_missing_pct = missing_pct["deck"]
embarked_missing_pct = missing_pct["embarked"]
age_missing_pct = missing_pct["age"]

log("## Task 2 — Missing-value handling\n")
log(
    "(Cleaning logic lives in `src/cleaning.py` — `02_modeling.py` calls the same "
    "function on the same raw `titanic.csv`, so both stages apply identical cleaning "
    "without re-deciding the policy or needing a second CSV.)\n"
)
log(
    f"- `deck`: **{deck_missing_pct:.2f}%** missing — far above the reliable-impute "
    f"range. **Decision: drop the column.** At this missing rate, imputation would "
    f"fabricate the large majority of the column's values, and the partial signal "
    f"it carries (cabin location as a wealth/class proxy) is already captured, with "
    f"far less missingness, by `pclass` and `fare`.\n"
)
log(
    f"- `embarked` / `embark_town`: **{embarked_missing_pct:.2f}%** missing "
    f"(< 5%) — **decision: drop those rows** ({n_dropped_embarked} rows removed). "
    f"At this rate, dropping costs negligible data and avoids inventing a port of "
    f"embarkation for the 2 affected passengers.\n"
)
log(
    f"- `age`: **{age_missing_pct:.2f}%** missing (5-30% range) — **decision: impute** "
    f"with the column median ({age_median:.1f} years). Median chosen over mean because "
    f"age is right-skewed by older outliers, and median is robust to that skew.\n"
)
log(f"Shape after cleaning: `{df_clean.shape}`\n")

# ---------------------------------------------------------------------------
# Task 3: Univariate analysis — age & fare
# ---------------------------------------------------------------------------
def iqr_outlier_count(series):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return int(((series < lower) | (series > upper)).sum()), lower, upper


for col in ["age", "fare"]:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(df_clean[col], bins=30, color="#4C72B0", edgecolor="white")
    axes[0].set_title(f"{col} — histogram")
    axes[1].boxplot(df_clean[col], vert=True)
    axes[1].set_title(f"{col} — box plot")
    fig.tight_layout()
    fig.savefig(f"{CHARTS_DIR}/univariate_{col}.png", dpi=120)
    plt.close(fig)

age_outliers, age_lo, age_hi = iqr_outlier_count(df_clean["age"])
fare_outliers, fare_lo, fare_hi = iqr_outlier_count(df_clean["fare"])

fare_mean = df_clean["fare"].mean()
fare_median = df_clean["fare"].median()
fare_mode = df_clean["fare"].mode().iloc[0]
skew_direction = (
    "right-skewed" if fare_mean > fare_median > fare_mode
    else "left-skewed" if fare_mean < fare_median < fare_mode
    else "roughly symmetric"
)

log("## Task 3 — Univariate analysis (age, fare)\n")
log(f"- `age` IQR outliers: **{age_outliers}** (outside [{age_lo:.1f}, {age_hi:.1f}])")
log(f"- `fare` IQR outliers: **{fare_outliers}** (outside [{fare_lo:.1f}, {fare_hi:.1f}])\n")
log(
    f"- `fare`: mean = **{fare_mean:.2f}**, median = **{fare_median:.2f}**, "
    f"mode = **{fare_mode:.2f}**. Since mean > median > mode, `fare` is "
    f"**{skew_direction}** — a small number of very expensive tickets pull the "
    f"mean well above the typical (median) fare.\n"
)
log("![age univariate](charts/univariate_age.png)")
log("![fare univariate](charts/univariate_fare.png)\n")

# ---------------------------------------------------------------------------
# Task 4: Bivariate analysis — boolean masking + correlation heatmap
# ---------------------------------------------------------------------------
survival_by_sex = df_clean.groupby("sex")["survived"].mean()
survival_by_pclass = df_clean.groupby("pclass")["survived"].mean()
survival_by_sex_pclass = df_clean.groupby(["sex", "pclass"])["survived"].mean()

# Same numbers via explicit boolean masking (&/|), per the task's requirement:
female_survival = df_clean.loc[df_clean["sex"] == "female", "survived"].mean()
male_survival = df_clean.loc[df_clean["sex"] == "male", "survived"].mean()
female_p1_survival = df_clean.loc[
    (df_clean["sex"] == "female") & (df_clean["pclass"] == 1), "survived"
].mean()

corr_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
corr_matrix = df_clean[corr_cols].corr()

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Correlation matrix (6 numeric columns)")
fig.tight_layout()
fig.savefig(f"{CHARTS_DIR}/correlation_heatmap.png", dpi=120)
plt.close(fig)

# Top 2 off-diagonal pairs by absolute correlation
corr_pairs = (
    corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    .stack()
    .rename("corr")
    .reset_index()
)
corr_pairs["abs_corr"] = corr_pairs["corr"].abs()
top2 = corr_pairs.sort_values("abs_corr", ascending=False).head(2)

log("## Task 4 — Bivariate analysis\n")
log(f"- Survival rate by sex: female = **{female_survival:.3f}**, male = **{male_survival:.3f}**")
log("\nSurvival rate by pclass:\n")
log(survival_by_pclass.to_frame("survival_rate").to_markdown())
log("\nSurvival rate by sex + pclass:\n")
log(survival_by_sex_pclass.to_frame("survival_rate").to_markdown())
log(
    f"\n(Boolean-masking spot check: female + 1st class survival rate = "
    f"**{female_p1_survival:.3f}**, matching the groupby table above.)\n"
)
log("![correlation heatmap](charts/correlation_heatmap.png)\n")
log("**Two strongest correlations (by absolute value):**\n")
for _, row in top2.iterrows():
    log(f"- `{row['level_0']}` vs `{row['level_1']}`: **{row['corr']:.3f}**")
log("")

# ---------------------------------------------------------------------------
# Task 5: Multivariate "data story" — >=4 charts, each with interpretation
# ---------------------------------------------------------------------------
# Chart 1: grouped bar — survival rate by pclass x sex
fig, ax = plt.subplots(figsize=(6, 4))
survival_by_sex_pclass.unstack().plot(kind="bar", ax=ax)
ax.set_ylabel("Survival rate")
ax.set_title("Survival rate by class and sex")
fig.tight_layout()
fig.savefig(f"{CHARTS_DIR}/story_1_class_sex_bar.png", dpi=120)
plt.close(fig)

# Chart 2: box — fare by survived
fig, ax = plt.subplots(figsize=(6, 4))
df_clean.boxplot(column="fare", by="survived", ax=ax)
ax.set_title("Fare by survival outcome")
plt.suptitle("")
fig.tight_layout()
fig.savefig(f"{CHARTS_DIR}/story_2_fare_by_survived_box.png", dpi=120)
plt.close(fig)

# Chart 3: scatter — age vs fare, colored by survived
fig, ax = plt.subplots(figsize=(6, 4))
for label, marker, color in [(0, "x", "#C44E52"), (1, "o", "#55A868")]:
    subset = df_clean[df_clean["survived"] == label]
    ax.scatter(subset["age"], subset["fare"], alpha=0.5, marker=marker, color=color,
               label=f"survived={label}")
ax.set_xlabel("age")
ax.set_ylabel("fare")
ax.legend()
ax.set_title("Age vs fare, by survival")
fig.tight_layout()
fig.savefig(f"{CHARTS_DIR}/story_3_age_fare_scatter.png", dpi=120)
plt.close(fig)

# Chart 4: bar — survival rate by family size (sibsp + parch)
df_clean["family_size"] = df_clean["sibsp"] + df_clean["parch"]
survival_by_family = df_clean.groupby("family_size")["survived"].mean()
fig, ax = plt.subplots(figsize=(6, 4))
survival_by_family.plot(kind="bar", ax=ax, color="#4C72B0")
ax.set_ylabel("Survival rate")
ax.set_title("Survival rate by family size (siblings/spouses + parents/children)")
fig.tight_layout()
fig.savefig(f"{CHARTS_DIR}/story_4_family_size_bar.png", dpi=120)
plt.close(fig)

log("## Task 5 — Multivariate data story\n")
log("![class and sex](charts/story_1_class_sex_bar.png)")
log(
    "Class and sex compound rather than substitute for each other: 1st-class women "
    "survived at the highest rate of any group, while 3rd-class men survived at the "
    "lowest — sex alone or class alone understates how much they interact.\n"
)
log("![fare by survived](charts/story_2_fare_by_survived_box.png)")
log(
    "Survivors paid a visibly higher median fare than non-survivors, and the survivor "
    "group has a longer upper tail — consistent with fare acting as a proxy for cabin "
    "location and deck access, which affected evacuation odds.\n"
)
log("![age vs fare](charts/story_3_age_fare_scatter.png)")
log(
    "Survivors (green) are not concentrated at young ages the way a pure 'children "
    "first' story would predict; they're spread across ages but skew toward the "
    "higher end of the fare range — reinforcing that fare/class mattered at least as "
    "much as age.\n"
)
log("![family size](charts/story_4_family_size_bar.png)")
log(
    "Survival peaks at a small-to-medium family size (1-3) rather than at 0: "
    "travelling completely alone or in a very large family group are both associated "
    "with lower survival — plausibly reflecting that solo travellers had no one "
    "helping them find a lifeboat, while very large families struggled to stay "
    "together and evacuate as a unit.\n"
)

# ---------------------------------------------------------------------------
# Task 6: EDA-stage standardization sanity check (age, fare) — NOT used by
# the modeling pipeline, which does its own train-only scaling in Task 8.
# ---------------------------------------------------------------------------
before_stats = df_clean[["age", "fare"]].agg(["mean", "std"])

age_z = (df_clean["age"] - df_clean["age"].mean()) / df_clean["age"].std()
fare_z = (df_clean["fare"] - df_clean["fare"].mean()) / df_clean["fare"].std()
after_stats = pd.DataFrame({"age_z": age_z, "fare_z": fare_z}).agg(["mean", "std"])

log("## Task 6 — Standardization sanity check (EDA-stage only)\n")
log("Before standardization:\n")
log(before_stats.to_markdown())
log("\nAfter z-score standardization (`age_z`, `fare_z`):\n")
log(after_stats.round(6).to_markdown())
log(
    "\nBoth transformed columns have mean ~0 and std ~1, confirming the z-score "
    "formula was applied correctly. This is an EDA-stage check only — the modeling "
    "pipeline in `02_modeling.py` fits its own `StandardScaler` on the training "
    "split alone (Task 8), it does not reuse these columns.\n"
)

# ---------------------------------------------------------------------------
# Write the findings file. 02_modeling.py continues from the same raw
# titanic.csv (saved above, Task 1) via src/cleaning.py's clean_titanic() —
# no second CSV is needed.
# ---------------------------------------------------------------------------
with open(FINDINGS_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(findings))
print(f"Wrote {FINDINGS_PATH}")
