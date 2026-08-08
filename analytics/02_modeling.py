"""
02_modeling.py — /analytics Part B (Tasks 7-15)

Continues from the SAME raw titanic.csv that 01_eda.py saved (Task 1) — does
NOT call sns.load_dataset again. Applies the identical cleaning policy via
src/cleaning.py, then runs the full modeling pipeline.

Saves:
  - charts/decision_tree.png, charts/roc_curves.png, charts/residuals.png
  - modeling_findings.md      (comparison tables + every required conclusion)
  - model_pipeline.joblib     (Task 15: full fitted pipeline, reloadable)

Usage:
    python 02_modeling.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_curve, roc_auc_score, mean_absolute_error,
    mean_squared_error, r2_score,
)
from imblearn.over_sampling import SMOTE
import joblib

from src.cleaning import clean_titanic

CHARTS_DIR = "charts"
FINDINGS_PATH = "modeling_findings.md"
findings = ["# Modeling Findings — /analytics\n"]


def log(md_text, also_print=True):
    findings.append(md_text)
    if also_print:
        print(md_text)


# ---------------------------------------------------------------------------
# Continue from the same committed titanic.csv (Task 1) — NOT a second
# sns.load_dataset call. Apply the identical Task-2 cleaning.
# ---------------------------------------------------------------------------
df_raw = pd.read_csv("titanic.csv")
df_clean, _, _ = clean_titanic(df_raw)

# Feature selection for modeling: the classic 7-feature set (pclass, sex,
# age, sibsp, parch, fare, embarked). Excluded and why:
#   - alive:        identical information to `survived` as a string -> would
#                    leak the label directly into the model. Never used.
#   - deck:          already dropped in cleaning (77% missing).
#   - class, who, adult_male, alone, embark_town:
#                    derived duplicates of pclass/sex/age/sibsp/parch/embarked
#                    already in the feature set; including them adds
#                    redundant/collinear columns without new information.
FEATURES = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
TARGET = "survived"

X = df_clean[FEATURES]
y = df_clean[TARGET]

# ---------------------------------------------------------------------------
# Task 7: stratified train/test split
# ---------------------------------------------------------------------------
class_balance = y.value_counts(normalize=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

log("## Task 7 — Stratified split\n")
log(f"Class balance (`survived`): {dict((int(k), round(v, 3)) for k, v in class_balance.items())}\n")
log(
    "The classes are imbalanced (~38% survived vs ~62% not). A plain random split "
    "risks over- or under-representing the minority (survived) class in the test "
    "fold by chance, which would make evaluation metrics for that class noisy and "
    "unreliable. Stratifying on `survived` forces both train and test to keep the "
    "same ~38/62 ratio as the full dataset, so metrics are measured on a "
    "representative test set.\n"
)
log(f"Train size: {len(X_train)}, test size: {len(X_test)}\n")

# ---------------------------------------------------------------------------
# Task 8: preprocessing — fit on train only, via ColumnTransformer + Pipeline
# ---------------------------------------------------------------------------
numeric_features = ["age", "sibsp", "parch", "fare", "pclass"]
categorical_features = ["sex", "embarked"]

# Note: Task 2 already resolved missingness in these columns (age imputed,
# embarked-missing rows dropped, deck dropped), so on THIS data the imputers
# below are no-ops. They're included anyway: (a) it's what Task 8 explicitly
# asks for, and (b) it's what makes the saved pipeline (Task 15) genuinely
# safe to run on fresh raw data later, which may well contain missing values
# this training data didn't.
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features),
])

log("## Task 8 — Preprocessing pipeline\n")
log(
    "`ColumnTransformer` applies a median `SimpleImputer` + `StandardScaler` to "
    "numeric columns and a most-frequent `SimpleImputer` + `OneHotEncoder` to "
    "`sex`/`embarked`, wrapped so every `.fit(...)` call below fits this "
    "preprocessor **only on `X_train`**; `X_test` only ever sees `.transform(...)`. "
    "This is enforced structurally by always fitting the full sklearn `Pipeline` "
    "(preprocessor + estimator) on `(X_train, y_train)` and only ever calling "
    "`.predict`/`.transform` on `X_test` — never `.fit` or `.fit_transform`.\n"
)

# ---------------------------------------------------------------------------
# Task 9: train three classifiers on the identical split
# ---------------------------------------------------------------------------
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=200),
}

fitted_pipelines = {}
for name, clf in classifiers.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])
    pipe.fit(X_train, y_train)
    fitted_pipelines[name] = pipe

# Decision tree visualization
dt_pipe = fitted_pipelines["Decision Tree"]
feature_names = (
    numeric_features
    + list(dt_pipe.named_steps["preprocessor"]
           .named_transformers_["cat"].named_steps["encoder"]
           .get_feature_names_out(categorical_features))
)
fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(
    dt_pipe.named_steps["classifier"],
    feature_names=feature_names,
    class_names=["did not survive", "survived"],
    filled=True, max_depth=3, fontsize=8, ax=ax,
)
fig.tight_layout()
fig.savefig(f"{CHARTS_DIR}/decision_tree.png", dpi=120)
plt.close(fig)

log("## Task 9 — Three classifiers trained\n")
log("Logistic Regression, Decision Tree (`max_depth=5`), Random Forest "
    "(`n_estimators=200`) — all trained on the identical `(X_train, y_train)`.\n")
log("![decision tree](charts/decision_tree.png) (first 3 levels shown for readability)\n")

# ---------------------------------------------------------------------------
# Task 10: evaluate all three — confusion matrix, accuracy, precision,
# recall, F1, ROC/AUC, side-by-side comparison table
# ---------------------------------------------------------------------------
eval_rows = []
fig, ax = plt.subplots(figsize=(6, 5))
for name, pipe in fitted_pipelines.items():
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

    eval_rows.append({
        "model": name, "accuracy": round(acc, 3), "precision": round(prec, 3),
        "recall": round(rec, 3), "f1": round(f1, 3), "auc": round(auc, 3),
        "confusion_matrix": cm.tolist(),
    })

ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="chance")
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title("ROC curves — all three classifiers")
ax.legend()
fig.tight_layout()
fig.savefig(f"{CHARTS_DIR}/roc_curves.png", dpi=120)
plt.close(fig)

eval_df = pd.DataFrame(eval_rows).set_index("model")

log("## Task 10 — Evaluation (side-by-side)\n")
log(eval_df[["accuracy", "precision", "recall", "f1", "auc"]].to_markdown())
log("\nConfusion matrices (rows = actual [0,1], cols = predicted [0,1]):\n")
for row in eval_rows:
    log(f"- **{row['model']}**: `{row['confusion_matrix']}`")
log("\n![ROC curves](charts/roc_curves.png)\n")

# ---------------------------------------------------------------------------
# Task 11: imbalance-handling comparison (Random Forest, 3 ways)
# ---------------------------------------------------------------------------
X_train_proc = preprocessor.fit_transform(X_train, y_train)  # fit on train only
X_test_proc = preprocessor.transform(X_test)

imbalance_variants = {}

rf_baseline = RandomForestClassifier(random_state=42, n_estimators=200)
rf_baseline.fit(X_train_proc, y_train)
imbalance_variants["baseline"] = rf_baseline

rf_balanced = RandomForestClassifier(random_state=42, n_estimators=200, class_weight="balanced")
rf_balanced.fit(X_train_proc, y_train)
imbalance_variants["class_weight=balanced"] = rf_balanced

smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train_proc, y_train)  # train fold only
rf_smote = RandomForestClassifier(random_state=42, n_estimators=200)
rf_smote.fit(X_train_sm, y_train_sm)
imbalance_variants["SMOTE"] = rf_smote

imbalance_rows = []
for variant_name, model in imbalance_variants.items():
    y_pred = model.predict(X_test_proc)
    imbalance_rows.append({
        "variant": variant_name,
        "precision": round(precision_score(y_test, y_pred), 3),
        "recall": round(recall_score(y_test, y_pred), 3),
        "f1": round(f1_score(y_test, y_pred), 3),
    })
imbalance_df = pd.DataFrame(imbalance_rows).set_index("variant")
best_variant = imbalance_df["f1"].idxmax()

log("## Task 11 — Imbalance handling comparison (Random Forest)\n")
log(f"Class balance: {dict((int(k), round(v, 3)) for k, v in class_balance.items())}\n")
log(imbalance_df.to_markdown())
log(
    f"\n**Conclusion:** `{best_variant}` gives the best F1 on the test set. "
    "SMOTE resamples only `X_train_proc`/`y_train` (the already-preprocessor-"
    "transformed training fold) — the test fold is never touched by SMOTE, "
    "avoiding leakage. In practice `class_weight='balanced'` is often preferred "
    "operationally over SMOTE here since it needs no synthetic data generation "
    "and is cheaper to reproduce, but the table above reports both so the "
    "trade-off is visible rather than assumed.\n"
)

# ---------------------------------------------------------------------------
# Task 12: GridSearchCV tuning (Random Forest), OOB score
# ---------------------------------------------------------------------------
param_grid = {
    "n_estimators": [100, 200, 400],
    "max_depth": [None, 5, 10],
    "max_features": ["sqrt", "log2"],
}
rf_for_tuning = RandomForestClassifier(random_state=42, oob_score=True, bootstrap=True)
grid_search = GridSearchCV(rf_for_tuning, param_grid, cv=5, scoring="f1", n_jobs=-1)
grid_search.fit(X_train_proc, y_train)

best_rf = grid_search.best_estimator_
log("## Task 12 — Hyperparameter tuning (Random Forest)\n")
log(f"Best params: `{grid_search.best_params_}`\n")
log(f"OOB score of the best estimator: **{best_rf.oob_score_:.4f}**\n")

# ---------------------------------------------------------------------------
# Task 13: regression side-task — predict fare
# ---------------------------------------------------------------------------
# Deliberately NOT using `survived` as a regression feature here, even though
# it correlates with fare: fare/class causally influenced survival odds, not
# the other way around, so treating survived as a predictor of fare would be
# modeling the relationship backwards. Kept to demographic/family features.
reg_features = ["pclass", "age", "sibsp", "parch"]
reg_categorical = ["sex", "embarked"]
X_reg = df_clean[reg_features + reg_categorical]
y_reg = df_clean["fare"]

reg_preprocessor = ColumnTransformer([
    ("num", StandardScaler(), reg_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), reg_categorical),
])
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

reg_pipe = Pipeline([("preprocessor", reg_preprocessor), ("regressor", LinearRegression())])
reg_pipe.fit(X_reg_train, y_reg_train)
y_reg_pred = reg_pipe.predict(X_reg_test)

mae = mean_absolute_error(y_reg_test, y_reg_pred)
rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))
r2 = r2_score(y_reg_test, y_reg_pred)
n, p = X_reg_test.shape[0], X_reg_test.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

residuals = y_reg_test - y_reg_pred
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(y_reg_pred, residuals, alpha=0.5)
ax.axhline(0, color="red", linestyle="--")
ax.set_xlabel("Predicted fare")
ax.set_ylabel("Residual")
ax.set_title("Residual plot — fare regression")
fig.tight_layout()
fig.savefig(f"{CHARTS_DIR}/residuals.png", dpi=120)
plt.close(fig)

residual_spread_increasing = np.corrcoef(y_reg_pred, np.abs(residuals))[0, 1] > 0.3
hetero_conclusion = (
    "shows heteroscedasticity — residual spread visibly widens as predicted fare "
    "increases, so the constant-variance assumption of ordinary least squares is "
    "violated"
    if residual_spread_increasing else
    "does not show strong heteroscedasticity — residual spread looks roughly "
    "constant across the range of predicted fare"
)

log("## Task 13 — Regression side-task (predict fare)\n")
log(f"MAE = **{mae:.2f}**, RMSE = **{rmse:.2f}**, R² = **{r2:.3f}**, Adjusted R² = **{adj_r2:.3f}**\n")
log("![residuals](charts/residuals.png)\n")
log(f"The residual plot **{hetero_conclusion}**.\n")

# ---------------------------------------------------------------------------
# Task 14: final model comparison table + recommendation
# ---------------------------------------------------------------------------
log("## Task 14 — Final model comparison\n")
log("**Classification models:**\n")
log(eval_df[["accuracy", "precision", "recall", "f1", "auc"]].to_markdown())
log("\n**Regression model (fare prediction):**\n")
log(pd.DataFrame([{
    "model": "Linear Regression", "MAE": round(mae, 2), "RMSE": round(rmse, 2),
    "R2": round(r2, 3), "Adjusted_R2": round(adj_r2, 3),
}]).set_index("model").to_markdown())

best_clf_name = eval_df["f1"].idxmax()
best_row = eval_df.loc[best_clf_name]
log(
    f"\n**Recommendation:** deploy **{best_clf_name}**. It has the highest F1 "
    f"({best_row['f1']:.3f}) among the three classifiers, balancing precision "
    f"({best_row['precision']:.3f}) and recall ({best_row['recall']:.3f}) rather than "
    f"optimizing one at the other's expense — important here since both false "
    f"positives and false negatives on 'survived' are costly in different ways. "
    f"Its AUC ({best_row['auc']:.3f}) also confirms it separates the two classes "
    f"well across thresholds, not just at the default 0.5 cutoff. Classification "
    f"and regression metrics above are reported as two separate tables/scales — "
    f"they are not directly comparable to each other.\n"
)

# ---------------------------------------------------------------------------
# Task 15: save the best full pipeline (preprocessing + estimator together)
# ---------------------------------------------------------------------------
best_full_pipeline = fitted_pipelines[best_clf_name]  # already preprocessor+estimator, fit on X_train
joblib.dump(best_full_pipeline, "model_pipeline.joblib")

reloaded = joblib.load("model_pipeline.joblib")
sample_raw = X_test.iloc[[0]]  # raw, unpreprocessed row
original_pred = best_full_pipeline.predict(sample_raw)[0]
reloaded_pred = reloaded.predict(sample_raw)[0]
assert original_pred == reloaded_pred, "Reloaded pipeline prediction mismatch!"

log("## Task 15 — Saved pipeline\n")
log(
    f"Saved `{best_clf_name}`'s full fitted pipeline (preprocessor + estimator "
    "together) to `model_pipeline.joblib` via `joblib.dump`. Reloaded it with "
    "`joblib.load` and confirmed it predicts identically on a raw (unpreprocessed) "
    f"test row: original prediction = `{original_pred}`, reloaded prediction = "
    f"`{reloaded_pred}` — **match confirmed**.\n"
)

with open(FINDINGS_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(findings))
print(f"\nWrote {FINDINGS_PATH}")
print("Saved model_pipeline.joblib")
