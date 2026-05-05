"""

HEART DISEASE RISK PREDICTION SYSTEM
Pipeline Overview:
  1. Load Data         – Read 3 raw CSV datasets
  2. Data Integration  – Map each dataset to a common schema
  3. Combine & Clean   – Merge, deduplicate, impute, remove outliers
  4. Encoding          – One-hot encode categoricals, cast binaries
  5. Standardisation   – Z-score scale continuous features
  6. Feature Selection – (optional SelectKBest block left for reference)
  7. Save              – Write final_unified_dataset.csv

  8. Star Schema DW    – Build fact + dimension tables in SQLite
  9. OLAP Queries      – Simple multidimensional roll-up examples

  10. Classification   – Logistic Regression, Decision Tree, Random Forest
  11. Regression       – Linear Regression on cholesterol
  12. Clustering       – K-Means (Low / Medium / High risk)
  13. Association Rules – Apriori via mlxtend

"""


# SECTION 0 – IMPORTS
import warnings
warnings.filterwarnings("ignore")         

import sqlite3                             # Built-in Python SQL engine
import pandas as pd
import numpy as np
import joblib                              # For persisting the scaler (FIX)
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn – preprocessing
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Scikit-learn – outlier detection
from sklearn.ensemble import IsolationForest

# Scikit-learn – feature selection
from sklearn.feature_selection import SelectKBest, f_classif

# Scikit-learn – model selection
from sklearn.model_selection import train_test_split, cross_val_score

# Scikit-learn – classification models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Scikit-learn – regression models
from sklearn.linear_model import LinearRegression

# Scikit-learn – clustering
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA  # for 2D cluster visualisation

# Scikit-learn – evaluation metrics
from sklearn.metrics import(
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
    mean_squared_error, r2_score
)

# mlxtend – association rule mining (Apriori)
# Install if missing:  pip install mlxtend
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder


# SECTION 1 – LOAD DATA
print("=" * 60)
print("SECTION 1 – LOADING RAW DATA")
print("=" * 60)

# Read each of the three source CSV files.
uci    = pd.read_csv("uci_heart.csv")
fr     = pd.read_csv("framingham.csv")
cardio = pd.read_csv("cardio.csv", sep=";")

print(f"UCI rows    : {len(uci)}")
print(f"Framingham  : {len(fr)}")
print(f"Cardio      : {len(cardio)}")


# SECTION 2 – DATA INTEGRATION (ETL – TRANSFORM PHASE)
print("\n" + "=" * 60)
print("SECTION 2 – DATA INTEGRATION")
print("=" * 60)

# ---------------------------------------------------------------------------
# 2A. CARDIO DATASET
# ---------------------------------------------------------------------------
# age: originally stored in DAYS → convert to years by dividing by 365
cardio["age"] = cardio["age"] / 365

# gender: original encoding  1 = female, 2 = male
#          canonical encoding 0 = female, 1 = male  (matches UCI & Framingham)
cardio["gender"] = cardio["gender"].map({1: 0, 2: 1})

# BMI: the dataset provides height (cm) and weight (kg) but not BMI directly.
#      Formula: BMI = weight / (height_in_metres)²
cardio["BMI"] = cardio["weight"] / ((cardio["height"] / 100) ** 2)

# cholesterol: original encoding is ordinal (1=normal, 2=above, 3=well above)
#              We map each level to a representative mg/dL value so the column
#              is comparable across datasets (all three will be in mg/dL).
cardio["cholesterol"] = cardio["cholesterol"].map({1: 180, 2: 220, 3: 260})

# gluc (glucose): same ordinal-to-numeric mapping as cholesterol
cardio["gluc"] = cardio["gluc"].map({1: 85, 2: 110, 3: 140})

# Build cardio_new with the canonical 24-column layout.
# Columns that the cardio dataset does NOT contain are set to np.nan (missing).
# Explicit NaN is BETTER than implicit because it makes missingness visible and
# consistent from the very beginning of the pipeline.
cardio_new = pd.DataFrame({
    "age":         cardio["age"],
    "gender":      cardio["gender"],
    "BMI":         cardio["BMI"],
    "sysBP":       cardio["ap_hi"],       # systolic blood pressure
    "diaBP":       cardio["ap_lo"],       # diastolic blood pressure
    "cholesterol": cardio["cholesterol"],
    "glucose":     cardio["gluc"],
    "smoke":       cardio["smoke"],
    "alcohol":     cardio["alco"],
    "active":      cardio["active"],
    # The following clinical columns are absent in cardio → NaN
    "BPMeds":      np.nan,
    "stroke":      np.nan,
    "hypertension": np.nan,
    "diabetes":    np.nan,
    "heartRate":   np.nan,
    "cp":          np.nan,    # chest pain type
    "restecg":     np.nan,    # resting ECG results
    "thalach":     np.nan,    # maximum heart rate achieved
    "exang":       np.nan,    # exercise-induced angina
    "oldpeak":     np.nan,    # ST depression induced by exercise
    "slope":       np.nan,    # slope of peak exercise ST segment
    "ca":          np.nan,    # number of major vessels coloured by fluoroscopy
    "thal":        np.nan,    # thalassemia type
    "target":      cardio["cardio"],   # 1 = has cardiovascular disease
})

# ---------------------------------------------------------------------------
# 2B. FRAMINGHAM DATASET
# ---------------------------------------------------------------------------
# BMI, sysBP, diaBP, cholesterol, glucose, heartRate already in correct units.
# 'male' column (1=male, 0=female) already matches our canonical gender coding.
# 'TenYearCHD' = 1 if the patient developed coronary heart disease in 10 years.
fr_new = pd.DataFrame({
    "age":         fr["age"],
    "gender":      fr["male"],
    "BMI":         fr["BMI"],
    "sysBP":       fr["sysBP"],
    "diaBP":       fr["diaBP"],
    "cholesterol": fr["totChol"],
    "glucose":     fr["glucose"],
    "smoke":       fr["currentSmoker"],
    "alcohol":     np.nan,              # not collected in Framingham
    "active":      np.nan,              # not collected in Framingham
    "BPMeds":      fr["BPMeds"],
    "stroke":      fr["prevalentStroke"],
    "hypertension": fr["prevalentHyp"],
    "diabetes":    fr["diabetes"],
    "heartRate":   fr["heartRate"],
    # UCI-specific clinical columns absent in Framingham → NaN
    "cp":          np.nan,
    "restecg":     np.nan,
    "thalach":     np.nan,
    "exang":       np.nan,
    "oldpeak":     np.nan,
    "slope":       np.nan,
    "ca":          np.nan,
    "thal":        np.nan,
    "target":      fr["TenYearCHD"],
})

# ---------------------------------------------------------------------------
# 2C. UCI HEART DISEASE DATASET
# ---------------------------------------------------------------------------
# fbs (fasting blood sugar): originally a binary flag (0 = ≤120 mg/dL, 1 = >120)
# We convert to a representative numeric glucose value for consistency.
uci["fbs"] = uci["fbs"].map({0: 90, 1: 130})

uci_new = pd.DataFrame({
    "age":         uci["age"],
    "gender":      uci["sex"],        # 1=male, 0=female  (already canonical)
    "BMI":         np.nan,            # not in UCI
    "sysBP":       uci["trestbps"],   # resting blood pressure (mm Hg)
    "diaBP":       np.nan,            # not in UCI
    "cholesterol": uci["chol"],       # serum cholesterol (mg/dL)
    "glucose":     uci["fbs"],        # fasting blood sugar (converted above)
    "smoke":       np.nan,
    "alcohol":     np.nan,
    "active":      np.nan,
    "BPMeds":      np.nan,
    "stroke":      np.nan,
    "hypertension": np.nan,
    "diabetes":    np.nan,
    "heartRate":   np.nan,
    "cp":          uci["cp"],         # chest pain type (0–3)
    "restecg":     uci["restecg"],    # resting ECG (0–2)
    "thalach":     uci["thalach"],    # max heart rate achieved
    "exang":       uci["exang"],      # exercise-induced angina (0/1)
    "oldpeak":     uci["oldpeak"],    # ST depression
    "slope":       uci["slope"],      # slope of ST segment
    "ca":          uci["ca"],         # number of major vessels (0–3)
    "thal":        uci["thal"],       # thalassemia (1=normal,2=fixed,3=reversible)
    "target":      uci["target"],     # 1 = heart disease present
})


# SECTION 3 – COMBINE & CLEAN
print("\n" + "=" * 60)
print("SECTION 3 – COMBINE, DEDUPLICATE, IMPUTE, OUTLIER REMOVAL")
print("=" * 60)

# --- 3A. Concatenate all three harmonised sub-frames ---
# ignore_index=True resets the row index to a continuous 0, 1, 2 … sequence
# so there are no duplicated index values from the original datasets.
df = pd.concat([cardio_new, fr_new, uci_new], ignore_index=True)
print(f"Combined shape (before cleaning): {df.shape}")

# --- 3B. Drop fully duplicate rows ---
# A duplicate row carries no new information and can bias model training.
before = len(df)
df.drop_duplicates(inplace=True)
print(f"Rows removed (duplicates): {before - len(df)}")

# --- 3C. Missing Value Imputation ---
# Strategy:
#   • Continuous / numeric columns  →  fill with COLUMN MEAN
#     (mean imputation preserves the overall distribution centre)
#   • Categorical / binary columns  →  fill with COLUMN MODE
#     (most-frequent value keeps the dominant category intact)

num_cols = ["age", "BMI", "sysBP", "diaBP", "heartRate",
            "thalach", "oldpeak", "cholesterol", "glucose"]

cat_cols = ["gender", "smoke", "alcohol", "active",
            "BPMeds", "stroke", "hypertension", "diabetes",
            "exang", "cp", "restecg", "slope", "thal", "ca"]

# Fill numeric columns with their respective column means
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

# Fill each categorical column independently with its mode (most common value).
# mode() returns a Series; [0] picks the first/only mode value.
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print(f"Missing values after imputation: {df.isnull().sum().sum()}")

# --- 3D. Outlier Removal with Isolation Forest ---

## contamination=0.05 tells the algorithm to flag the most-anomalous 5 %
# of rows as outliers.  random_state=42 ensures reproducibility.
iso = IsolationForest(contamination=0.05, random_state=42)

# fit_predict returns +1 for inliers and -1 for outliers.
outlier_labels = iso.fit_predict(df[num_cols])

before = len(df)
df = df[outlier_labels == 1].copy()   # keep only inliers
print(f"Rows removed (outliers): {before - len(df)}")
print(f"Shape after outlier removal: {df.shape}")


# SECTION 4 – ENCODING
print("\n" + "=" * 60)
print("SECTION 4 – ENCODING")
print("=" * 60)

# --- 4A. One-Hot Encoding for nominal multi-class columns ---
# These columns have 3+ categories with NO natural ordering, so we
# create a binary dummy column for each category.
# drop_first=True drops the first dummy to avoid multicollinearity
# (a.k.a. the "dummy variable trap").
ohe_cols = ["cp", "restecg", "slope", "thal"]

# Cast to int first so get_dummies treats them as integers, not floats.
df[ohe_cols] = df[ohe_cols].astype(int)
df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)

# get_dummies creates bool columns in newer pandas → convert back to int
# (0 / 1) so all downstream numeric operations work uniformly.
bool_cols = [col for col in df.columns if df[col].dtype == bool]
df[bool_cols] = df[bool_cols].astype(int)

# --- 4B. Binary columns – just cast to int ---
# These are already 0/1 values; we just ensure the dtype is integer.
binary_cols = ["gender", "smoke", "alcohol", "active",
               "BPMeds", "stroke", "hypertension", "diabetes", "exang", "ca"]
df[binary_cols] = df[binary_cols].astype(int)

print(f"Columns after encoding: {df.shape[1]}")
print(f"Column list: {list(df.columns)}")


# SECTION 5 – STANDARDISATION
print("\n" + "=" * 60)
print("SECTION 5 – STANDARDISATION")
print("=" * 60)

# Separate target before scaling so we don't accidentally scale it.
target = df["target"].copy()

# Continuous features to standardise.
# One-hot encoded dummy columns are already 0/1 so they don't need scaling.
num_cols = ["age", "BMI", "sysBP", "diaBP",
            "cholesterol", "heartRate", "thalach", "oldpeak", "glucose"]

# StandardScaler transforms each column to zero mean and unit variance:
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# FIX: Persist the scaler so predict.py can load it and apply the same
# mean/std to raw user input.  Without this, predict.py re-fits a new
# StandardScaler on the already-z-scored CSV, which has mean≈0 and std≈1,
# so transform(raw_value) ≈ raw_value — passing wildly out-of-range values
# to the model and producing the same constant prediction for every patient.
joblib.dump(scaler, "scaler.pkl")
print("✓ Scaler saved: scaler.pkl  (used by predict.py)")

# Rebuild the scaled DataFrame and reattach the unscaled target column.
df_scaled = df.copy()
df_scaled["target"] = target.values   # .values strips the old index

print("Standardisation complete.")
print(df_scaled[num_cols].describe().round(2))


# ===========================================================================
# SECTION 6 – FEATURE SELECTION (optional-currently using all feature)
# ===========================================================================
# SelectKBest with f_classif scores each feature using the ANOVA F-statistic
# between the feature and the target variable.  The top k features are kept.
# This is left commented out so you can experiment.
#
# To activate: uncomment the block below and comment out the final_df = line.
# --------------------------------------------------------------------------
# selector = SelectKBest(score_func=f_classif, k=18)
# X_selected = selector.fit_transform(
#     df_scaled.drop("target", axis=1),
#     df_scaled["target"]
# )
# selected_cols = df_scaled.drop("target", axis=1).columns[selector.get_support()]
# final_df = pd.DataFrame(X_selected, columns=selected_cols)
# final_df["target"] = df_scaled["target"].values
# ----------------------------------------------------------------------------

# Use all features for now
final_df = df_scaled.copy()



# SECTION 7 – SAVE FINAL DATASET
print("\n" + "=" * 60)
print("SECTION 7 – SAVING FINAL DATASET")
print("=" * 60)

final_df.to_csv("final_unified_dataset.csv", index=False)
print("✓ Saved: final_unified_dataset.csv")
print(f"  Shape : {final_df.shape}")
print(final_df.head(3))


# SECTION 8 – STAR SCHEMA DATA WAREHOUSE (SQLite)
# The project report specifies a Star Schema with:
#   • Fact Table   : heart_disease_fact
#   • Dimension Tables: dim_patient, dim_clinical, dim_lifestyle
#
# We implement this using Python's built-in sqlite3 module so no external
# database server is needed.
print("\n" + "=" * 60)
print("SECTION 8 – STAR SCHEMA DATA WAREHOUSE")
print("=" * 60)

# Connect to (or create) a SQLite database file.
conn = sqlite3.connect("heart_disease_dw.db")
cursor = conn.cursor()

# Drop tables if they exist so re-running the script doesn't cause errors.
cursor.execute("DROP TABLE IF EXISTS heart_disease_fact")
cursor.execute("DROP TABLE IF EXISTS dim_patient")
cursor.execute("DROP TABLE IF EXISTS dim_clinical")
cursor.execute("DROP TABLE IF EXISTS dim_lifestyle")

# ---- DIMENSION TABLE: dim_patient ----------------------------------------
# Stores demographic attributes that describe WHO the patient is.
cursor.execute("""
CREATE TABLE dim_patient (
    patient_id  INTEGER PRIMARY KEY,  -- Surrogate key (auto-generated)
    age         REAL,                 -- Patient age in years
    gender      INTEGER,              -- 0 = Female, 1 = Male
    BMI         REAL,                 -- Body Mass Index (kg/m²)
    hypertension INTEGER,             -- 1 = has hypertension, 0 = no
    diabetes    INTEGER,              -- 1 = diabetic, 0 = not
    stroke      INTEGER               -- 1 = prior stroke, 0 = no
)
""")

# ---- DIMENSION TABLE: dim_clinical ----------------------------------------
# Stores clinical measurement attributes (lab results, ECG, etc.)
cursor.execute("""
CREATE TABLE dim_clinical (
    clinical_id  INTEGER PRIMARY KEY, -- Surrogate key
    sysBP        REAL,                -- Systolic blood pressure (mm Hg)
    diaBP        REAL,                -- Diastolic blood pressure (mm Hg)
    cholesterol  REAL,                -- Total cholesterol (mg/dL)
    glucose      REAL,                -- Blood glucose (mg/dL)
    heartRate    REAL,                -- Resting heart rate (bpm)
    BPMeds       INTEGER              -- 1 = on BP medication, 0 = not
)
""")

# ---- DIMENSION TABLE: dim_lifestyle ----------------------------------------
# Stores lifestyle / behavioural attributes.
cursor.execute("""
CREATE TABLE dim_lifestyle (
    lifestyle_id INTEGER PRIMARY KEY, -- Surrogate key
    smoke        INTEGER,             -- 1 = smoker, 0 = non-smoker
    alcohol      INTEGER,             -- 1 = drinks alcohol, 0 = does not
    active       INTEGER              -- 1 = physically active, 0 = sedentary
)
""")

# ---- FACT TABLE: heart_disease_fact ----------------------------------------
# The fact table references each dimension via its surrogate key and stores
# the outcome measure (heart disease target) and a derived risk score.
cursor.execute("""
CREATE TABLE heart_disease_fact (
    fact_id      INTEGER PRIMARY KEY,
    patient_id   INTEGER REFERENCES dim_patient(patient_id),
    clinical_id  INTEGER REFERENCES dim_clinical(clinical_id),
    lifestyle_id INTEGER REFERENCES dim_lifestyle(lifestyle_id),
    target       INTEGER,   -- 1 = heart disease present / high risk
    risk_score   REAL       -- Derived: proportion of risk indicators active
)
""")

# ---- Populate dimension tables with rows from final_df --------------------
# We use the row index as the surrogate key (1-based for clarity).
# Using executemany() is faster than a Python-level loop.

patient_rows = []
clinical_rows = []
lifestyle_rows = []
fact_rows = []

# Identify which OHE columns exist (they vary based on data values present).
# We use the original unscaled df for the DW insert so values are interpretable.
# We load the raw combined frame again from final_df, but note that num_cols
# have been z-scored.  For the DW we'll insert the scaled values as-is.

for i, row in final_df.reset_index(drop=True).iterrows():
    pid  = i + 1   # 1-based surrogate key
    cid  = i + 1
    lid  = i + 1

    # -dim_patient row 
    patient_rows.append((
        pid,
        row.get("age", None),
        int(row.get("gender", 0)),
        row.get("BMI", None),
        int(row.get("hypertension", 0)),
        int(row.get("diabetes", 0)),
        int(row.get("stroke", 0)),
    ))

    # -dim_clinical row 
    clinical_rows.append((
        cid,
        row.get("sysBP", None),
        row.get("diaBP", None),
        row.get("cholesterol", None),
        row.get("glucose", None),
        row.get("heartRate", None),
        int(row.get("BPMeds", 0)),
    ))

    # -dim_lifestyle row
    lifestyle_rows.append((
        lid,
        int(row.get("smoke", 0)),
        int(row.get("alcohol", 0)),
        int(row.get("active", 0)),
    ))

    # --- Derived risk score ---
    # Simple heuristic: count how many binary risk factors are present
    # (smoke, hypertension, diabetes, stroke, BPMeds) divided by 5.
    # This gives a 0–1 proportional risk score independent of the classifier.
    risk_factors = ["smoke", "hypertension", "diabetes", "stroke", "BPMeds"]
    risk_score = sum(int(row.get(c, 0)) for c in risk_factors) / len(risk_factors)

    fact_rows.append((i + 1, pid, cid, lid, int(row["target"]), risk_score))

cursor.executemany(
    "INSERT INTO dim_patient VALUES (?,?,?,?,?,?,?)", patient_rows)
cursor.executemany(
    "INSERT INTO dim_clinical VALUES (?,?,?,?,?,?,?)", clinical_rows)
cursor.executemany(
    "INSERT INTO dim_lifestyle VALUES (?,?,?,?)", lifestyle_rows)
cursor.executemany(
    "INSERT INTO heart_disease_fact VALUES (?,?,?,?,?,?)", fact_rows)

conn.commit()
print("✓ Star schema created and populated in heart_disease_dw.db")

# Verify row counts
for tbl in ["dim_patient", "dim_clinical", "dim_lifestyle", "heart_disease_fact"]:
    count = cursor.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
    print(f"  {tbl}: {count} rows")


# SECTION 9 – OLAP QUERIES (Multidimensional Analysis)
# OLAP (Online Analytical Processing) lets us slice and roll-up the fact
# table across multiple dimensions to discover aggregate patterns.
print("\n" + "=" * 60)
print("SECTION 9 – OLAP / MULTIDIMENSIONAL QUERIES")
print("=" * 60)

# --- OLAP Query 1: Average risk score by gender ---
# "Roll up" across all patients and group by gender dimension.
q1 = pd.read_sql_query("""
    SELECT
        p.gender,
        AVG(f.risk_score) AS avg_risk_score,
        COUNT(*)          AS patient_count,
        SUM(f.target)     AS disease_count
    FROM heart_disease_fact f
    JOIN dim_patient p ON f.patient_id = p.patient_id
    GROUP BY p.gender
""", conn)
print("\nOLAP Q1 – Average risk score by gender (0=Female, 1=Male):")
print(q1.to_string(index=False))

# --- OLAP Query 2: Disease prevalence by smoking status ---
q2 = pd.read_sql_query("""
    SELECT
        l.smoke,
        AVG(f.target)     AS disease_rate,
        COUNT(*)          AS patient_count
    FROM heart_disease_fact f
    JOIN dim_lifestyle l ON f.lifestyle_id = l.lifestyle_id
    GROUP BY l.smoke
""", conn)
print("\nOLAP Q2 – Disease rate by smoking status (0=Non-smoker, 1=Smoker):")
print(q2.to_string(index=False))

# --- OLAP Query 3: Disease rate for patients on BP medication ---
q3 = pd.read_sql_query("""
    SELECT
        c.BPMeds,
        AVG(f.target)     AS disease_rate,
        COUNT(*)          AS patient_count
    FROM heart_disease_fact f
    JOIN dim_clinical c ON f.clinical_id = c.clinical_id
    GROUP BY c.BPMeds
""", conn)
print("\nOLAP Q3 – Disease rate by BP medication (0=No, 1=Yes):")
print(q3.to_string(index=False))

conn.close()   # Done with the DW for now


# ===========================================================================
# SECTION 10 – TRAIN / TEST SPLIT
# We split ONCE and reuse the same split for all models so results are
# directly comparable (same test set seen by every model).
# ===========================================================================
print("\n" + "=" * 60)
print("SECTION 10 – TRAIN / TEST SPLIT")
print("=" * 60)

X = final_df.drop("target", axis=1)   # Features (everything except label)
y = final_df["target"]                 # Labels

# 80 % training, 20 % test.
# stratify=y ensures the class ratio (disease vs no-disease) is the same in
# both splits, which is important when one class is more common than the other.
# random_state=42 makes the split reproducible.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set : {X_train.shape[0]} rows")
print(f"Test set     : {X_test.shape[0]} rows")
print(f"Class distribution (train): {y_train.value_counts().to_dict()}")


# ===========================================================================
# SECTION 11 – CLASSIFICATION
# Three algorithms are trained and evaluated.
# Metrics used: Accuracy, Precision, Recall, F1-Score
# ===========================================================================
print("\n" + "=" * 60)
print("SECTION 11 – CLASSIFICATION")
print("=" * 60)

# Helper function to evaluate any classifier and print a summary.
def evaluate_classifier(name, model, X_tr, y_tr, X_te, y_te):
    """
    Train `model` on training data, predict on test data, and print
    accuracy, precision, recall, and F1-score.

    Parameters
    ----------
    name  : str  – label for printing
    model : sklearn estimator
    X_tr, y_tr : training features and labels
    X_te, y_te : test features and labels

    Returns
    -------
    dict with metric scores
    """
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, zero_division=0)
    rec  = recall_score(y_te, y_pred, zero_division=0)
    f1   = f1_score(y_te, y_pred, zero_division=0)

    print(f"\n── {name} ──")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}  (of all predicted positives, how many are true positives)")
    print(f"  Recall    : {rec:.4f}  (of all actual positives, how many did we catch)")
    print(f"  F1-Score  : {f1:.4f}  (harmonic mean of precision and recall)")
    print(classification_report(y_te, y_pred, zero_division=0))

    return {"model": name, "accuracy": acc, "precision": prec,
            "recall": rec, "f1": f1, "fitted_model": model}


# --- 11A. Logistic Regression ---
# A linear model that estimates the probability of the positive class.
# max_iter=1000 prevents convergence warnings on larger datasets.
lr_result = evaluate_classifier(
    "Logistic Regression",
    LogisticRegression(max_iter=1000, random_state=42),
    X_train, y_train, X_test, y_test
)

# --- 11B. Decision Tree ---
# A non-linear model that splits data on feature thresholds recursively.
# max_depth=5 prevents overfitting by limiting tree depth.
dt_result = evaluate_classifier(
    "Decision Tree",
    DecisionTreeClassifier(max_depth=5, random_state=42),
    X_train, y_train, X_test, y_test
)

# --- 11C. Random Forest ---
# An ensemble of many decision trees; each tree is trained on a random
# bootstrap sample of the data and a random subset of features.
# This reduces variance (overfitting) compared to a single tree.
# n_estimators=100 means 100 trees in the forest.
rf_result = evaluate_classifier(
    "Random Forest",
    RandomForestClassifier(n_estimators=100, random_state=42),
    X_train, y_train, X_test, y_test
)

# --- Summary table ---
clf_summary = pd.DataFrame([
    {k: v for k, v in r.items() if k != "fitted_model"}
    for r in [lr_result, dt_result, rf_result]
])
print("\nClassification Summary:")
print(clf_summary.to_string(index=False))

# --- Feature importance from Random Forest ---
# Random Forest can rank features by how much each one reduces impurity
# across all trees.  Higher = more informative.
rf_model = rf_result["fitted_model"]
feat_imp = pd.Series(rf_model.feature_importances_, index=X.columns)
feat_imp = feat_imp.sort_values(ascending=False).head(10)
print("\nTop-10 Feature Importances (Random Forest):")
print(feat_imp.round(4))


# SECTION 12 – REGRESSION
# We predict the continuous 'cholesterol' column (mg/dL) using all other
# features as it is one of the few truly continuous, clinically meaningful values.
# Note: the cholesterol column in df_scaled is z-scored, so
# predicted values will also be in standardised units.
# Metrics: RMSE (Root Mean Squared Error) and R² (coefficient of determination)
print("\n" + "=" * 60)
print("SECTION 12 – REGRESSION (Predicting Cholesterol)")
print("=" * 60)

# Use final_df but swap the response variable to cholesterol.
reg_df = final_df.dropna(subset=["cholesterol"]).copy()

X_reg = reg_df.drop(columns=["cholesterol", "target"])
y_reg = reg_df["cholesterol"]

X_reg_tr, X_reg_te, y_reg_tr, y_reg_te = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# - 12A. Simple Linear Regression on single feature (sysBP)
# This is for interpretability: higher systolic BP is expected to correlate
# with higher cholesterol.
lr_simple = LinearRegression()
lr_simple.fit(X_reg_tr[["sysBP"]], y_reg_tr)
y_pred_simple = lr_simple.predict(X_reg_te[["sysBP"]])

rmse_simple = np.sqrt(mean_squared_error(y_reg_te, y_pred_simple))
r2_simple   = r2_score(y_reg_te, y_pred_simple)
print(f"\nSimple Linear Regression (sysBP → cholesterol):")
print(f"  RMSE : {rmse_simple:.4f}  (lower is better)")
print(f"  R²   : {r2_simple:.4f}   (1.0 = perfect, 0.0 = baseline mean)")

# - 12B. Multiple Linear Regression (all features) 
# Using all features gives the model more signals to work with.
lr_multi = LinearRegression()
lr_multi.fit(X_reg_tr, y_reg_tr)
y_pred_multi = lr_multi.predict(X_reg_te)

rmse_multi = np.sqrt(mean_squared_error(y_reg_te, y_pred_multi))
r2_multi   = r2_score(y_reg_te, y_pred_multi)
print(f"\nMultiple Linear Regression (all features → cholesterol):")
print(f"  RMSE : {rmse_multi:.4f}")
print(f"  R²   : {r2_multi:.4f}")

# Top 5 regression coefficients (most influential features)
coef_series = pd.Series(lr_multi.coef_, index=X_reg_tr.columns)
print("\nTop 5 positive predictors of cholesterol:")
print(coef_series.nlargest(5).round(4))
print("Top 5 negative predictors of cholesterol:")
print(coef_series.nsmallest(5).round(4))


# SECTION 13 – CLUSTERING (K-Means)
# K-Means partitions patients into k groups by minimising the within-cluster
# sum of squared distances to the cluster centroid.
# We use 3 clusters: Low / Medium / High risk.
print("\n" + "=" * 60)
print("SECTION 13 – CLUSTERING (K-Means, k=3)")
print("=" * 60)

# We cluster on the continuous (already standardised) features only.
cluster_features = num_cols   # the z-scored list from Section 5

# --- Find optimal k using the Elbow Method ---
# WCSS (Within-Cluster Sum of Squares): measures how tight the clusters are.
# The "elbow" point — where reducing k further gives diminishing returns —
# suggests the best k.
wcss = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(final_df[cluster_features])
    wcss.append(km.inertia_)   # inertia_ = WCSS

# Fit the final model with k=3 as specified in the project report.
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
final_df["cluster"] = kmeans.fit_predict(final_df[cluster_features])

# --- Interpret clusters: map each cluster ID to a risk label ---
# Calculate the mean target (disease rate) per cluster and rank them:
#   highest disease rate → High Risk
#   lowest  disease rate → Low Risk
cluster_profile = final_df.groupby("cluster")["target"].mean().sort_values()
risk_map = {
    cluster_profile.index[0]: "Low Risk",
    cluster_profile.index[1]: "Medium Risk",
    cluster_profile.index[2]: "High Risk",
}
final_df["risk_label"] = final_df["cluster"].map(risk_map)

print("\nCluster sizes and disease rates:")
print(final_df.groupby("risk_label")["target"].agg(["count", "mean"]).round(3))
print("\nCluster centroids (standardised scale, selected features):")
print(pd.DataFrame(kmeans.cluster_centers_, columns=cluster_features).round(2))

# SECTION 13B – K-MEANS CLUSTER VISUALISATIONS
# Three graphs that visually show what the clustering found:
#   Graph 1 – PCA Scatter   : patients plotted in 2D, coloured by cluster
#   Graph 2 – Disease Rate   : how many in each cluster actually have disease
#   Graph 3 – Centroid Heatmap: the average feature value for each cluster
print("\n" + "=" * 60)
print("SECTION 13B – K-MEANS CLUSTER VISUALISATIONS")
print("=" * 60)

# Colour and label for each cluster (Low / Medium / High risk)
cluster_colors = {"Low Risk": "#4CAF50", "Medium Risk": "#FF9800", "High Risk": "#F44336"}

# ── GRAPH 1: PCA Scatter Plot 
# K-Means clusters in high-dimensional space are hard to see directly.
# PCA (Principal Component Analysis) compresses all features into 2 numbers
# (PC1 and PC2) that capture the most variance, so we can plot each patient
# as a single dot and colour it by its cluster assignment.

pca = PCA(n_components=2, random_state=42)
pca_coords = pca.fit_transform(final_df[cluster_features])   # 2-column array

plt.figure(figsize=(8, 6))
for label in ["Low Risk", "Medium Risk", "High Risk"]:
    # Boolean mask to select only the rows belonging to this cluster label
    mask = final_df["risk_label"] == label
    plt.scatter(
        pca_coords[mask, 0],          # x = first principal component
        pca_coords[mask, 1],          # y = second principal component
        label=label,
        color=cluster_colors[label],
        alpha=0.5,                    # semi-transparent so overlaps are visible
        s=20                          # dot size
    )

plt.title("K-Means Clusters — PCA 2D View\n(each dot = one patient)")
plt.xlabel(f"Principal Component 1  ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
plt.ylabel(f"Principal Component 2  ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
plt.legend(title="Risk Cluster")
plt.tight_layout()
plt.savefig("graph1_pca_scatter.png", dpi=150)
plt.show()
print("Graph 1 saved: graph1_pca_scatter.png")

# ── GRAPH 2: Disease Prevalence per Cluster 
# Each bar shows the percentage of patients IN that cluster who actually have
# heart disease (target=1).  A good clustering should show Low Risk having a
# low disease rate and High Risk having a high disease rate.

disease_rate = (
    final_df.groupby("risk_label")["target"]
    .mean() * 100                        # convert to percentage
).reindex(["Low Risk", "Medium Risk", "High Risk"])   # fix bar order

plt.figure(figsize=(7, 5))
bars = plt.bar(
    disease_rate.index,
    disease_rate.values,
    color=[cluster_colors[l] for l in disease_rate.index],
    edgecolor="white",
    width=0.5
)

# Add percentage label on top of each bar
for bar, val in zip(bars, disease_rate.values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,   # x centre of bar
        bar.get_height() + 0.8,               # just above the bar
        f"{val:.1f}%",
        ha="center", va="bottom", fontsize=11, fontweight="bold"
    )

plt.title("Heart Disease Rate per Cluster\n(% of patients with disease in each group)")
plt.xlabel("Risk Cluster")
plt.ylabel("Disease Prevalence (%)")
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig("graph2_disease_prevalence.png", dpi=150)
plt.show()
print("Graph 2 saved: graph2_disease_prevalence.png")

# ── GRAPH 3: Centroid Heatmap 
# The centroid is the "average patient" for each cluster.
# This heatmap shows each feature's standardised centroid value, making it
# easy to see which features are high or low for each risk group.
# Blue = low/negative value,  Orange/Red = high/positive value.

centroid_df = pd.DataFrame(
    kmeans.cluster_centers_,         # shape: (3, n_features)
    columns=cluster_features
)
# Map cluster IDs (0, 1, 2) to their risk labels so rows are readable
centroid_df.index = [risk_map[i] for i in range(3)]
centroid_df = centroid_df.reindex(["Low Risk", "Medium Risk", "High Risk"])

plt.figure(figsize=(12, 4))
sns.heatmap(
    centroid_df,
    annot=True,           # show the number inside each cell
    fmt=".2f",            # 2 decimal places
    cmap="RdYlGn_r",      # red = high risk values, green = low risk values
    linewidths=0.5,       # thin grid lines between cells
    cbar_kws={"label": "Standardised value (z-score)"}
)
plt.title("Cluster Centroid Values\n(standardised — positive = above average, negative = below average)")
plt.xlabel("Clinical Feature")
plt.ylabel("Risk Cluster")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("graph3_centroid_heatmap.png", dpi=150)
plt.show()
print("Graph 3 saved: graph3_centroid_heatmap.png")

print("\nAll 3 cluster graphs generated successfully.")


# SECTION 14 – ASSOCIATION RULE MINING (Apriori)
# Association rule mining finds patterns like:
#   "If a patient smokes AND has hypertension, they are likely to have diabetes"
# Metrics:
#   Support    – fraction of records where the rule's entire itemset appears
#   Confidence – P(consequent | antecedent)
#   Lift       – how much more likely the consequent is given the antecedent
#                vs. the base rate (lift > 1 means positive association)
print("\n" + "=" * 60)
print("SECTION 14 – ASSOCIATION RULE MINING (Apriori)")
print("=" * 60)

# Apriori works on a binary transaction matrix.
# We use the binary lifestyle/clinical indicator columns.
arm_cols = ["smoke", "alcohol", "active", "hypertension",
            "diabetes", "stroke", "BPMeds", "exang", "target"]

# Keep only the ARM columns, drop any rows with NaN, cast to bool
arm_df = final_df[arm_cols].dropna().astype(bool)

# The mlxtend apriori function needs a boolean DataFrame where
# True = item is present in that transaction (row).
frequent_itemsets = apriori(
    arm_df,
    min_support=0.1,       # itemset must appear in at least 10 % of rows
    use_colnames=True      # use column names instead of column indices
)

if frequent_itemsets.empty:
    print("No frequent itemsets found. Try lowering min_support.")
else:
    # Generate rules from frequent itemsets.
    # min_threshold=0.6 means only rules with confidence ≥ 60 % are returned.
    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=0.6
    )

    # Sort by lift so the most surprising / meaningful rules appear first.
    rules = rules.sort_values("lift", ascending=False)

    print(f"\nFound {len(frequent_itemsets)} frequent itemsets")
    print(f"Found {len(rules)} association rules (confidence ≥ 0.6)\n")
    print("Top 10 rules by Lift:")
    print(
        rules[["antecedents", "consequents", "support", "confidence", "lift"]]
        .head(10)
        .to_string(index=False)
    )


# SECTION 15 – RISK SCORING SYSTEM
# A simple interpretable risk score (0–100) that summarises the model outputs
# and could be shown to a clinician or patient.
print("\n" + "=" * 60)
print("SECTION 15 – RISK SCORING SYSTEM")
print("=" * 60)

# Use the best-performing classifier (Random Forest) to get probability of
# heart disease for every patient in the test set.
rf_best = rf_result["fitted_model"]
risk_proba = rf_best.predict_proba(X_test)[:, 1]   # P(class=1)

# Scale to 0–100 for readability.
risk_scores = (risk_proba * 100).round(1)

risk_output = pd.DataFrame({
    "predicted_risk_score": risk_scores,
    "risk_band": pd.cut(
        risk_scores,
        bins=[0, 33, 66, 100],
        labels=["Low (<33)", "Medium (33–66)", "High (>66)"]
    ),
    "actual_target": y_test.values
})

print("\nRisk band distribution in test set:")
print(risk_output["risk_band"].value_counts())
print("\nMean actual disease rate per risk band:")
print(risk_output.groupby("risk_band")["actual_target"].mean().round(3))
print("\nSample risk scores (first 10 test patients):")
print(risk_output.head(10).to_string(index=False))


# SECTION 16 – FINAL SUMMARY
print("\n" + "=" * 60)
print("SECTION 16 – FINAL SUMMARY")
print("=" * 60)

print(f"""
  Dataset Shape      : {final_df.shape}
  Target Distribution:
    No Heart Disease : {(final_df['target'] == 0).sum()}
    Heart Disease    : {(final_df['target'] == 1).sum()}

  Classification Results:
    Best model by F1 : {clf_summary.loc[clf_summary['f1'].idxmax(), 'model']}
    Best F1-Score    : {clf_summary['f1'].max():.4f}

  Regression (Multiple):
    RMSE (cholesterol): {rmse_multi:.4f}
    R²   (cholesterol): {r2_multi:.4f}

  Clustering: 3 clusters (Low / Medium / High Risk)

  Star Schema DW saved to: heart_disease_dw.db
  Final Dataset saved to : final_unified_dataset.csv
""")

print("=" * 60)
print("PREPROCESSING AND ANALYSIS COMPLETE")
print("=" * 60)
