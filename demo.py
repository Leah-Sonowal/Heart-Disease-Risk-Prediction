"""
===========================================================================
HEART DISEASE RISK PREDICTION SYSTEM
Pipeline (this file):
  PART 1: Data Preprocessing
      1a. Load raw CSVs
      1b. Data Integration  (map each source to a common 24-column schema)
      1c. Combine & Deduplicate
      1d. Missing Value Imputation
      1e. Outlier Removal  (Isolation Forest)
      1f. Encoding         (One-Hot + binary cast)
      1g. Standardisation  (Z-score)
      1h. Save final_unified_dataset.csv

  PART 2: Star Schema Data Warehouse  (SQLite)
      Tables : dim_patient, dim_clinical, dim_lifestyle, heart_disease_fact
      OLAP   : 3 roll-up queries across dimensions

  PART 3: Train / Test Split

  PART 4: Classification
      Logistic Regression, Decision Tree, Random Forest
      Metrics : Accuracy, Precision, Recall, F1-Score
      Bonus   : Top-10 feature importances from Random Forest

  PART 5: Clustering  (K-Means)
      Elbow method to validate k=3
      Automatic Low / Medium / High risk labelling

"""
import warnings
warnings.filterwarnings("ignore")   # suppress sklearn convergence warnings

import sqlite3                     

import pandas as pd
import numpy as np

# Preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

# Feature selection 
from sklearn.feature_selection import SelectKBest, f_classif

# Model selection
from sklearn.model_selection import train_test_split

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Clustering
from sklearn.cluster import KMeans

# Evaluation metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report
)

# PART 1 – DATA PREPROCESSING
print("=" * 60)
print("PART 1 – DATA PREPROCESSING")
print("=" * 60)


# 1a. LOAD RAW DATA
# Each dataset comes from a different medical study and uses different column
# names, units, and encodings.  We read them all as-is before any changes.

uci    = pd.read_csv("uci_heart.csv")
fr     = pd.read_csv("framingham.csv")
cardio = pd.read_csv("cardio.csv", sep=";")

print(f"\n[1a] Raw row counts loaded:")
print(f"     UCI Heart Disease : {len(uci):>6} rows")
print(f"     Framingham Study  : {len(fr):>6} rows")
print(f"     Cardio Dataset    : {len(cardio):>6} rows")



# 1b. DATA INTEGRATION
# We define a canonical (standard) schema with 24 columns.
# Every source dataset is mapped into this schema.
# Any column a dataset does NOT have is filled with np.nan (explicit missing).

print("\n[1b] Data Integration – mapping all sources to the canonical schema ...")

# ─ CARDIO 

# age: stored in DAYS in the original file → divide by 365 to get years.
cardio["age"] = cardio["age"] / 365

# gender: original  1=female, 2=male
#         canonical 0=female, 1=male  (to match UCI and Framingham conventions)
cardio["gender"] = cardio["gender"].map({1: 0, 2: 1})

# BMI: not stored directly; calculated from height (cm) and weight (kg).
#      Formula: BMI = weight_kg / (height_m)^2
cardio["BMI"] = cardio["weight"] / ((cardio["height"] / 100) ** 2)

# cholesterol: stored as ordinal levels → mapped to representative mg/dL values
#   1 = normal        →  180 mg/dL
#   2 = above normal  →  220 mg/dL
#   3 = well above    →  260 mg/dL
cardio["cholesterol"] = cardio["cholesterol"].map({1: 180, 2: 220, 3: 260})

# gluc (glucose): same ordinal-to-numeric mapping as cholesterol.
#   1 = normal        →  85 mg/dL
#   2 = above normal  → 110 mg/dL
#   3 = well above    → 140 mg/dL
cardio["gluc"] = cardio["gluc"].map({1: 85, 2: 110, 3: 140})

# Build cardio_new – map available columns; filled absent ones with explicit NaN.
cardio_new = pd.DataFrame({
    "age"         : cardio["age"],
    "gender"      : cardio["gender"],
    "BMI"         : cardio["BMI"],
    "sysBP"       : cardio["ap_hi"],       # systolic blood pressure  (mm Hg)
    "diaBP"       : cardio["ap_lo"],       # diastolic blood pressure (mm Hg)
    "cholesterol" : cardio["cholesterol"],
    "glucose"     : cardio["gluc"],
    "smoke"       : cardio["smoke"],
    "alcohol"     : cardio["alco"],
    "active"      : cardio["active"],
    # ---------- columns absent in the cardio dataset → explicit NaN ----------
    "BPMeds"      : np.nan,
    "stroke"      : np.nan,
    "hypertension": np.nan,
    "diabetes"    : np.nan,
    "heartRate"   : np.nan,
    "cp"          : np.nan,    # chest pain type
    "restecg"     : np.nan,    # resting ECG result
    "thalach"     : np.nan,    # max heart rate during exercise
    "exang"       : np.nan,    # exercise-induced angina
    "oldpeak"     : np.nan,    # ST depression (exercise vs rest)
    "slope"       : np.nan,    # slope of peak exercise ST segment
    "ca"          : np.nan,    # major vessels coloured by fluoroscopy (0-3)
    "thal"        : np.nan,    # thalassemia type
    # -------------------------------------------------------------------------
    "target"      : cardio["cardio"],      # 1 = cardiovascular disease present
})

# ─FRAMINGHAM 
# Most columns are already in the correct units and format.
# 'male' (1=male, 0=female) already matches the canonical gender convention.
# 'TenYearCHD' = 1 if the patient developed coronary heart disease within 10 yr.

fr_new = pd.DataFrame({
    "age"         : fr["age"],
    "gender"      : fr["male"],
    "BMI"         : fr["BMI"],
    "sysBP"       : fr["sysBP"],
    "diaBP"       : fr["diaBP"],
    "cholesterol" : fr["totChol"],
    "glucose"     : fr["glucose"],
    "smoke"       : fr["currentSmoker"],
    # ---------- columns absent in Framingham → explicit NaN -----------------
    "alcohol"     : np.nan,
    "active"      : np.nan,
    # ---------- columns present in Framingham --------------------------------
    "BPMeds"      : fr["BPMeds"],
    "stroke"      : fr["prevalentStroke"],
    "hypertension": fr["prevalentHyp"],
    "diabetes"    : fr["diabetes"],
    "heartRate"   : fr["heartRate"],
    # ---------- UCI-specific ECG/clinical columns absent → NaN --------------
    "cp"          : np.nan,
    "restecg"     : np.nan,
    "thalach"     : np.nan,
    "exang"       : np.nan,
    "oldpeak"     : np.nan,
    "slope"       : np.nan,
    "ca"          : np.nan,
    "thal"        : np.nan,
    # -------------------------------------------------------------------------
    "target"      : fr["TenYearCHD"],
})

# ── UCI ──────────────────────────────────────────────────────────────────────
# fbs (fasting blood sugar): stored as a binary flag → convert to numeric.
#   0 = blood sugar <= 120 mg/dL → representative value  90 mg/dL (normal)
#   1 = blood sugar >  120 mg/dL → representative value 130 mg/dL (elevated)
uci["fbs"] = uci["fbs"].map({0: 90, 1: 130})

uci_new = pd.DataFrame({
    "age"         : uci["age"],
    "gender"      : uci["sex"],           # 1=male, 0=female (already canonical)
    # ---------- columns absent in UCI → explicit NaN -------------------------
    "BMI"         : np.nan,
    # -------------------------------------------------------------------------
    "sysBP"       : uci["trestbps"],      # resting blood pressure  (mm Hg)
    # ---------- columns absent in UCI → explicit NaN -------------------------
    "diaBP"       : np.nan,
    # -------------------------------------------------------------------------
    "cholesterol" : uci["chol"],          # serum cholesterol (mg/dL)
    "glucose"     : uci["fbs"],           # converted from binary flag above
    # ---------- columns absent in UCI → explicit NaN -------------------------
    "smoke"       : np.nan,
    "alcohol"     : np.nan,
    "active"      : np.nan,
    "BPMeds"      : np.nan,
    "stroke"      : np.nan,
    "hypertension": np.nan,
    "diabetes"    : np.nan,
    "heartRate"   : np.nan,
    # ---------- UCI-specific clinical / ECG columns --------------------------
    "cp"          : uci["cp"],            # chest pain type        (0-3)
    "restecg"     : uci["restecg"],       # resting ECG result     (0-2)
    "thalach"     : uci["thalach"],       # max heart rate achieved
    "exang"       : uci["exang"],         # exercise-induced angina (0/1)
    "oldpeak"     : uci["oldpeak"],       # ST depression
    "slope"       : uci["slope"],         # slope of ST segment
    "ca"          : uci["ca"],            # number of major vessels (0-3)
    "thal"        : uci["thal"],          # thalassemia (1=normal, 2=fixed, 3=reversible)
    # -------------------------------------------------------------------------
    "target"      : uci["target"],        # 1 = heart disease present
})

print("     Integration complete.")


# 1c. COMBINE & DEDUPLICATE
# pd.concat stacks the three harmonised frames vertically.
# ignore_index=True resets the row index to a fresh 0-based integer sequence
# so there are no duplicate index values carried over from the original frames.

df = pd.concat([cardio_new, fr_new, uci_new], ignore_index=True)

before_dedup  = len(df)
df.drop_duplicates(inplace=True)      # remove fully identical rows in place
rows_dropped  = before_dedup - len(df)

print(f"\n[1c] Combine & Deduplicate:")
print(f"     Total rows after concat  : {before_dedup}")
print(f"     Duplicate rows removed   : {rows_dropped}")
print(f"     Rows remaining           : {len(df)}")


# 1d. MISSING VALUE IMPUTATION
# Two strategies are used based on the nature of each column:
#
# Continuous columns  →  MEAN imputation
#   The mean is the best single "neutral" estimate for a numeric quantity.
#   It preserves the column's overall average and doesn't introduce a value
#   that doesn't naturally exist in the data.
#
# Categorical / binary columns  →  MODE imputation
#   The mode (most frequent value) preserves the dominant class.
#   Using the mean on a 0/1 column would produce fractions like 0.3 which
#   are not valid category labels and mislead the model.

# Continuous columns (will also be standardised later in step 1g)
num_cols = [
    "age", "BMI", "sysBP", "diaBP", "heartRate",
    "thalach", "oldpeak", "cholesterol", "glucose"
]

# Categorical / binary columns
cat_cols = [
    "gender", "smoke", "alcohol", "active",
    "BPMeds", "stroke", "hypertension", "diabetes",
    "exang", "cp", "restecg", "slope", "thal", "ca"
]

# Vectorised mean imputation across all numeric columns at once
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

# Mode imputation column-by-column (mode can differ between columns)
# .mode() returns a Series; [0] selects the first (most common) value.
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

remaining_nulls = df.isnull().sum().sum()
print(f"\n[1d] Missing Value Imputation:")
print(f"     Numeric  columns imputed with mean  ({len(num_cols)} cols)")
print(f"     Categoric columns imputed with mode ({len(cat_cols)} cols)")
print(f"     Remaining null values after imputation : {remaining_nulls}")


# 1e. OUTLIER REMOVAL – Isolation Forest
# Why Isolation Forest instead of IQR or Z-score?
#   Our dataset has many features (high-dimensional).  Univariate methods
#   like IQR examine each column independently and miss MULTIVARIATE outliers –
#   rows where no single column is extreme but the combination is unusual
#   (e.g. a very young patient with very high cholesterol AND high BP).


iso            = IsolationForest(contamination=0.05, random_state=42)
outlier_labels = iso.fit_predict(df[num_cols])

before_outlier   = len(df)
df               = df[outlier_labels == 1].copy()   # keep only inliers 
outliers_removed = before_outlier - len(df)

print(f"\n[1e] Outlier Removal (Isolation Forest, contamination=5%):")
print(f"     Rows flagged as outliers : {outliers_removed}")
print(f"     Clean rows remaining     : {len(df)}")


# 1f. ENCODING
# Machine-learning models only accept numeric inputs.
# We handle two types of categorical columns differently:
#
# Type A – Nominal multi-class columns  (cp, restecg, slope, thal)
#   These have 3+ categories with NO natural numeric ordering (e.g. chest pain
#   type 0, 1, 2, 3 do not imply any magnitude ordering).
#   Solution: ONE-HOT ENCODING – expand each column into k-1 binary columns.
#   drop_first=True drops the first dummy to prevent the DUMMY VARIABLE TRAP:
#   when k dummies are created for k categories, one is always a linear
#   combination of the others, causing multicollinearity in linear models.
#
# Type B – Binary / already-numeric columns  (gender, smoke, BPMeds, …)
#   Already 0/1 or naturally ordinal.  We just cast to int to ensure a
#   consistent dtype throughout the DataFrame.

ohe_cols = ["cp", "restecg", "slope", "thal"]

# Cast to int BEFORE get_dummies so dummy column names use integers
# (e.g. cp_1, cp_2) rather than floats (cp_1.0, cp_2.0).
df[ohe_cols] = df[ohe_cols].astype(int)

# Expand OHE columns into binary dummy columns.
# The original columns are automatically dropped from df.
df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)

# Newer versions of pandas return bool dtype for dummy columns → cast to int
# so the entire DataFrame stays uniformly numeric (all 0s and 1s).
bool_cols     = [col for col in df.columns if df[col].dtype == bool]
df[bool_cols] = df[bool_cols].astype(int)

# Cast all remaining binary columns to int for type consistency
binary_cols = [
    "gender", "smoke", "alcohol", "active",
    "BPMeds", "stroke", "hypertension", "diabetes", "exang", "ca"
]
df[binary_cols] = df[binary_cols].astype(int)

print(f"\n[1f] Encoding:")
print(f"     OHE applied to columns   : {ohe_cols}")
print(f"     Total columns after OHE  : {df.shape[1]}")
print(f"     Full column list         : {list(df.columns)}")


# 1g. STANDARDISATION
#   Features have very different scales:
#     cholesterol  ~ 200 mg/dL
#     age          ~  50 years
#     oldpeak      ~   1.0
#   Without standardisation, features with large magnitudes dominate
#   and end up biasing the model.
#
# StandardScaler applies Z-score normalisation per column:
#       z = (x - column_mean) / column_std
#
# After scaling:
#   Each continuous column has mean ≈ 0 and standard deviation ≈ 1.

# Save the target column
target = df["target"].copy()

scaler       = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# Rebuild the scaled DataFrame and reattach the original (unscaled) target.
df_scaled           = df.copy()
df_scaled["target"] = target.values

print(f"\n[1g] Standardisation (Z-score, continuous columns only):")
print(df_scaled[num_cols].describe().round(2))


# 1h. SAVE FINAL DATASET
final_df = df_scaled.copy()

final_df.to_csv("final_unified_dataset.csv", index=False)

print(f"\n[1h] Final dataset saved to final_unified_dataset.csv")
print(f"     Shape   : {final_df.shape}")
print(f"     Preview (first 3 rows):")
print(final_df.head(3).to_string())

# PART 2 – STAR SCHEMA DATA WAREHOUSE
# Our Star Schema design:
#   Fact table  : heart_disease_fact
#   Dimensions  : dim_patient   (demographics + pre-existing conditions)
#                 dim_clinical  (lab results, vital signs, medication)
#                 dim_lifestyle (behavioural / modifiable risk factors)

print("\n" + "=" * 60)
print("PART 2 – STAR SCHEMA DATA WAREHOUSE")
print("=" * 60)

#Construct database using sqlite
conn   = sqlite3.connect("heart_disease_dw.db")
cursor = conn.cursor()

cursor.execute("DROP TABLE IF EXISTS heart_disease_fact")
cursor.execute("DROP TABLE IF EXISTS dim_patient")
cursor.execute("DROP TABLE IF EXISTS dim_clinical")
cursor.execute("DROP TABLE IF EXISTS dim_lifestyle")

# ─Dimension Table: dim_patient 

cursor.execute("""
CREATE TABLE dim_patient (
    patient_id   INTEGER PRIMARY KEY,   -- Surrogate key (auto-assigned integer)
    age          REAL,                  -- Age in years (z-scored)
    gender       INTEGER,               -- 0 = Female,  1 = Male
    BMI          REAL,                  -- Body Mass Index (z-scored)
    hypertension INTEGER,               -- 1 = diagnosed hypertension, 0 = no
    diabetes     INTEGER,               -- 1 = diabetic,               0 = no
    stroke       INTEGER                -- 1 = prior stroke,            0 = no
)
""")

# ── Dimension Table: dim_clinical ─────────────────────────────────────────
# Answers: WHAT are the patient's clinical measurements?
# Contains lab results and medication status at the time of assessment.
cursor.execute("""
CREATE TABLE dim_clinical (
    clinical_id  INTEGER PRIMARY KEY,   -- Surrogate key
    sysBP        REAL,                  -- Systolic blood pressure  (z-scored)
    diaBP        REAL,                  -- Diastolic blood pressure (z-scored)
    cholesterol  REAL,                  -- Total cholesterol        (z-scored)
    glucose      REAL,                  -- Blood glucose            (z-scored)
    heartRate    REAL,                  -- Resting heart rate       (z-scored)
    BPMeds       INTEGER                -- 1 = on BP medication, 0 = not
)
""")

# ── Dimension Table: dim_lifestyle ────────────────────────────────────────
# Answers: HOW does the patient live?
# Contains modifiable behavioural attributes (can change with intervention).
cursor.execute("""
CREATE TABLE dim_lifestyle (
    lifestyle_id INTEGER PRIMARY KEY,   -- Surrogate key
    smoke        INTEGER,               -- 1 = current smoker, 0 = non-smoker
    alcohol      INTEGER,               -- 1 = regular alcohol use, 0 = no
    active       INTEGER                -- 1 = physically active,   0 = sedentary
)
""")

# ── Fact Table: heart_disease_fact ────────────────────────────────────────
# The central hub of the star schema.
# Contains:
#   • Foreign keys linking to each dimension (patient, clinical, lifestyle)
#   • The measured outcome  (target = 1 means heart disease / high risk)
#   • A heuristic risk_score: fraction of binary risk flags that are set
cursor.execute("""
CREATE TABLE heart_disease_fact (
    fact_id      INTEGER PRIMARY KEY,
    patient_id   INTEGER REFERENCES dim_patient(patient_id),
    clinical_id  INTEGER REFERENCES dim_clinical(clinical_id),
    lifestyle_id INTEGER REFERENCES dim_lifestyle(lifestyle_id),
    target       INTEGER,    -- 1 = heart disease / high cardiovascular risk
    risk_score   REAL        -- Heuristic 0-1 score (fraction of risk flags active)
)
""")

# ── Populate all four tables ───────────────────────────────────────────────
# We build one row-tuple per table per patient and then use executemany()
# for bulk insertion, which is far faster than looping cursor.execute().
#
# Surrogate keys are 1-based (pk = i + 1) so they are more readable in SQL.

patient_rows   = []
clinical_rows  = []
lifestyle_rows = []
fact_rows      = []

# Risk factors used to compute the heuristic risk_score.
# risk_score = (number of active flags) / (total number of flags)
risk_factor_cols = ["smoke", "hypertension", "diabetes", "stroke", "BPMeds"]

for i, row in final_df.reset_index(drop=True).iterrows():
    pk = i + 1    # 1-based surrogate key shared across all four tables

    # dim_patient: demographics and pre-existing conditions
    patient_rows.append((
        pk,
        row.get("age", None),
        int(row.get("gender", 0)),
        row.get("BMI", None),
        int(row.get("hypertension", 0)),
        int(row.get("diabetes", 0)),
        int(row.get("stroke", 0)),
    ))

    # dim_clinical: lab results and medication
    clinical_rows.append((
        pk,
        row.get("sysBP", None),
        row.get("diaBP", None),
        row.get("cholesterol", None),
        row.get("glucose", None),
        row.get("heartRate", None),
        int(row.get("BPMeds", 0)),
    ))

    # dim_lifestyle: behavioural attributes
    lifestyle_rows.append((
        pk,
        int(row.get("smoke", 0)),
        int(row.get("alcohol", 0)),
        int(row.get("active", 0)),
    ))

    # Compute risk_score
    active_flags = sum(int(row.get(c, 0)) for c in risk_factor_cols)
    risk_score   = active_flags / len(risk_factor_cols)

    # fact row: links all three dimensions + stores the outcome
    fact_rows.append((pk, pk, pk, pk, int(row["target"]), risk_score))

# Bulk insert into each table
cursor.executemany("INSERT INTO dim_patient   VALUES (?,?,?,?,?,?,?)", patient_rows)
cursor.executemany("INSERT INTO dim_clinical  VALUES (?,?,?,?,?,?,?)", clinical_rows)
cursor.executemany("INSERT INTO dim_lifestyle VALUES (?,?,?,?)",       lifestyle_rows)
cursor.executemany("INSERT INTO heart_disease_fact VALUES (?,?,?,?,?,?)", fact_rows)
conn.commit()

print("\nStar schema created and populated → heart_disease_dw.db")
print("Table row counts:")
for tbl in ["dim_patient", "dim_clinical", "dim_lifestyle", "heart_disease_fact"]:
    n = cursor.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
    print(f"  {tbl:<25} : {n} rows")

# ─OLAP Queries 
print("\n[OLAP] Multidimensional Roll-Up Queries:")

# Shows how the average risk score and total disease count differ by sex.
q1 = pd.read_sql_query("""
    SELECT
        p.gender,
        ROUND(AVG(f.risk_score), 4) AS avg_risk_score,
        COUNT(*)                    AS patient_count,
        SUM(f.target)               AS disease_count
    FROM heart_disease_fact f
    JOIN dim_patient p ON f.patient_id = p.patient_id
    GROUP BY p.gender
""", conn)
print("\n  Q1 – Average risk score by gender  (0=Female, 1=Male):")
print(q1.to_string(index=False))

# Q2: Roll up by SMOKING STATUS (lifestyle dimension)
q2 = pd.read_sql_query("""
    SELECT
        l.smoke,
        ROUND(AVG(f.target), 4) AS disease_rate,
        COUNT(*)                AS patient_count
    FROM heart_disease_fact f
    JOIN dim_lifestyle l ON f.lifestyle_id = l.lifestyle_id
    GROUP BY l.smoke
""", conn)
print("\n  Q2 – Disease rate by smoking status  (0=Non-smoker, 1=Smoker):")
print(q2.to_string(index=False))

# Q3: Roll up by BP MEDICATION status (clinical dimension)
# Patients on BP medication are typically already hypertensive, so we expect
# a higher disease rate in this group.
q3 = pd.read_sql_query("""
    SELECT
        c.BPMeds,
        ROUND(AVG(f.target), 4) AS disease_rate,
        COUNT(*)                AS patient_count
    FROM heart_disease_fact f
    JOIN dim_clinical c ON f.clinical_id = c.clinical_id
    GROUP BY c.BPMeds
""", conn)
print("\n  Q3 – Disease rate by BP medication  (0=Not on meds, 1=On meds):")
print(q3.to_string(index=False))

conn.close()   # close the database connection; we are done with it


# PART 3 – TRAIN / TEST SPLIT
# We have performed a SINGLE split and reuse the same X_train / X_test / y_train /
#
# Parameters:
#   test_size=0.2      → 80% training data, 20% test data
#   stratify=y         → preserves the class ratio (disease vs no-disease) in
#                        both splits.  Important when classes are imbalanced.
#   random_state=42    → fixed seed for reproducibility

print("\n" + "=" * 60)
print("PART 3 – TRAIN / TEST SPLIT")
print("=" * 60)

X = final_df.drop("target", axis=1)   
y = final_df["target"]                 # binary outcome label

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\n  Training set size  : {X_train.shape[0]} rows  ({X_train.shape[1]} features)")
print(f"  Test set size      : {X_test.shape[0]} rows")
print(f"  Class balance (train) : {y_train.value_counts().to_dict()}")
print(f"  Class balance (test)  : {y_test.value_counts().to_dict()}")


# ===========================================================================
# PART 4 – CLASSIFICATION
# ===========================================================================
# Goal: predict the binary target (heart disease present = 1 / absent = 0).
#
# Three classifiers are trained and compared:
#   1. Logistic Regression  – linear probabilistic baseline
#   2. Decision Tree        – rule-based tree (interpretable)
#   3. Random Forest        – ensemble of many trees (typically highest accuracy)

print("\n" + "=" * 60)
print("PART 4 – CLASSIFICATION")
print("=" * 60)


def evaluate_classifier(name, model, X_tr, y_tr, X_te, y_te):
    """
    Train a classifier on X_tr/y_tr, predict on X_te, and print metrics.

    Parameters
    ----------
    name  : str                – human-readable model name for display
    model : sklearn estimator  – untrained classifier instance
    X_tr  : DataFrame          – training features
    y_tr  : Series             – training labels
    X_te  : DataFrame          – test features (never used during training)
    y_te  : Series             – true test labels

    Returns
    -------
    dict with keys: model, accuracy, precision, recall, f1, fitted_model
    """
    model.fit(X_tr, y_tr)        # train on training partition
    y_pred = model.predict(X_te) # predict on held-out test partition

    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, zero_division=0)
    rec  = recall_score(y_te, y_pred, zero_division=0)
    f1   = f1_score(y_te, y_pred, zero_division=0)

    print(f"\n  -- {name} {'─' * max(0, 42 - len(name))}")
    print(f"     Accuracy  : {acc:.4f}")
    print(f"     Precision : {prec:.4f}  "
          "(of predicted positives, what fraction are truly positive)")
    print(f"     Recall    : {rec:.4f}  "
          "(of actual positives, what fraction were correctly found)")
    print(f"     F1-Score  : {f1:.4f}  "
          "(harmonic mean of precision and recall)")
    print()
    print(classification_report(y_te, y_pred, zero_division=0,
                                 target_names=["No Disease", "Disease"]))

    return {
        "model"       : name,
        "accuracy"    : acc,
        "precision"   : prec,
        "recall"      : rec,
        "f1"          : f1,
        "fitted_model": model,
    }


# ── 4A. Logistic Regression ──────────────────────────────────────────────
# A linear model that learns a weight for each feature.
# It applies the logistic (sigmoid) function to produce a probability in (0,1),
# then classifies the patient as disease=1 if that probability exceeds 0.5.
# Simple, fast, and coefficients are directly interpretable.
# max_iter=1000 gives the optimiser enough iterations to converge on large data.
lr_result = evaluate_classifier(
    "Logistic Regression",
    LogisticRegression(max_iter=1000, random_state=42),
    X_train, y_train, X_test, y_test
)

# ── 4B. Decision Tree ────────────────────────────────────────────────────
# Builds a binary tree of if-then-else rules by recursively choosing the
# feature split that best separates the classes.
# max_depth=5 limits the depth to prevent overfitting.
dt_result = evaluate_classifier(
    "Decision Tree",
    DecisionTreeClassifier(max_depth=5, random_state=42),
    X_train, y_train, X_test, y_test
)

# ── 4C. Random Forest ────────────────────────────────────────────────────
# Trains n_estimators independent decision trees.  Each tree:
#   • Is trained on a RANDOM BOOTSTRAP SAMPLE of the training data.
#   • Splits on a RANDOM SUBSET of features at each node.
# Final prediction = MAJORITY VOTE across all trees.
#
# Why better than a single tree?
#   Individual trees overfit their bootstrap sample.  Averaging many
#   uncorrelated trees cancels out individual errors (variance reduction).
#   This is called "bagging" (Bootstrap AGGregatING).
#
# n_estimators=100 → 100 trees.
# n_jobs=-1        → use all CPU cores for faster parallel training.
rf_result = evaluate_classifier(
    "Random Forest",
    RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    X_train, y_train, X_test, y_test
)

# ── Summary Table ─────────────────────────────────────────────────────────
clf_summary = pd.DataFrame([
    {k: v for k, v in r.items() if k != "fitted_model"}
    for r in [lr_result, dt_result, rf_result]
])
print("  Classification Summary (all models evaluated on the same test set):")
print(clf_summary.to_string(index=False))

best = clf_summary.loc[clf_summary["f1"].idxmax()]
print(f"\n  Best model by F1-Score : {best['model']}  (F1 = {best['f1']:.4f})")

# -Feature Importance – Random Forest 
# Each feature receives an importance score 
# that splitting on that feature causes, averaged across all trees.
# Higher importance → the feature contributes more to correct predictions.
rf_model = rf_result["fitted_model"]
feat_imp = pd.Series(rf_model.feature_importances_, index=X.columns)
feat_imp = feat_imp.sort_values(ascending=False).head(10)

print("\n  Top-10 Feature Importances (Random Forest – mean Gini decrease):")
for rank, (feat, score) in enumerate(feat_imp.items(), 1):
    bar = "=" * int(score * 300)    # ASCII bar proportional to importance
    print(f"    {rank:>2}. {feat:<22}  {bar}  {score:.4f}")



# PART 5 – CLUSTERING  (K-Means)

# We use k=3 as specified in the project:
#   One cluster → Low Risk patients
#   One cluster → Medium Risk patients
#   One cluster → High Risk patients
#
# The cluster-to-risk mapping is determined AUTOMATICALLY:
#   Compute the MEAN TARGET (disease rate) within each cluster and rank.
#   Cluster with highest disease rate  →  "High Risk"
#   Cluster with lowest disease rate   →  "Low Risk"
#
# We validate k=3 with the ELBOW METHOD:
#   Plot WCSS (Within-Cluster Sum of Squares) for k = 2..8.
#   The "elbow" is the k where WCSS stops dropping sharply.

print("\n" + "=" * 60)
print("PART 5 – CLUSTERING  (K-Means, k=3)")
print("=" * 60)

cluster_features = num_cols   # same list as used for standardisation in 1g

# ─ Elbow Method 
print("\n  Elbow Method – WCSS for k=2 to k=8:")
wcss = []
for k in range(2, 9):
    km_elbow = KMeans(n_clusters=k, random_state=42, n_init=10)
    km_elbow.fit(final_df[cluster_features])
    wcss.append(km_elbow.inertia_)

    bar = "#" * int(km_elbow.inertia_ / wcss[0] * 32)    
    print(f"    k={k}  WCSS={km_elbow.inertia_:>12,.1f}  {bar}")

print("  (The 'elbow' is where the bar length stops dropping sharply → optimal k.)")

# ─ Final K-Means fit with k=3 
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

# fit_predict trains the model AND assigns a cluster label to every row.
# Cluster labels are arbitrary integers (0, 1, 2) – we map them below.
final_df["cluster"] = kmeans.fit_predict(final_df[cluster_features])

# ─ Automatic risk labelling
# K-Means does not know which cluster means "high risk" – that depends on the
# disease rate within each cluster.  We compute mean target per cluster and rank:
#   Highest disease rate → "High Risk"
#   Lowest  disease rate → "Low Risk"
cluster_disease_rate = final_df.groupby("cluster")["target"].mean().sort_values()

risk_map = {
    cluster_disease_rate.index[0]: "Low Risk",     # lowest disease rate
    cluster_disease_rate.index[1]: "Medium Risk",
    cluster_disease_rate.index[2]: "High Risk",    # highest disease rate
}
final_df["risk_label"] = final_df["cluster"].map(risk_map)

# ─ Cluster summary 
print("\n  Cluster profiles (auto-labelled by disease rate):")
cluster_summary = (
    final_df.groupby("risk_label")["target"]
    .agg(patient_count="count", disease_rate="mean")
    .round(3)
)
print(cluster_summary.to_string())

print("\n  Cluster centroids (standardised scale, continuous features):")
centroids       = pd.DataFrame(kmeans.cluster_centers_, columns=cluster_features)
centroids.index = [risk_map[i] for i in range(3)]   # label rows by risk band
print(centroids.round(3).to_string())


# FINAL SUMMARY
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print(f"""
  PART 1 – PREPROCESSING
    Final dataset            : {final_df.shape[0]} rows x {final_df.shape[1]} columns
    Saved to                 : final_unified_dataset.csv
    Class balance:
      No heart disease (0)   : {(final_df['target'] == 0).sum()}
      Heart disease    (1)   : {(final_df['target'] == 1).sum()}

  PART 2 – DATA WAREHOUSE
    Schema type              : Star schema  (1 fact + 3 dimensions)
    Database file            : heart_disease_dw.db  (SQLite)
    OLAP queries run         : 3 (by gender, smoking, BP medication)

  PART 3 – TRAIN / TEST SPLIT
    Training rows            : {X_train.shape[0]}
    Test rows                : {X_test.shape[0]}
    Split ratio              : 80 / 20  (stratified)

  PART 4 – CLASSIFICATION  (test set = {X_test.shape[0]} rows)
    Models trained           : Logistic Regression, Decision Tree, Random Forest
    Best model by F1         : {clf_summary.loc[clf_summary['f1'].idxmax(), 'model']}
    Best F1-Score            : {clf_summary['f1'].max():.4f}
    Best Accuracy            : {clf_summary.loc[clf_summary['accuracy'].idxmax(), 'accuracy']:.4f}

  PART 5 – CLUSTERING
    Algorithm                : K-Means  (k=3)
    Risk labels              : Low Risk / Medium Risk / High Risk
    Elbow method range       : k = 2 to 8
""")

print("=" * 60)
print("PIPELINE COMPLETE")
print("=" * 60)
