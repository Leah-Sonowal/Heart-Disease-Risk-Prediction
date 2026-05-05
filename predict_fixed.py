"""

HEART DISEASE RISK PREDICTOR  –  Interactive Terminal Tool
This script loads the final_unified_dataset.csv produced by the main
pipeline (heart_disease_project.py), trains a Random Forest classifier on
it, and then walks any user through a series of plain-English questions to
collect their health data.  It then:
  • Scales the inputs the same way the training data was scaled
  • Runs the model and reports the predicted risk (Low / Medium / High)
  • Prints a probability bar and a brief explanation of the top risk factors

"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib                              # FIX: load the saved scaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ── ANSI colour helpers ──────
RESET  = "\033[0m"
BOLD   = "\033[1m"
RED    = "\033[91m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"
DIM    = "\033[2m"


def clr(text, colour):
    """Wrap text in an ANSI colour code."""
    return f"{colour}{text}{RESET}"


def banner():
    """Print the welcome banner."""
    print()
    print(clr("=" * 62, CYAN))
    print(clr("    HEART DISEASE RISK PREDICTOR", BOLD + RED))
    #print(clr("       NIT Sikkim – DWM Lab (CS14211)", DIM))
    print(clr("=" * 62, CYAN))
    print()
    print("  This tool asks a few questions about your health and")
    print("  lifestyle, then uses a machine-learning model trained")
    print("  on 80 000+ patient records to estimate your risk of")
    print("  heart disease.\n")
    print(clr("=" * 62, CYAN))
    print()


# INPUT HELPERS
# Every helper below validates the user's input and keeps asking until a
# valid value is provided.  This prevents crashes from incorrect or invalid input.

def ask_float(prompt, lo, hi, unit=""):
    """
    Ask for a decimal number in the range [lo, hi].
    Re-prompts on invalid input.
    """
    hint = f"  ({lo}–{hi}{' ' + unit if unit else ''}): "
    while True:
        try:
            val = float(input(clr(prompt, BOLD) + hint))
            if lo <= val <= hi:
                return val
            print(clr(f"  ✗  Please enter a value between {lo} and {hi}.", RED))
        except ValueError:
            print(clr("  ✗  Please enter a number.", RED))


def ask_int(prompt, lo, hi, unit=""):
    """Ask for an integer in [lo, hi]."""
    hint = f"  ({lo}–{hi}{' ' + unit if unit else ''}): "
    while True:
        try:
            val = int(input(clr(prompt, BOLD) + hint))
            if lo <= val <= hi:
                return val
            print(clr(f"  ✗  Please enter a whole number between {lo} and {hi}.", RED))
        except ValueError:
            print(clr("  ✗  Please enter a whole number.", RED))


def ask_yn(prompt):
    """
    Ask a yes/no question.
    Returns 1 for yes, 0 for no.
    Accepts: y, yes, n, no (case-insensitive).
    """
    while True:
        raw = input(clr(prompt, BOLD) + "  (y/n): ").strip().lower()
        if raw in ("y", "yes", "1"):
            return 1
        if raw in ("n", "no", "0"):
            return 0
        print(clr("  ✗  Please type y or n.", RED))


def ask_choice(prompt, options):
    """
    Display a numbered menu and return the integer choice.
    options: list of strings describing each choice.
    Returns the chosen index (0-based).
    """
    print(clr(prompt, BOLD))
    for i, opt in enumerate(options):
        print(f"    {clr(str(i), CYAN)}  {opt}")
    while True:
        try:
            val = int(input("  Enter number: ").strip())
            if 0 <= val < len(options):
                return val
            print(clr(f"  ✗  Enter a number between 0 and {len(options)-1}.", RED))
        except ValueError:
            print(clr("  ✗  Please enter a number.", RED))


def section(title):
    """Print a visual section divider."""
    print()
    print(clr(f"  ── {title} ", CYAN) + clr("─" * (52 - len(title)), DIM))
    print()



# DATA COLLECTION
def collect_user_data():
    """
    Interactively collect all feature values from the user.

    The feature set matches the canonical schema in the main pipeline
    BEFORE one-hot encoding.  We apply OHE afterwards so the user only
    sees human-readable options.

    Returns
    -------
    dict : raw (unscaled) feature values in the canonical schema
    """
    data = {}

    # ── DEMOGRAPHICS 
    section("DEMOGRAPHICS")

    data["age"] = ask_float("Age", 18, 100, "years")

    gender_idx = ask_choice(
        "Biological sex:",
        ["Female", "Male"]
    )
    data["gender"] = gender_idx   # 0=Female, 1=Male

    data["BMI"] = ask_float(
        "BMI (Body Mass Index)",
        10.0, 70.0,
        "kg/m²  [weight_kg / height_m²]"
    )

    # ── VITAL SIGNS 
    section("VITAL SIGNS & LAB RESULTS")

    data["sysBP"] = ask_float(
        "Systolic blood pressure (upper number)",
        60, 260, "mm Hg"
    )
    data["diaBP"] = ask_float(
        "Diastolic blood pressure (lower number)",
        40, 160, "mm Hg"
    )
    data["heartRate"] = ask_float(
        "Resting heart rate",
        30, 220, "bpm"
    )
    data["cholesterol"] = ask_float(
        "Total cholesterol",
        80, 600, "mg/dL"
    )
    data["glucose"] = ask_float(
        "Fasting blood glucose",
        50, 400, "mg/dL"
    )

    # ── LIFESTYLE 
    section("LIFESTYLE")

    data["smoke"]   = ask_yn("Do you currently smoke?")
    data["alcohol"] = ask_yn("Do you regularly drink alcohol?")
    data["active"]  = ask_yn("Are you physically active (exercise ≥3×/week)?")

    # ── MEDICAL HISTORY 
    section("MEDICAL HISTORY")

    data["hypertension"] = ask_yn("Have you been diagnosed with hypertension (high BP)?")
    data["diabetes"]     = ask_yn("Have you been diagnosed with diabetes?")
    data["stroke"]       = ask_yn("Have you ever had a stroke?")
    data["BPMeds"]       = ask_yn("Are you currently on blood pressure medication?")

    # ── CLINICAL / ECG (optional but improves accuracy) 
    section("CLINICAL DETAILS ")

    print()

    # Chest pain type (0–3)
    cp_idx = ask_choice(
        "Chest pain type:",
        [
            "0 – No chest pain / typical angina",
            "1 – Atypical angina",
            "2 – Non-anginal pain",
            "3 – Asymptomatic"
        ]
    )
    data["cp"] = cp_idx

    # Max heart rate during exercise (thalach)
    data["thalach"] = ask_float(
        "Maximum heart rate achieved during exercise",
        60, 220, "bpm"
    )

    # Exercise-induced angina
    data["exang"] = ask_yn("Did exercise cause chest pain (exercise-induced angina)?")

    # ST depression (oldpeak)
    data["oldpeak"] = ask_float(
        "ST depression induced by exercise relative to rest",
        0.0, 10.0, ""
    )

    # Slope of peak exercise ST segment
    slope_idx = ask_choice(
        "Slope of peak exercise ST segment:",
        [
            "0 – Upsloping",
            "1 – Flat",
            "2 – Downsloping"
        ]
    )
    data["slope"] = slope_idx

    # Resting ECG results
    restecg_idx = ask_choice(
        "Resting ECG results:",
        [
            "0 – Normal",
            "1 – ST-T wave abnormality",
            "2 – Left ventricular hypertrophy"
        ]
    )
    data["restecg"] = restecg_idx

    # Number of major vessels coloured by fluoroscopy (0–3)
    data["ca"] = ask_int(
        "Number of major vessels coloured by fluoroscopy",
        0, 3
    )

    # Thalassemia type
    thal_idx = ask_choice(
        "Thalassemia result:",
        [
            "1 – Normal",
            "2 – Fixed defect",
            "3 – Reversible defect"
        ]
    )
    data["thal"] = thal_idx + 1   # options are labelled 1–3 in the dataset

    return data


# FEATURE ENGINEERING
# Replicate the exact same transformations the main pipeline applies so the
# model sees inputs in the format it was trained on.

def build_feature_row(raw, train_columns, scaler, num_cols):
    """
    Convert the raw user dict → a single-row DataFrame that matches the
    exact column layout the trained model expects.

    Steps:
      1. One-hot encode cp, restecg, slope, thal  (drop_first=True)
      2. Add any OHE dummy columns present in training but absent here (= 0)
      3. Drop any extra columns not in training
      4. Scale continuous columns with the fitted scaler
      5. Return a DataFrame with column order matching train_columns

    Parameters
    ----------
    raw          : dict   – raw values from collect_user_data()
    train_columns: list   – ordered list of column names the model was trained on
    scaler       : fitted StandardScaler
    num_cols     : list   – continuous column names to scale

    Returns
    -------
    pd.DataFrame  shape (1, len(train_columns))
    """

    # Start with a single-row DataFrame of the raw values
    row = pd.DataFrame([raw])

    # ── Apply the same OHE as the main pipeline 
    ohe_cols = ["cp", "restecg", "slope", "thal"]
    row[ohe_cols] = row[ohe_cols].astype(int)
    row = pd.get_dummies(row, columns=ohe_cols, drop_first=True)

    # Convert any booleans created by get_dummies to int
    bool_cols = [c for c in row.columns if row[c].dtype == bool]
    row[bool_cols] = row[bool_cols].astype(int)

    # Cast binary columns to int
    binary_cols = ["gender", "smoke", "alcohol", "active",
                   "BPMeds", "stroke", "hypertension", "diabetes", "exang", "ca"]
    for col in binary_cols:
        if col in row.columns:
            row[col] = row[col].astype(int)

    # ── Align columns with training set ───────────────────────────────────
    # Add any columns the training set has that we haven't generated (= 0)
    for col in train_columns:
        if col not in row.columns:
            row[col] = 0

    # Keep only the training columns in the correct order
    row = row[train_columns]

    # ── Scale continuous columns 
    # IMPORTANT: use the SAME scaler fitted on RAW data in index3.py.
    # The scaler was loaded from scaler.pkl and has the original population
    # mean and std.  This converts raw user values into the same z-score
    # space the training data lives in.
    
  
    COLLAPSED_COLS = {"heartRate", "thalach", "oldpeak"}
    # Pass ALL num_cols to transform at once — the scaler was fitted on all 9
    # together and requires all 9 column names to be present.
    present_num = [c for c in num_cols if c in row.columns]
    row[present_num] = scaler.transform(row[present_num])
    # Then overwrite the collapsed columns with 0.0 (their training z-score mean).
    # This neutralises the distortion caused by near-zero std in those columns.
    for col in COLLAPSED_COLS:
        if col in row.columns:
            row[col] = 0.0

    # ── Debugging: verify the scaled values are in-distribution ──────────
    safe_num = [c for c in num_cols if c in row.columns and c not in COLLAPSED_COLS]
    for col in safe_num:
        val = row[col].values[0]
        if abs(val) > 6:
            print(f"  ⚠ WARNING: '{col}' z-score={val:.1f} is far out of range. "
                  f"Check that scaler.pkl was produced by index3.py on raw data.")

    return row



# RISK DISPLAY
def probability_bar(prob, width=40):
    """
    Render a simple text progress bar showing the risk probability.

    prob  : float in [0, 1]
    width : total bar width in characters
    """
    filled = int(round(prob * width))
    empty  = width - filled

    if prob < 0.33:
        colour = GREEN
    elif prob < 0.66:
        colour = YELLOW
    else:
        colour = RED

    bar = clr("█" * filled, colour) + clr("░" * empty, DIM)
    return f"  [{bar}]  {clr(f'{prob*100:.1f}%', colour + BOLD)}"


def display_result(prob, feature_row, train_columns, model):
    """
    Print the final risk assessment to the terminal.

    Includes:
      • Risk band (Low / Medium / High)
      • Probability bar
      • Top contributing risk factors from the Random Forest
      • Brief lifestyle advice
    """
    print()
    print(clr("=" * 62, CYAN))
    print(clr("  PREDICTION RESULT", BOLD))
    print(clr("=" * 62, CYAN))
    print()

    # ── Risk band 
    if prob < 0.33:
        band        = "LOW RISK"
        band_colour = GREEN
        verdict     = "Your indicators suggest a relatively low risk of heart disease."
    elif prob < 0.66:
        band        = "MEDIUM RISK"
        band_colour = YELLOW
        verdict     = "Some risk factors are present.  Consult your doctor."
    else:
        band        = "HIGH RISK"
        band_colour = RED
        verdict     = "Multiple significant risk factors detected.  Please see a doctor."

    print(f"  Risk Band  :  {clr(band, band_colour + BOLD)}")
    print(f"  Risk Score :  {probability_bar(prob)}")
    print()
    print(f"  {verdict}")
    print()

    # ── Top contributing features ──────────────────────────────────────────
    # Random Forest exposes feature_importances_.  We multiply by the user's
    # actual (scaled) value to get a simple contribution signal:
    #   positive value → pushes risk UP
    #   negative value → pushes risk DOWN
    importances = model.feature_importances_
    user_vals   = feature_row.values[0]           # 1-D array of scaled values

    contributions = importances * user_vals        # element-wise product

    contrib_series = pd.Series(contributions, index=train_columns)

    top_pos = contrib_series.nlargest(5)           # top 5 risk-increasing factors
    top_neg = contrib_series.nsmallest(3)          # top 3 risk-reducing factors

    print(clr("  ── Factors increasing your risk ─────────────────────────", RED))
    for feat, score in top_pos.items():
        bar = clr("▮" * min(int(abs(score) * 200), 20), RED)
        print(f"    {feat:<20} {bar}")

    print()
    print(clr("  ── Factors working in your favour ───────────────────────", GREEN))
    for feat, score in top_neg.items():
        bar = clr("▮" * min(int(abs(score) * 200), 20), GREEN)
        print(f"    {feat:<20} {bar}")

    # ── General advice 
    print()
    print(clr("  ── General Advice ───────────────────────────────────────", CYAN))
    tips = []
    if prob >= 0.33:
        tips += [
            "  • Schedule a cardiology check-up.",
            "  • Monitor blood pressure and cholesterol regularly.",
        ]
    tips += [
        "  • Maintain a balanced diet low in saturated fats.",
        "  • Aim for 150 min of moderate exercise per week.",
        "  • Avoid smoking and limit alcohol intake.",
        "  • Manage stress through sleep and relaxation.",
    ]
    for tip in tips:
        print(tip)

    print()
    print(clr("  ⚠  This tool is for educational purposes only.", YELLOW))
    print(clr("     It is not a substitute for professional medical advice.", YELLOW))
    print()
    print(clr("=" * 62, CYAN))
    print()


# MODEL TRAINING
# We train a Random Forest on the saved final_unified_dataset.csv so the
# predictor always uses the same cleaned, integrated dataset.

def load_and_train(csv_path="final_unified_dataset.csv"):
    """
    Load the preprocessed dataset and train the Random Forest classifier.

    Returns
    -------
    model        : fitted RandomForestClassifier
    train_columns: list of feature column names (in training order)
    scaler       : fitted StandardScaler
    num_cols     : list of continuous column names
    """
    if not os.path.exists(csv_path):
        print(clr(f"\n  ✗  Cannot find '{csv_path}'.", RED))
        print("     Please run  heart_disease_project.py  first to generate it.\n")
        sys.exit(1)

    print(clr(f"  Loading dataset from {csv_path} …", DIM), end="", flush=True)
    df = pd.read_csv(csv_path)
    print(clr("  done.", GREEN))

    X = df.drop("target", axis=1)
    y = df["target"]

    # Identify continuous columns that need scaling.
    # These are the same columns standardised in the main pipeline.
    num_cols = ["age", "BMI", "sysBP", "diaBP",
                "cholesterol", "heartRate", "thalach", "oldpeak", "glucose"]
    # Keep only those actually present in the saved CSV
    num_cols = [c for c in num_cols if c in X.columns]

    # Load the scaler that was fitted on RAW data in index3.py.
    scaler_path = "scaler.pkl"
    if not os.path.exists(scaler_path):
        print(clr(f"\n  ✗  Cannot find '{scaler_path}'.", RED))
        print("     Please re-run  index3.py  first to regenerate it.\n")
        sys.exit(1)
    scaler = joblib.load(scaler_path)
    print(clr(f"  Loaded scaler from {scaler_path}.", GREEN))

    print(clr("  Training Random Forest …", DIM), end="", flush=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1          # use all CPU cores for speed
    )
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)
    print(clr(f"  done.  Test accuracy: {acc:.1%}", GREEN))

    train_columns = list(X.columns)
    return model, train_columns, scaler, num_cols



# MAIN LOOP
# Allows the user to run multiple predictions in one session.
def main():
    banner()

    # Train once; reuse the model for every prediction in this session.
    model, train_columns, scaler, num_cols = load_and_train()

    while True:
        print()
        print(clr("  Answer the questions below.", BOLD))
        print(clr("  (Press Ctrl+C at any time to exit.)\n", DIM))

        try:
            raw = collect_user_data()
        except KeyboardInterrupt:
            print("\n\n  Goodbye!\n")
            break

        # Build the feature vector and predict
        feature_row = build_feature_row(raw, train_columns, scaler, num_cols)
        prob        = model.predict_proba(feature_row)[0, 1]   # P(heart disease)

        display_result(prob, feature_row, train_columns, model)

        # Ask if the user wants to run another prediction
        again = input("  Run another prediction?  (y/n): ").strip().lower()
        if again not in ("y", "yes"):
            print(clr("\n  Take care of your heart!  Goodbye.\n", GREEN + BOLD))
            break


if __name__ == "__main__":
    main()
