import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

import seaborn as sns
import matplotlib.pyplot as plt

print("Program started")
 
cardio = pd.read_csv("cardio.csv", sep=";") 
framingham = pd.read_csv("framingham.csv")
uci = pd.read_csv("uci_heart.csv")

 
print("\nCardio Shape:", cardio.shape)
print("Framingham Shape:", framingham.shape)
print("UCI Shape:", uci.shape)

 
cardio.fillna(cardio.mean(numeric_only=True), inplace=True)
framingham.fillna(framingham.mean(numeric_only=True), inplace=True)
uci.fillna(uci.mean(numeric_only=True), inplace=True)
 
cardio.drop_duplicates(inplace=True)
framingham.drop_duplicates(inplace=True)
uci.drop_duplicates(inplace=True)
 
cardio["age_years"] = cardio["age"] / 365
cardio["BMI"] = cardio["weight"] / ((cardio["height"]/100)**2)

print("\nFeature engineering done")
 
def remove_outliers(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    
    df_clean = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
    return df_clean

cardio = remove_outliers(cardio)

print("After outlier removal shape:", cardio.shape)

 
print("\nEDA phase\n")

plt.figure(figsize=(10,8))
sns.heatmap(cardio.corr(), cmap="coolwarm")
plt.title("Correlation heatmap")
plt.show()

cardio.hist(figsize=(12,10))
plt.suptitle("Feature distributions")
plt.show()

sns.countplot(x="cardio", data=cardio)
plt.title("Target distribution")
plt.show()
 
def encode_data(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    return df

cardio = encode_data(cardio)
framingham = encode_data(framingham)
uci = encode_data(uci)
 
def select_features(df, target, threshold=0.1):
    corr = df.corr()[target].abs().sort_values(ascending=False)

    print(f"\nCorrelation with {target}:\n", corr)

    selected = corr[corr > threshold].index.tolist()

    if target in selected:
        selected.remove(target)

    print(f"\nSelected features for {target}: {selected}")

    X = df[selected]
    y = df[target]

    return X, y

X_cardio, y_cardio = select_features(cardio, "cardio")
X_fram, y_fram = select_features(framingham, "TenYearCHD")
X_uci, y_uci = select_features(uci, "target")

 
scaler = StandardScaler()

X_cardio_scaled = pd.DataFrame(
    scaler.fit_transform(X_cardio),
    columns=X_cardio.columns
)

X_fram_scaled = pd.DataFrame(
    scaler.fit_transform(X_fram),
    columns=X_fram.columns
)

X_uci_scaled = pd.DataFrame(
    scaler.fit_transform(X_uci),
    columns=X_uci.columns
)

 
print("\nTrain test split\n")

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_cardio_scaled, y_cardio,
    test_size=0.2,
    random_state=42,
    stratify=y_cardio
)

Xf_train, Xf_test, yf_train, yf_test = train_test_split(
    X_fram_scaled, y_fram,
    test_size=0.2,
    random_state=42,
    stratify=y_fram
)

Xu_train, Xu_test, yu_train, yu_test = train_test_split(
    X_uci_scaled, y_uci,
    test_size=0.2,
    random_state=42,
    stratify=y_uci
)

print("Cardio train:", Xc_train.shape, "test:", Xc_test.shape)
print("Framingham train:", Xf_train.shape, "test:", Xf_test.shape)
print("UCI train:", Xu_train.shape, "test:", Xu_test.shape)

print("\nPreprocessing completed")
