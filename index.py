import pandas as pd
print("Program started")
cardio = pd.read_csv("cardio.csv")
framingham = pd.read_csv("framingham.csv")
uci = pd.read_csv("uci_heart.csv")
print("Cardio Dataset")
print(cardio.head())
print("\nFramingham Dataset")
print(framingham.head())
print("\nUCI Heart Dataset")
print(uci.head())
print("\nCardio Shape:", cardio.shape)
print("Framingham Shape:", framingham.shape)
print("UCI Shape:", uci.shape)
print("\nMissing Values Cardio")
print(cardio.isnull().sum())
print("\nMissing Values Framingham")
print(framingham.isnull().sum())
print("\nMissing Values UCI")
print(uci.isnull().sum())

cardio.fillna(cardio.mean(numeric_only=True), inplace=True)
framingham.fillna(framingham.mean(numeric_only=True), inplace=True)
uci.fillna(uci.mean(numeric_only=True), inplace=True)

cardio.drop_duplicates(inplace=True)
framingham.drop_duplicates(inplace=True)
uci.drop_duplicates(inplace=True)

print("\nCardio Summary")
print(cardio.describe())

print("\nFramingham Summary")
print(framingham.describe())

print("\nUCI Summary")
print(uci.describe())

combined = pd.concat([cardio, framingham, uci], ignore_index=True)

print("\nCombined Dataset Shape:", combined.shape)
print("\nCombined Dataset Preview:")
print(combined.head())

