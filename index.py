#import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import SelectKBest, f_classif


# LOAD DATA
uci = pd.read_csv("uci_heart.csv")
fr = pd.read_csv("framingham.csv")
cardio = pd.read_csv("cardio.csv", sep=";")


# DATA INTEGRATION
# CARDIO 
cardio['age'] = cardio['age'] / 365 # because age was in days so here we are converting it to years
cardio['gender'] = cardio['gender'].map({1:0, 2:1}) #earlier gender was defined as 1:female and 2:male so we are converting it to 0:female and 1:male
cardio['BMI'] = cardio['weight'] / ((cardio['height']/100)**2) #calculating bmi from height and weight
cardio['cholesterol'] = cardio['cholesterol'].map({
    1: 180,
    2: 220,
    3: 260
}) #because it was given level wise so we have to convert into numeric format
cardio['gluc'] = cardio['gluc'].map({
    1: 85,
    2: 110,
    3: 140
})

cardio_new = pd.DataFrame({
    'age': cardio['age'],
    'gender': cardio['gender'],
    'BMI': cardio['BMI'],
    'sysBP': cardio['ap_hi'],
    'diaBP': cardio['ap_lo'],
    'cholesterol': cardio['cholesterol'],
    'glucose': cardio['gluc'],
    'smoke': cardio['smoke'],
    'alcohol': cardio['alco'],
    'active': cardio['active'],
    'BPMeds': np.nan,
    'stroke': np.nan,
    'hypertension': np.nan,
    'diabetes': np.nan,
    'heartRate': np.nan,
    'cp': np.nan,
    'restecg': np.nan,
    'thalach': np.nan,
    'exang': np.nan,
    'oldpeak': np.nan,
    'slope': np.nan,
    'ca': np.nan,
    'thal': np.nan,
    'target': cardio['cardio']
})

# FRAMINGHAM
fr_new = pd.DataFrame({
    'age': fr['age'],
    'gender': fr['male'],
    'BMI': fr['BMI'],
    'sysBP': fr['sysBP'],
    'diaBP': fr['diaBP'],
    'cholesterol': fr['totChol'],
    'glucose': fr['glucose'],
    'smoke': fr['currentSmoker'],
    'alcohol': np.nan,
    'active': np.nan,
    'BPMeds': fr['BPMeds'],
    'stroke': fr['prevalentStroke'],
    'hypertension': fr['prevalentHyp'],
    'diabetes': fr['diabetes'],
    'heartRate': fr['heartRate'],
    'cp': np.nan,
    'restecg': np.nan,
    'thalach': np.nan,
    'exang': np.nan,
    'oldpeak': np.nan,
    'slope': np.nan,
    'ca': np.nan,
    'thal': np.nan,
    'target': fr['TenYearCHD']
})

# UCI
uci['fbs'] = uci['fbs'].map({
    0: 90,
    1: 130
})
uci_new = pd.DataFrame({
    'age': uci['age'],
    'gender': uci['sex'],
    'BMI': np.nan,
    'sysBP': uci['trestbps'],
    'diaBP': np.nan,
    'cholesterol': uci['chol'],
    'glucose': uci['fbs'],
    'smoke': np.nan,
    'alcohol': np.nan,
    'active': np.nan,
    'BPMeds': np.nan,
    'stroke': np.nan,
    'hypertension': np.nan,
    'diabetes': np.nan,
    'heartRate': np.nan,
    'cp': uci['cp'],
    'restecg': uci['restecg'],
    'thalach': uci['thalach'],
    'exang': uci['exang'],
    'oldpeak': uci['oldpeak'],
    'slope': uci['slope'],
    'ca': uci['ca'],
    'thal': uci['thal'],
    'target': uci['target']
})
#We don’t rely on implicit NaN because it happens automatically during merging
#which can be unclear and harder to track.
#Explicit NaN is better because it clearly shows missing data from the beginning 
#and keeps the structure consistent and easy to understand



# COMBINE DATASETS
df = pd.concat([cardio_new, fr_new, uci_new], ignore_index=True)
#resets the index and creates a new continuous numbering after combining datasets.
#Index is a label or number used to uniquely identify each row in a dataset.


# CLEANING-Drop duplicates rows
df.drop_duplicates(inplace=True)


# MISSING VALUE HANDLING
# Numeric columns (continuous)
num_cols = ['age', 'BMI', 'sysBP', 'diaBP', 'heartRate',
            'thalach', 'oldpeak', 'cholesterol', 'glucose']
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
# Categorical / binary columns
cat_cols = ['gender', 'smoke', 'alcohol', 'active',
            'BPMeds', 'stroke', 'hypertension', 'diabetes',
            'exang', 'cp', 'restecg', 'slope', 'thal', 'ca']
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])
#Mean imputation is suitable only for continuous variables
#while categorical features should be imputed using mode to preserve data integrity and model accuracy


# OUTLIER REMOVAL-isloation forest method is used because we have high dimension data
iso = IsolationForest(contamination=0.05, random_state=42)
outliers = iso.fit_predict(df[num_cols])
df = df[outliers == 1]


# ENCODING (ONE HOT)
ohe_cols = ['cp', 'restecg', 'slope', 'thal']
df[ohe_cols] = df[ohe_cols].astype(int)
df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)
dummy_cols = [col for col in df.columns if df[col].dtype == bool]
df[dummy_cols] = df[dummy_cols].astype(int)

binary_cols = ['gender', 'smoke', 'alcohol', 'active',
               'BPMeds', 'stroke', 'hypertension', 'diabetes', 'exang', 'ca']
df[binary_cols] = df[binary_cols].astype(int)

# STANDARDIZATION
target = df['target']
num_cols = [
    'age','BMI','sysBP','diaBP',
    'cholesterol','heartRate',
    'thalach','oldpeak','glucose'
]
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
df_scaled = df.copy()
df_scaled['target'] = target.values


# FEATURE SELECTION
#selector = SelectKBest(score_func=f_classif, k=18) #here the value of k will determine how many features to be selected which will be done based on algo we choosed
#X_new = selector.fit_transform(df_scaled.drop('target', axis=1), df_scaled['target'])
#selected_cols = df_scaled.drop('target', axis=1).columns[selector.get_support()]
#final_df = pd.DataFrame(X_new, columns=selected_cols)
#final_df['target'] = df_scaled['target'].values
final_df = df_scaled.copy()


# SAVE FINAL DATASET
final_df.to_csv("final_unified_dataset.csv", index=False)
print("Final dataset created successfully!")
print(f"Shape: {final_df.shape}")
print(final_df.head())
print("UCI train:", Xu_train.shape, "test:", Xu_test.shape)


print("\nPreprocessing completed")

 
