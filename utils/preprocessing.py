import pandas as pd
from sklearn.preprocessing import LabelEncoder

def basic_preprocessing(df, target):
    df['subject_age'] = df['subject_age'].fillna(df['subject_age'].median())
    df['district'] = df['district'].fillna('Unknown')
    df['subject_race'] = df['subject_race'].fillna('Unknown')
    df['subject_sex'] = df['subject_sex'].fillna('Unknown')
    df['reason_for_stop'] = df['reason_for_stop'].fillna('Unknown')
    df['search_conducted'] = df['search_conducted'].fillna(False)

    # Convert search_conducted to bool if needed
    if df['search_conducted'].dtype != bool:
        df['search_conducted'] = df['search_conducted'].astype(str).str.lower().map({
            'true': True, 'yes': True, '1': True,
            'false': False, 'no': False, '0': False
        }).fillna(False)

    df = df[df[target].notnull()]
    y = df[target].astype(int).values
    features = ['subject_age', 'subject_race', 'subject_sex', 'district', 'reason_for_stop', 'search_conducted']
    X = df[features].copy()
    return X, y

def encode_for_xgb_mlp(X):
    X_enc = X.copy()
    for col in X_enc.select_dtypes(include='object').columns:
        le = LabelEncoder()
        X_enc[col] = le.fit_transform(X_enc[col])
    if 'search_conducted' in X_enc.columns and X_enc['search_conducted'].dtype == bool:
        X_enc['search_conducted'] = X_enc['search_conducted'].astype(int)
    return X_enc.values

def encode_for_tabnet(X):
    X_enc = X.copy()
    cat_idxs, cat_dims = [], []
    for i, col in enumerate(X_enc.columns):
        if X_enc[col].dtype == 'object' or X_enc[col].dtype == 'bool':
            le = LabelEncoder()
            X_enc[col] = le.fit_transform(X_enc[col])
            cat_idxs.append(i)
            cat_dims.append(len(le.classes_))
    return X_enc.values, cat_idxs, cat_dims
