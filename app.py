import os
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils.preprocessing import basic_preprocessing, encode_for_xgb_mlp, encode_for_tabnet
from utils.training import train_and_save_xgb, train_and_save_tabnet_with_logs, train_tfmlp_with_logs
from utils.evaluation import evaluate_model, plot_metrics

os.makedirs('models', exist_ok=True)

@st.cache_data
def load_data():
    return pd.read_csv('data/ca_san_francisco_2020_04_01.csv', low_memory=False)

st.set_page_config(page_title="🚦 Police Stops Model Explorer", layout="wide")

st.title("🚔 Police Stops Predictive Modeling")
st.markdown("Compare XGBoost, TabNet, and Keras MLP on San Francisco police stop data.", unsafe_allow_html=True)

df = load_data()
target = 'arrest_made'

with st.expander("Show raw data sample"):
    st.dataframe(df.head())

X, y = basic_preprocessing(df, target)

X_xgb_mlp = encode_for_xgb_mlp(X)
X_tabnet, cat_idxs, cat_dims = encode_for_tabnet(X)

# IMPORTANT: Use oversampled data ONLY for TabNet training; splitting adjusted below:

from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=42)
X_tabnet_bal, y_bal = ros.fit_resample(X_tabnet, y)

# Use balanced data for TabNet train/test split
X_train_tab, X_test_tab, y_train_tab, y_test_tab = train_test_split(X_tabnet_bal, y_bal, test_size=0.2, random_state=42)
# Use original data for XGBoost and MLP train/test split
X_train_xgb, X_test_xgb, y_train, y_test = train_test_split(X_xgb_mlp, y, test_size=0.2, random_state=42)

# TensorFlow MLP scaling (subject_age at column 0)
scaler = StandardScaler()
X_train_mlp = X_train_xgb.copy()
X_test_mlp = X_test_xgb.copy()
X_train_mlp[:, 0] = scaler.fit_transform(X_train_mlp[:, [0]]).flatten()
X_test_mlp[:, 0] = scaler.transform(X_test_mlp[:, [0]]).flatten()

# Save the scaler to use during prediction
joblib.dump(scaler, 'models/subject_age_scaler.pkl')

st.sidebar.header("🔍 Choose Models & Epochs")
use_xgb = st.sidebar.checkbox("XGBoost", True)
use_tabnet = st.sidebar.checkbox("TabNet", True)
use_tfmlp = st.sidebar.checkbox("Keras MLP", True)
max_epochs = st.sidebar.slider("TabNet Max Epochs", 10, 100, 30)

results = {}

col1, col2, col3 = st.columns(3)

if use_xgb:
    with col1:
        st.markdown("### 🌟 XGBoost")
        model_xgb = train_and_save_xgb(X_train_xgb, y_train)
        acc, roc = evaluate_model(model_xgb, X_test_xgb, y_test, 'xgb')
        st.metric(label="Accuracy", value=f"{acc:.3f}")
        st.metric(label="ROC AUC", value=f"{roc:.3f}")
        results['XGBoost'] = (acc, roc)

if use_tabnet:
    with col2:
        st.markdown("### 🚀 TabNet")
        model_tabnet, tabnet_logs = train_and_save_tabnet_with_logs(
            X_train_tab, y_train_tab, cat_idxs, cat_dims, max_epochs
        )
        acc, roc = evaluate_model(model_tabnet, X_test_tab, y_test_tab, 'tabnet')
        st.metric(label="Accuracy", value=f"{acc:.3f}")
        st.metric(label="ROC AUC", value=f"{roc:.3f}")
        results['TabNet'] = (acc, roc)
        with st.expander("Show TabNet Training Logs"):
            st.text_area("", tabnet_logs, height=300)

if use_tfmlp:
    with col3:
        st.markdown("### 🤖 MLP")
        model_tfmlp, training_logs = train_tfmlp_with_logs(
            X_train_mlp, y_train, X_test_mlp, y_test, X_train_mlp.shape[1]
        )
        acc, roc = evaluate_model(model_tfmlp, X_test_mlp, y_test, 'tfmlp')
        st.metric(label="Accuracy", value=f"{acc:.3f}")
        st.metric(label="ROC AUC", value=f"{roc:.3f}")
        results['MLP (TF)'] = (acc, roc)
        with st.expander("Show MLP Training Logs"):
            st.text_area("", training_logs, height=300)

if results:
    st.divider()
    st.markdown("## 📊 Model Comparison Overview")
    plot_metrics(results)
