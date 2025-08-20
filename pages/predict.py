import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from pytorch_tabnet.tab_model import TabNetClassifier  # Add this import

st.title("🚦 Predict Police Stop Arrest")

@st.cache_resource
def load_xgb():
    return joblib.load('models/xgb_model.pkl')

@st.cache_resource
def load_mlp():
    return load_model('models/tfmlp_model.keras')

@st.cache_resource
def load_tabnet():
    model = TabNetClassifier()
    model.load_model('models/tabnet_model.zip')
    return model

@st.cache_resource
def load_scaler():
    return joblib.load('models/subject_age_scaler.pkl')

scaler = load_scaler()

race_map = {'Unknown': 0, 'White': 1, 'Black': 2, 'Asian': 3, 'Latino': 4, 'Other': 5}
sex_map = {'Unknown': 0, 'Male': 1, 'Female': 2}
district_map = {'Unknown': 0, 'District1': 1, 'District2': 2, 'District3': 3}
reason_map = {'Unknown': 0, 'Suspicious Behavior': 1, 'Traffic Violation': 2, 'Other': 3}
search_conducted_map = {'No': 0, 'Yes': 1}

with st.form("predict_form"):
    subject_age = st.number_input("Subject Age", 0, 100, 30)
    subject_race = st.selectbox("Subject Race", list(race_map.keys()))
    subject_sex = st.selectbox("Subject Sex", list(sex_map.keys()))
    district = st.selectbox("District", list(district_map.keys()))
    reason_for_stop = st.selectbox("Reason for Stop", list(reason_map.keys()))
    search_conducted = st.selectbox("Search Conducted?", list(search_conducted_map.keys()))

    model_type = st.selectbox("Choose model:", ["XGBoost", "TabNet", "MLP (TF)"])
    submit = st.form_submit_button("Predict")

X_new = np.array([[
    subject_age,
    race_map.get(subject_race, 0),
    sex_map.get(subject_sex, 0),
    district_map.get(district, 0),
    reason_map.get(reason_for_stop, 0),
    search_conducted_map.get(search_conducted, 0),
]])

# Apply scaling ONLY when model is MLP (TensorFlow)
if submit:
    try:
        if model_type == "MLP (TF)":
            X_new[:, 0] = scaler.transform(X_new[:, [0]]).flatten()

        if model_type == "XGBoost":
            model = load_xgb()
            y_pred_prob = model.predict_proba(X_new)[0, 1]
        elif model_type == "MLP (TF)":
            model = load_mlp()
            y_pred_prob = model.predict(X_new)[0, 0]
        elif model_type == "TabNet":
            model = load_tabnet()
            y_pred_prob = model.predict_proba(X_new)[0, 1]
        else:
            y_pred_prob = float('nan')

        st.subheader(f"Probability of Arrest: {y_pred_prob:.1%}")
        st.success("Prediction: YES" if y_pred_prob > 0.5 else "Prediction: NO")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.info("Ensure all categorical encodings match the training data exactly.")
