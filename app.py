import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Loading the trained model

model = tf.keras.models.load_model('model.h5')

# load the encoders :

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)

with open("one_hot_encoder_geo.pkl", "rb") as file:
    one_hot_encoder_geo = pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)

import streamlit as st
import pandas as pd

# streamlit app:S

# -------------------------------------------------
# Page configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="centered"
)

st.title(" Customer Churn Prediction")
st.caption("Predict whether a customer is likely to churn using machine learning")
st.divider()

# -------------------------------------------------
# Input Form
# -------------------------------------------------
with st.form("churn_form"):
    st.subheader(" Customer Information")

    col1, col2 = st.columns(2)

    with col1:
        geography = st.selectbox(
            " Geography",
            one_hot_encoder_geo.categories_[0]
        )
        gender = st.selectbox(
            "⚧ Gender",
            label_encoder_gender.classes_
        )
        age = st.slider(" Age", 18, 92)
        tenure = st.slider(" Tenure (years)", 0, 10)

    with col2:
        credit_score = st.number_input(
            " Credit Score",
            min_value=300,
            max_value=900,
            value=650
        )
        balance = st.number_input(
            " Balance",
            min_value=0.0,
            step=1000.0
        )
        estimated_salary = st.number_input(
            " Estimated Salary",
            min_value=0.0,
            step=1000.0
        )
        num_of_products = st.slider(
            " Number of Products",
            1,
            4
        )

    st.subheader(" Account Status")

    col3, col4 = st.columns(2)

    with col3:
        has_cr_card = st.selectbox(
            " Has Credit Card",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

    with col4:
        is_active_member = st.selectbox(
            " Active Member",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

    submit = st.form_submit_button("🔍 Predict Churn")

# -------------------------------------------------
# Prediction Logic
# -------------------------------------------------
if submit:

    # Prepare input data
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    # One-hot encode Geography
    geo_encoded = one_hot_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(
        geo_encoded,
        columns=one_hot_encoder_geo.get_feature_names_out(['Geography'])
    )

    # Combine all features
    input_data = pd.concat(
        [input_data.reset_index(drop=True), geo_encoded_df],
        axis=1
    )

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Predict churn
    prediction = model.predict(input_data_scaled)
    prediction_proba = prediction[0][0]

    st.divider()
    st.subheader(" Prediction Result")

    # Display probability
    st.metric(
        label="Churn Probability",
        value=f"{prediction_proba:.2%}"
    )

    # Probability bar
    st.progress(float(prediction_proba))

    # Interpretation
    if prediction_proba > 0.5:
        st.error(" High risk of churn")
    else:
        st.success(" Customer is likely to stay")
