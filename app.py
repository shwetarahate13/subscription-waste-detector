import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page config
st.set_page_config(page_title="Subscription Waste Detector", page_icon="💳")

st.title("💳 Subscription Waste Detector")
st.markdown("### Predict whether to Keep, Review, or Cancel a subscription")

# Load model + encoders + feature order
model = joblib.load("model.pkl")
le_type = joblib.load("type_encoder.pkl")
le_label = joblib.load("label_encoder.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# Sidebar inputs
st.sidebar.header("📥 Enter Subscription Details")

cost = st.sidebar.number_input("Monthly Cost", 0, 5000, 500)
usage = st.sidebar.number_input("Usage per Month", 0, 50, 5)
session = st.sidebar.number_input("Avg Session Time", 0, 300, 30)
last_use = st.sidebar.number_input("Days Since Last Use", 0, 100, 10)
auto = st.sidebar.selectbox("Auto Renew", [0, 1])
rating = st.sidebar.slider("Value Rating", 1, 5, 3)
sub_type = st.sidebar.selectbox(
    "Subscription Type",
    ["OTT", "Gym", "Software", "Music", "News"]
)

# Predict button
if st.button("🔍 Predict"):
    try:
        # Encode subscription type
        sub_type_encoded = le_type.transform([sub_type])[0]

        # Create input dataframe EXACTLY matching training columns
        input_dict = {
            'monthly_cost': cost,
            'usage_per_month': usage,
            'avg_session_time': session,
            'days_since_last_use': last_use,
            'auto_renew': auto,
            'value_rating': rating,
            'subscription_type': sub_type_encoded
        }

        input_df = pd.DataFrame([input_dict])

        # Ensure correct column order
        input_df = input_df[feature_columns]

        # Predict
        pred = model.predict(input_df)
        result = le_label.inverse_transform(pred)[0]

        st.subheader("🎯 Result")

        if result == "Keep":
            st.success("✅ KEEP this subscription")
        elif result == "Review":
            st.warning("⚠️ REVIEW this subscription")
        else:
            st.error("❌ CANCEL this subscription")

    except Exception as e:
        st.error(f"Error: {e}")

