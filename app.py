import streamlit as st
import numpy as np
import pickle

# ================== LOAD FILES ==================
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("le_type.pkl", "rb") as f:
    le_type = pickle.load(f)

with open("le_label.pkl", "rb") as f:
    le_label = pickle.load(f)

# ================== PREDICTION FUNCTION ==================
def predict_subscription(cost, usage, session, last_use, auto, rating, sub_type):
    try:
        sub_type_encoded = le_type.transform([sub_type])[0]

        input_data = np.array([[cost, usage, session,
                                last_use, auto, rating,
                                sub_type_encoded]])

        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)
        result = le_label.inverse_transform(prediction)[0]

        return result

    except Exception as e:
        return str(e)

# ================== UI ==================
st.set_page_config(page_title="Subscription Waste Detector")

st.title("💳 Subscription Waste Detector")
st.write("Predict whether to Keep, Review, or Cancel a subscription")

cost = st.number_input("Monthly Cost", min_value=0.0)
usage = st.number_input("Usage per Month", min_value=0.0)
session = st.number_input("Avg Session Time", min_value=0.0)
last_use = st.number_input("Days Since Last Use", min_value=0.0)
auto = st.radio("Auto Renew", [0,1])
rating = st.slider("Value Rating", 1, 5)
sub_type = st.selectbox("Subscription Type", ["OTT","Gym","Software","Music","News"])

if st.button("Predict"):
    with st.spinner("Analyzing..."):
        result = predict_subscription(cost, usage, session, last_use, auto, rating, sub_type)

    if result == "Keep":
        st.success(f"✅ Recommendation: {result}")
    elif result == "Review":
        st.warning(f"⚠️ Recommendation: {result}")
    else:
        st.error(f"❌ Recommendation: {result}")
