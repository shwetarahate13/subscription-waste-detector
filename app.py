import streamlit as st
import numpy as np
import pickle

# LOAD FILES
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
le_label = pickle.load(open("label_encoder.pkl", "rb"))
le_type = pickle.load(open("type_encoder.pkl", "rb"))

st.title("Subscription Decision Predictor")

# INPUTS
cost = st.number_input("Monthly Cost")
usage = st.number_input("Usage Per Month")
session = st.number_input("Sessions Per Month")
last_use = st.number_input("Days Since Last Use")
auto = st.selectbox("Auto Renewal", [0, 1])
rating = st.slider("User Rating", 1, 5)

sub_type = st.selectbox("Subscription Type", list(le_type.classes_))

# PREDICT BUTTON
if st.button("Predict"):

    if sub_type not in le_type.classes_:
        st.error("Invalid subscription type")
    else:
        sub_type_encoded = le_type.transform([sub_type])[0]

        input_data = np.array([[cost, usage, session, last_use, auto, rating, sub_type_encoded]])
        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)

        result = le_label.inverse_transform(prediction)[0]

        st.success(f"Prediction: {result}")
