import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ================== LOAD DATA ==================
df = pd.read_csv("subscription_waste_dataset.csv")

# Remove unwanted columns
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Encode
le_type = LabelEncoder()
df['subscription_type'] = le_type.fit_transform(df['subscription_type'])

le_label = LabelEncoder()
df['label'] = le_label.fit_transform(df['label'])

# Features & target
X = df.drop('label', axis=1)
y = df['label']

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = DecisionTreeClassifier(max_depth=8, min_samples_split=5)
model.fit(X_scaled, y)

# ================== PREDICTION FUNCTION ==================
def predict_subscription(cost, usage, session, last_use, auto, rating, sub_type):
    sub_type_encoded = le_type.transform([sub_type])[0]

    input_data = np.array([[cost, usage, session,
                            last_use, auto, rating,
                            sub_type_encoded]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    return le_label.inverse_transform(prediction)[0]

# ================== UI ==================
st.title("💳 Subscription Waste Detector")

cost = st.number_input("Monthly Cost")
usage = st.number_input("Usage per Month")
session = st.number_input("Avg Session Time")
last_use = st.number_input("Days Since Last Use")
auto = st.radio("Auto Renew", [0,1])
rating = st.slider("Value Rating", 1, 5)
sub_type = st.selectbox("Subscription Type", ["OTT","Gym","Software","Music","News"])

if st.button("Predict"):
    result = predict_subscription(cost, usage, session, last_use, auto, rating, sub_type)
    st.success(f"Prediction: {result}")
