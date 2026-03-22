import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Page
st.set_page_config(page_title="Subscription Waste Detector", page_icon="💳")

st.title("💳 Subscription Waste Detector")

# Load dataset
df = pd.read_csv("subscription_waste_dataset.csv")
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Encode
le_type = LabelEncoder()
df['subscription_type'] = le_type.fit_transform(df['subscription_type'])

le_label = LabelEncoder()
df['label'] = le_label.fit_transform(df['label'])

# Train model
X = df.drop('label', axis=1)
y = df['label']

model = RandomForestClassifier()
model.fit(X, y)

# Inputs
st.sidebar.header("Enter Details")

cost = st.sidebar.number_input("Monthly Cost", 0, 5000, 500)
usage = st.sidebar.number_input("Usage per Month", 0, 50, 5)
session = st.sidebar.number_input("Avg Session Time", 0, 300, 30)
last_use = st.sidebar.number_input("Days Since Last Use", 0, 100, 10)
auto = st.sidebar.selectbox("Auto Renew", [0,1])
rating = st.sidebar.slider("Value Rating", 1, 5, 3)
sub_type = st.sidebar.selectbox(
    "Subscription Type",
    ["OTT","Gym","Software","Music","News"]
)

# Predict
if st.button("Predict"):
    sub_type_encoded = le_type.transform([sub_type])[0]

    input_df = pd.DataFrame([{
        'monthly_cost': cost,
        'usage_per_month': usage,
        'avg_session_time': session,
        'days_since_last_use': last_use,
        'auto_renew': auto,
        'value_rating': rating,
        'subscription_type': sub_type_encoded
    }])

    prediction = model.predict(input_df)
    result = le_label.inverse_transform(prediction)[0]

    st.subheader("Result")

    if result == "Keep":
        st.success("✅ KEEP")
    elif result == "Review":
        st.warning("⚠️ REVIEW")
    else:
        st.error("❌ CANCEL")
