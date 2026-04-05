import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# LOAD DATA 
df = pd.read_csv("subscription_waste_dataset.csv")
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
# PREDICTION FUNCTION 
def predict_subscription(cost, usage, session, last_use, auto, rating, sub_type):
    sub_type_encoded = le_type.transform([sub_type])[0]
    input_data = np.array([[cost, usage, session, last_use, auto, rating, sub_type_encoded]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    return le_label.inverse_transform(prediction)[0]

#  STREAMLIT UI 
st.set_page_config(
    page_title="Subscription Waste Detector",
    page_icon="💳",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Sidebar
st.sidebar.header("📊 About App")
st.sidebar.info(
    """
    This app predicts whether you should **Keep, Review, or Cancel** a subscription.
    \n
    Enter your subscription details below to get instant suggestions.
    """
)
st.sidebar.header("🔍 Features Used")
st.sidebar.write("""
- Monthly Cost  
- Usage per Month  
- Avg Session Time  
- Days Since Last Use  
- Auto Renew  
- Value Rating  
- Subscription Type
""")

# Main Title
st.markdown("<h1 style='text-align: center; color: #4B0082;'>💳 Subscription Waste Detector</h1>", unsafe_allow_html=True)
st.markdown("---")

# Input form in columns
with st.form("subscription_form"):
    col1, col2 = st.columns(2)

    with col1:
        cost = st.number_input("💰 Monthly Cost", min_value=0.0, step=1.0, format="%.2f")
        usage = st.number_input("📈 Usage per Month", min_value=0.0, step=1.0)
        session = st.number_input("⏱ Avg Session Time (minutes)", min_value=0.0, step=1.0)
        last_use = st.number_input("📅 Days Since Last Use", min_value=0, step=1)

    with col2:
        auto = st.radio("🔄 Auto Renew", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        rating = st.slider("⭐ Value Rating", 1, 5)
        sub_type = st.selectbox("📂 Subscription Type", ["OTT","Gym","Software","Music","News"])

    submitted = st.form_submit_button("Predict")

# Prediction Result
if submitted:
    result = predict_subscription(cost, usage, session, last_use, auto, rating, sub_type)
    st.markdown(f"""
        <div style='padding: 20px; border-radius: 10px; background-color: #4B0082; color: white; text-align: center; font-size: 24px;'>
        🏷 Prediction: <strong>{result}</strong>
        </div>
    """, unsafe_allow_html=True)
