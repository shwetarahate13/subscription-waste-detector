import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Subscription Waste Detector", page_icon="💳", layout="wide")


st.markdown("<h1 style='text-align: center;'>💳 Subscription Waste Detector</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>AI-powered decision for your subscriptions</h4>", unsafe_allow_html=True)


df = pd.read_csv("subscription_waste_dataset.csv")
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]


le_type = LabelEncoder()
df['subscription_type'] = le_type.fit_transform(df['subscription_type'])

le_label = LabelEncoder()
df['label'] = le_label.fit_transform(df['label'])


X = df.drop('label', axis=1)
y = df['label']

model = RandomForestClassifier()
model.fit(X, y)


col1, col2 = st.columns([1,1])


with col1:
    st.subheader("📥 Enter Subscription Details")

    cost = st.slider("Monthly Cost", 100, 2000, 500)
    usage = st.slider("Usage per Month", 0, 30, 5)
    session = st.slider("Avg Session Time", 5, 120, 30)
    last_use = st.slider("Days Since Last Use", 0, 60, 10)
    auto = st.radio("Auto Renew", [0,1])
    rating = st.slider("Value Rating", 1, 5, 3)
    sub_type = st.selectbox(
        "Subscription Type",
        ["OTT","Gym","Software","Music","News"]
    )

    if st.button("🔍 Predict Decision"):
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

        st.subheader("🎯 Result")

        if result == "Keep":
            st.success("✅ KEEP this subscription")
        elif result == "Review":
            st.warning("⚠️ REVIEW this subscription")
        else:
            st.error("❌ CANCEL this subscription")

with col2:
    st.subheader("📊 Insights & Analysis")

    fig1, ax1 = plt.subplots()
    sns.countplot(x='label', data=df, ax=ax1)
    ax1.set_title("Subscription Decisions Distribution")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    sns.scatterplot(x='monthly_cost', y='usage_per_month', hue='label', data=df, ax=ax2)
    ax2.set_title("Cost vs Usage")
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots()
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax3)
    ax3.set_title("Correlation Heatmap")
    st.pyplot(fig3)


st.markdown("---")
st.markdown("<p style='text-align: center;'>Built with ❤️ using Streamlit</p>", unsafe_allow_html=True)

    
