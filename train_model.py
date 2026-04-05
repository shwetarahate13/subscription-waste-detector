import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
import pickle

# Load dataset 
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

# ✅ Best Model: Decision Tree
model = DecisionTreeClassifier(
    max_depth=8,
    min_samples_split=5,
    random_state=42
)

model.fit(X_scaled, y)

# Save files
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("le_type.pkl", "wb") as f:
    pickle.dump(le_type, f)

with open("le_label.pkl", "wb") as f:
    pickle.dump(le_label, f)

print("✅ Model trained & saved")
