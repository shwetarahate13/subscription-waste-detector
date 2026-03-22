import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("subscription_waste_dataset.csv")

# Remove unwanted column (if exists)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

print("Dataset Loaded:")
print(df.head())

# Encode categorical columns
le_type = LabelEncoder()
df['subscription_type'] = le_type.fit_transform(df['subscription_type'])

le_label = LabelEncoder()
df['label'] = le_label.fit_transform(df['label'])

# Split features & target
X = df.drop('label', axis=1)
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\nModel trained successfully!")
print("Accuracy:", accuracy)

# Save model (IMPORTANT)
import joblib
joblib.dump(model, "model.pkl")

# Save encoders
joblib.dump(le_type, "type_encoder.pkl")
joblib.dump(le_label, "label_encoder.pkl")

print("\nModel and encoders saved!")
