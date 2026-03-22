import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


df = pd.read_csv("subscription_waste_dataset.csv")


df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

print("Columns in dataset:", df.columns)


le_type = LabelEncoder()
df['subscription_type'] = le_type.fit_transform(df['subscription_type'])

le_label = LabelEncoder()
df['label'] = le_label.fit_transform(df['label'])


X = df.drop('label', axis=1)
y = df['label']

print("Feature columns used:", X.columns)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n✅ Model trained successfully! Accuracy: {acc:.2f}")


joblib.dump(model, "model.pkl")
joblib.dump(le_type, "type_encoder.pkl")
joblib.dump(le_label, "label_encoder.pkl")
joblib.dump(list(X.columns), "feature_columns.pkl")

print("✅ Model + encoders + feature columns saved!")
