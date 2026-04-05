# ================== IMPORTS ==================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_curve, auc
)
from sklearn.svm import SVC
from sklearn.preprocessing import label_binarize

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# ================== 1. LOAD DATA ==================
df = pd.read_csv("/content/drive/MyDrive/subscription_waste_dataset.csv")

print("\n===== DATA HEAD =====")
print(df.head())

print("\n===== DATA INFO =====")
print(df.info())

# Remove unwanted columns
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# ================== 2. EDA ==================
sns.countplot(x='label', data=df)
plt.title("Subscription Decision Distribution")
plt.show()

sns.scatterplot(x='monthly_cost', y='usage_per_month', hue='label', data=df)
plt.title("Cost vs Usage")
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()




# ================== 3. PREPROCESSING ==================
le_type = LabelEncoder()
df['subscription_type'] = le_type.fit_transform(df['subscription_type'])

le_label = LabelEncoder()
df['label'] = le_label.fit_transform(df['label'])

X = df.drop('label', axis=1)
y = df['label']

# Scaling (important for LR)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n===== PREPROCESSED DATA SHAPE =====")
print(X_scaled.shape)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42,stratify=y
)

# ================== 4. TRAIN MULTIPLE MODELS ==================
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=8,
        min_samples_split=5
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=10,class_weight='balanced'
    ),

    "SVM": SVC(
        C=1,
        kernel='rbf',
        probability=True
    )
}



results = {}
model_predictions = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    model_predictions[name] = y_pred

    print(f"\n===== {name} =====")
    print("Accuracy:", acc)
    print(classification_report(y_test, y_pred))

# ================== 5. MODEL COMPARISON TABLE ==================
comparison_df = pd.DataFrame({
    "Model": results.keys(),
    "Accuracy": results.values()
})

print("\n===== MODEL COMPARISON TABLE =====")
print(comparison_df)

plt.figure()
sns.barplot(x="Model", y="Accuracy", data=comparison_df)
plt.title("Model Accuracy Comparison")
plt.xticks(rotation=30)
plt.show()

# ================== 6. CONFUSION MATRIX ==================
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

y_pred_best = model_predictions[best_model_name]

cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ================== 7. ROC CURVE ==================
# Convert labels to binary format
y_test_bin = label_binarize(y_test, classes=[0,1,2])

# Use probabilities
y_score = best_model.predict_proba(X_test)

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC
plt.figure()
for i in range(3):
    plt.plot(fpr[i], tpr[i], label=f"Class {i} (AUC = {roc_auc[i]:.2f})")

plt.plot([0,1], [0,1], 'k--')
plt.title("ROC Curve (Multiclass)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

# ================== 8. BEST MODEL SELECTION ==================
print("\n===== BEST MODEL =====")
print("Best Model:", best_model_name)
print("Accuracy:", results[best_model_name])

# ================== FEATURE IMPORTANCE ==================
if best_model_name == "Random Forest":
    importance = best_model.feature_importances_
    features = X.columns

    plt.figure(figsize=(8,5))
    sns.barplot(x=importance, y=features)
    plt.title("Feature Importance")
    plt.show()

# ================== 9. PREDICTION FUNCTION ==================
def predict_subscription(cost, usage, session, last_use, auto, rating, sub_type):
    try:
        sub_type_encoded = le_type.transform([sub_type])[0]

        input_data = np.array([[cost, usage, session,
                                last_use, auto, rating,
                                sub_type_encoded]])

        input_scaled = scaler.transform(input_data)

        prediction = best_model.predict(input_scaled)

        return le_label.inverse_transform(prediction)[0]

    except Exception as e:
        return str(e)

# ================== 10. GUI ==================
import gradio as gr

interface = gr.Interface(
    fn=predict_subscription,
    inputs=[
        gr.Number(label="Monthly Cost"),
        gr.Number(label="Usage per Month"),
        gr.Number(label="Avg Session Time"),
        gr.Number(label="Days Since Last Use"),
        gr.Radio([0,1], label="Auto Renew (0=No, 1=Yes)"),
        gr.Slider(1,5,label="Value Rating"),
        gr.Dropdown(["OTT","Gym","Software","Music","News"], label="Subscription Type")
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="💳 Subscription Waste Detector",
    description="Predict whether to Keep, Review, or Cancel a subscription"
)

interface.launch()
