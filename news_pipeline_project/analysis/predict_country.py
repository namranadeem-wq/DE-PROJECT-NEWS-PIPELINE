import os
import json
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

os.makedirs("analysis_outputs", exist_ok=True)

print("Loading cleaned data...")

df = pd.read_csv("data/cleaned_news.csv")

df = df.dropna(subset=["title", "country"])
df["title"] = df["title"].astype(str)
df["country"] = df["country"].astype(str).str.strip()

# Remove noisy labels
df = df[~df["country"].isin(["Unknown", "Other"])]

print("\nCountry counts before filtering:")
print(df["country"].value_counts())

# Keep only countries with at least 5 examples
counts = df["country"].value_counts()
valid_classes = counts[counts >= 5].index
df = df[df["country"].isin(valid_classes)]

print("\nCountry counts used for model:")
print(df["country"].value_counts())
print("Final ML shape:", df.shape)

X = df["title"]
y = df["country"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95
    )),
    ("clf", LinearSVC(class_weight="balanced", C=0.5))
])

print("\nTraining model...")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

metrics = {
    "accuracy": accuracy,
    "train_size": len(X_train),
    "test_size": len(X_test),
    "classes": sorted(y.unique().tolist())
}

with open("analysis_outputs/prediction_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

report_df = pd.DataFrame(
    classification_report(y_test, y_pred, output_dict=True, zero_division=0)
).transpose()

report_df.to_csv("analysis_outputs/classification_report.csv")

predictions_df = pd.DataFrame({
    "title": X_test.values,
    "actual_country": y_test.values,
    "predicted_country": y_pred
})
predictions_df.to_csv("analysis_outputs/test_predictions.csv", index=False)

labels = sorted(y.unique().tolist())
cm = confusion_matrix(y_test, y_pred, labels=labels)

fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
plt.title("Confusion Matrix - Country Prediction")
plt.tight_layout()
plt.savefig("analysis_outputs/confusion_matrix.png")
plt.close()

sample_texts = [
    "Missile strike in Gaza escalates tensions",
    "US officials discuss new sanctions on Iran",
    "India and China hold border talks",
    "Pakistan responds to regional security concerns",
    "Turkey announces diplomatic talks"
]

sample_preds = model.predict(sample_texts)

sample_df = pd.DataFrame({
    "text": sample_texts,
    "prediction": sample_preds
})
sample_df.to_csv("analysis_outputs/sample_predictions.csv", index=False)

print("\nSaved ML outputs:")
print("analysis_outputs/prediction_metrics.json")
print("analysis_outputs/classification_report.csv")
print("analysis_outputs/test_predictions.csv")
print("analysis_outputs/confusion_matrix.png")
print("analysis_outputs/sample_predictions.csv")