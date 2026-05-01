import pandas as pd
import os

file_path = "data/combined_news.csv"

if not os.path.exists(file_path):
    raise FileNotFoundError(f"Missing file: {file_path}")

df = pd.read_csv(file_path)

print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

if "url" in df.columns:
    print("\nDuplicate URLs:", df.duplicated(subset=["url"]).sum())

if "published_at" in df.columns:
    published = pd.to_datetime(df["published_at"], errors="coerce")
    print("Invalid published_at values:", published.isna().sum())

if "title" in df.columns:
    empty_titles = df["title"].isna().sum() + (df["title"].astype(str).str.strip() == "").sum()
    print("Empty title rows:", empty_titles)

if "source_system" in df.columns:
    print("\nRows by source_system:")
    print(df["source_system"].value_counts(dropna=False))