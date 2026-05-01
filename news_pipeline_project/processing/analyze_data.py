import pandas as pd
import os

file_path = "data/combined_news.csv"

if not os.path.exists(file_path):
    raise FileNotFoundError(f"Missing file: {file_path}")

df = pd.read_csv(file_path)

print("Total records:", len(df))

# ---- Convert datetime ----
df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")

# ---- Articles per source ----
print("\nArticles by source:")
print(df["source_system"].value_counts())

# ---- Articles per day ----
print("\nArticles per day:")
print(df["published_at"].dt.date.value_counts().sort_index())

# ---- Top domains ----
if "domain" in df.columns:
    print("\nTop domains:")
    print(df["domain"].value_counts().head(10))

# ---- Top countries ----
if "country" in df.columns:
    print("\nTop countries:")
    print(df["country"].value_counts().head(10))

# ---- Missing summary ----
print("\nMissing values:")
print(df.isnull().sum())

# ---- Save summary (optional) ----
summary = {
    "total_records": len(df),
    "sources": df["source_system"].value_counts().to_dict()
}

print("\nSummary:", summary)