import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = PROCESSED_DIR / "combined_news.csv"

csv_files = list(DATA_DIR.rglob("*.csv"))

# avoid merging old outputs
skip_files = {
    "combined_news.csv",
    "cleaned_news.csv",
    "old_combined.csv",
    "old_cleaned.csv",
}

csv_files = [f for f in csv_files if f.name not in skip_files]

print("CSV files found:")
for f in csv_files:
    print("-", f)

all_dfs = []

for file_path in csv_files:
    if file_path.stat().st_size == 0:
        print(f"Skipping empty file: {file_path.name}")
        continue

    name = file_path.name.lower()

    if "gdelt" in name:
        source_system = "GDELT"
    elif "world" in name:
        source_system = "WorldNewsAPI"
    elif "newsapi" in name or "news" in name:
        source_system = "NewsAPI"
    else:
        source_system = "Unknown"

    try:
        df = pd.read_csv(file_path)
    except pd.errors.EmptyDataError:
        print(f"Skipping unreadable empty file: {file_path.name}")
        continue
    except Exception as e:
        print(f"Skipping file due to error: {file_path.name} -> {e}")
        continue

    if df.empty:
        print(f"Skipping file with 0 rows: {file_path.name}")
        continue

    df["source_system"] = source_system

    required_cols = [
        "source_name", "title", "description", "url",
        "published_at", "fetched_at", "country", "category",
        "domain", "language", "social_image", "source_system"
    ]

    for col in required_cols:
        if col not in df.columns:
            df[col] = None

    df = df[required_cols]
    all_dfs.append(df)

if not all_dfs:
    raise ValueError("No valid non-empty CSV files found to merge.")

combined = pd.concat(all_dfs, ignore_index=True)

combined["title"] = combined["title"].fillna("").astype(str).str.strip()
combined["url"] = combined["url"].fillna("").astype(str).str.strip()

combined = combined[(combined["title"] != "") & (combined["url"] != "")]
combined = combined.drop_duplicates(subset=["url", "source_system"], keep="first")

combined.to_csv(OUTPUT_PATH, index=False)

print("\nSaved:", OUTPUT_PATH)
print("Shape:", combined.shape)
print("\nSource distribution:")
print(combined["source_system"].value_counts())