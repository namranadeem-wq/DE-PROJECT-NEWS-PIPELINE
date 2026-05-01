import os
import pandas as pd

# -----------------------------
# Paths
# -----------------------------
COMBINED_PATH = "data/combined_news.csv"

SOURCE_FILES = [
    ("data/processed_news.csv", "NewsAPI"),
    ("data/processed_gdelt_news.csv", "GDELT"),
    ("data/processed_worldnews.csv", "WorldNewsAPI"),
]

print("===== MERGE STARTED =====")

# -----------------------------
# Load previous combined file
# -----------------------------
if os.path.exists(COMBINED_PATH):
    old_df = pd.read_csv(COMBINED_PATH)
    print("Existing combined shape:", old_df.shape)
else:
    old_df = pd.DataFrame()
    print("No existing combined file found. Starting fresh.")

# -----------------------------
# Load new source files
# -----------------------------
new_dfs = []

for file_path, source_name in SOURCE_FILES:
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)

        # force correct source name
        df["source_system"] = source_name

        new_dfs.append(df)
        print(f"{source_name} shape:", df.shape)
    else:
        print(f"{source_name} file not found:", file_path)

if not new_dfs and old_df.empty:
    raise FileNotFoundError("No source files found to merge.")

# -----------------------------
# Combine new files
# -----------------------------
if new_dfs:
    new_df = pd.concat(new_dfs, ignore_index=True)
else:
    new_df = pd.DataFrame()

print("New incoming shape:", new_df.shape)

# -----------------------------
# Align columns
# -----------------------------
all_columns = sorted(set(old_df.columns) | set(new_df.columns))

if not old_df.empty:
    old_df = old_df.reindex(columns=all_columns)

if not new_df.empty:
    new_df = new_df.reindex(columns=all_columns)

# -----------------------------
# Append old + new
# -----------------------------
combined_df = pd.concat([old_df, new_df], ignore_index=True)

print("Combined before dedup:", combined_df.shape)

# -----------------------------
# Remove duplicates
# -----------------------------
if "url" in combined_df.columns:
    before = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=["url"])
    print("Duplicate URLs removed:", before - len(combined_df))
else:
    before = len(combined_df)
    combined_df = combined_df.drop_duplicates()
    print("Full-row duplicates removed:", before - len(combined_df))

# -----------------------------
# Sort by date
# -----------------------------
if "published_at" in combined_df.columns:
    combined_df["published_at"] = pd.to_datetime(
        combined_df["published_at"],
        errors="coerce"
    )
    combined_df = combined_df.sort_values("published_at", ascending=False)

# -----------------------------
# Save combined file
# -----------------------------
os.makedirs("data", exist_ok=True)
combined_df.to_csv(COMBINED_PATH, index=False)

print("\nFinal combined shape:", combined_df.shape)

if "source_system" in combined_df.columns:
    print("\nRows by source_system:")
    print(combined_df["source_system"].value_counts(dropna=False))

print("\nSaved to:", COMBINED_PATH)
print("===== MERGE COMPLETE =====")