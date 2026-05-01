import pandas as pd

df = pd.read_csv("data/combined_news.csv")

print("Columns:")
print(df.columns.tolist())

print("\nCountry column exists?:", "country" in df.columns)

if "country" in df.columns:
    print("\nFirst 10 values:")
    print(df["country"].head(10))

    print("\nUnique values (first 20):")
    print(df["country"].dropna().astype(str).str.strip().unique()[:20])

    print("\nValue counts:")
    print(df["country"].value_counts(dropna=False).head(20))
else:
    print("❌ No country column found")