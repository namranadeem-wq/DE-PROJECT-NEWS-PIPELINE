import pandas as pd

df = pd.read_csv("data/combined_news.csv")

print("Total records:", len(df))
print("\nRecords by source:")
print(df["source_system"].value_counts())

print("\nRecords by country:")
print(df["country"].value_counts())

print("\nDate range:")
print(df["published_at"].min(), "to", df["published_at"].max())