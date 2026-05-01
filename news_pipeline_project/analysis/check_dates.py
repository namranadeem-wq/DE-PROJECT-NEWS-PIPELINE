import pandas as pd

df = pd.read_csv("data/cleaned_news.csv")

df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")

print(df["published_at"].dt.date.value_counts().sort_index())