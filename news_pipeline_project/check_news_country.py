import pandas as pd
df = pd.read_csv("data/processed_news.csv")
print(df["country"].value_counts())
