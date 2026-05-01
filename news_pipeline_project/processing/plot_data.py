import pandas as pd
import matplotlib.pyplot as plt
import os

file_path = "data/combined_news.csv"

if not os.path.exists(file_path):
    raise FileNotFoundError(f"Missing file: {file_path}")

df = pd.read_csv(file_path)

# Convert date
df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")

# ---- Plot 1: Articles by Source ----
source_counts = df["source_system"].value_counts()

plt.figure()
source_counts.plot(kind="bar")
plt.title("Articles by Source")
plt.xlabel("Source")
plt.ylabel("Count")
plt.xticks(rotation=0)

os.makedirs("reports", exist_ok=True)
plt.savefig("reports/source_distribution.png")
plt.close()

# ---- Plot 2: Articles over Time ----
daily_counts = df["published_at"].dt.date.value_counts().sort_index()

weekly_counts = df.set_index("published_at").resample("W").size()
weekly_counts.plot()

plt.figure()
daily_counts.plot()
plt.title("Articles Over Time")
plt.xlabel("Date")
plt.ylabel("Number of Articles")

plt.savefig("reports/articles_over_time.png")
plt.close()

print("Plots saved in reports/")