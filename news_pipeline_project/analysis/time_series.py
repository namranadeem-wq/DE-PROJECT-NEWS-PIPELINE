import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv("data/cleaned_news.csv")

df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
df = df.dropna(subset=["published_at"])

df["date"] = df["published_at"].dt.date
daily = df.groupby("date").size()

rolling = daily.rolling(2).mean()

plt.figure(figsize=(10,5))
daily.plot(label="Daily", marker="o")
rolling.plot(label="Moving Avg")

plt.legend()

desktop = os.path.join(os.path.expanduser("~"), "Desktop", "final_timeseries.png")
plt.savefig(desktop)

print("Saved:", desktop)