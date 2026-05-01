import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("analysis_outputs", exist_ok=True)

df = pd.read_csv("data/cleaned_news.csv")

df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
df = df.dropna(subset=["published_at"])
df["date"] = df["published_at"].dt.date

daily_counts = df.groupby("date").size()
moving_avg = daily_counts.rolling(3).mean()

plt.figure(figsize=(10, 5))
daily_counts.plot(label="Daily Articles", marker="o")
moving_avg.plot(label="3-Day Moving Average")
plt.title("News Volume Trend Over Time")
plt.xlabel("Date")
plt.ylabel("Number of Articles")
plt.legend()
plt.tight_layout()

trend_path = "analysis_outputs/trend_plot.png"
plt.savefig(trend_path)
plt.close()

print("Trend plot saved to:", trend_path)

top_countries = df["country"].value_counts().head(5).index

country_trends = (
    df[df["country"].isin(top_countries)]
    .groupby(["date", "country"])
    .size()
    .unstack(fill_value=0)
)

plt.figure(figsize=(10, 5))
for col in country_trends.columns:
    country_trends[col].plot(label=col)

plt.title("Top Countries Trend Over Time")
plt.xlabel("Date")
plt.ylabel("Articles")
plt.legend()
plt.tight_layout()

country_path = "analysis_outputs/country_trend_plot.png"
plt.savefig(country_path)
plt.close()

print("Country trend plot saved to:", country_path)

print("\nTop 5 highest activity days:")
print(daily_counts.sort_values(ascending=False).head())