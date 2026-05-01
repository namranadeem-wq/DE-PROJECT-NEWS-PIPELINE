import os
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

os.makedirs("analysis_outputs", exist_ok=True)

# Load data
df = pd.read_csv("data/cleaned_news.csv")

df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
df = df.dropna(subset=["published_at"])

# Proper daily datetime index
df["date"] = df["published_at"].dt.floor("D")

# Stable subset
df = df[
    (df["date"] >= "2026-04-21") &
    (df["date"] <= "2026-04-24")
]

daily = df.groupby("date").size()
daily.index = pd.to_datetime(daily.index)
daily = daily.asfreq("D").fillna(0)

# Smooth data
daily_smooth = daily.rolling(2).mean().dropna()

print("Daily counts used:")
print(daily_smooth)

# ARIMA
model = ARIMA(daily_smooth, order=(1, 0, 1))
model_fit = model.fit()

forecast = model_fit.forecast(steps=3)
forecast = forecast.clip(lower=0)

# Proper forecast dates
forecast_dates = pd.date_range(
    start=daily_smooth.index[-1] + pd.Timedelta(days=1),
    periods=3,
    freq="D"
)
forecast.index = forecast_dates

print("\nForecast:")
print(forecast)

# Plot
plt.figure(figsize=(9, 4))
daily_smooth.plot(label="Actual", marker="o")
forecast.plot(label="Forecast", marker="o")

plt.title("ARIMA Forecast (Smoothed Daily News Volume)")
plt.xlabel("Date")
plt.ylabel("Articles")
plt.legend()
plt.tight_layout()

plt.savefig("analysis_outputs/arima_forecast.png")
plt.close()

# Save forecast values
forecast_df = pd.DataFrame({
    "date": forecast.index,
    "predicted_articles": forecast.values
})
forecast_df.to_csv("analysis_outputs/arima_forecast.csv", index=False)

print("\nSaved:")
print("analysis_outputs/arima_forecast.png")
print("analysis_outputs/arima_forecast.csv")