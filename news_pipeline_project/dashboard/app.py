import os
import json
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="News Dashboard", layout="wide")

st.title("News Pipeline Dashboard")
st.write("Multi-source news data from NewsAPI + GDELT + Worldnews")

DATA_PATH = "data/cleaned_news.csv"
METRICS_PATH = "analysis_outputs/prediction_metrics.json"
REPORT_PATH = "analysis_outputs/classification_report.csv"
PREDICTIONS_PATH = "analysis_outputs/test_predictions.csv"
SAMPLE_PRED_PATH = "analysis_outputs/sample_predictions.csv"
CONF_MATRIX_PATH = "analysis_outputs/confusion_matrix.png"

if not os.path.exists(DATA_PATH):
    st.error(f"File not found: {DATA_PATH}")
    st.stop()

df = pd.read_csv(DATA_PATH)

# -----------------------------
# Basic cleaning
# -----------------------------
if "published_at" in df.columns:
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    df["published_at"] = df["published_at"].dt.tz_localize(None)
else:
    df["published_at"] = pd.NaT

defaults = {
    "source_name": "Unknown",
    "description": "No description",
    "domain": "Unknown",
    "language": "unknown",
    "social_image": "N/A",
    "category": "Unknown",
    "source_system": "Unknown",
    "title": "",
    "url": "",
    "country": "Unknown",
}

for col, default in defaults.items():
    if col not in df.columns:
        df[col] = default
    df[col] = df[col].fillna(default).astype(str).str.strip()
    df.loc[df[col] == "", col] = default

country_col = "topic_country" if "topic_country" in df.columns else "country"

if country_col not in df.columns:
    df[country_col] = "Unknown"

df[country_col] = df[country_col].fillna("Unknown").astype(str).str.strip()
df.loc[df[country_col] == "", country_col] = "Unknown"

if country_col == "country":
    df[country_col] = df[country_col].replace({
        "us": "United States",
        "usa": "United States"
    })

# -----------------------------
# Sidebar filters
# -----------------------------
st.sidebar.header("Filters")

source_options = sorted(df["source_system"].dropna().unique().tolist())
selected_sources = st.sidebar.multiselect(
    "Source System",
    options=source_options,
    default=source_options
)

category_options = sorted(df["category"].dropna().unique().tolist())
selected_categories = st.sidebar.multiselect(
    "Category",
    options=category_options,
    default=category_options
)

country_options = sorted(df[country_col].dropna().unique().tolist())
selected_countries = st.sidebar.multiselect(
    "Country",
    options=country_options,
    default=[]
)

search = st.sidebar.text_input("Search")

valid_dates = df["published_at"].dropna()
if not valid_dates.empty:
    min_date = valid_dates.min().date()
    max_date = valid_dates.max().date()

    start_date = st.sidebar.date_input(
        "Start Date",
        value=min_date,
        min_value=min_date,
        max_value=max_date
    )

    end_date = st.sidebar.date_input(
        "End Date",
        value=max_date,
        min_value=min_date,
        max_value=max_date
    )
else:
    start_date = None
    end_date = None

# -----------------------------
# Apply filters
# -----------------------------
filtered_df = df.copy()

if selected_sources:
    filtered_df = filtered_df[filtered_df["source_system"].isin(selected_sources)]

if selected_categories:
    filtered_df = filtered_df[filtered_df["category"].isin(selected_categories)]

if selected_countries:
    filtered_df = filtered_df[filtered_df[country_col].isin(selected_countries)]

if search:
    search_series = (
        filtered_df["title"].astype(str) + " " +
        filtered_df["description"].astype(str) + " " +
        filtered_df["category"].astype(str) + " " +
        filtered_df[country_col].astype(str)
    )
    filtered_df = filtered_df[
        search_series.str.contains(search, case=False, na=False)
    ]

if start_date is not None and end_date is not None:
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    if start <= end:
        filtered_df = filtered_df[
            (filtered_df["published_at"] >= start) &
            (filtered_df["published_at"] <= end)
        ]
    else:
        st.sidebar.error("End Date must be after Start Date")

# -----------------------------
# Overview
# -----------------------------
st.subheader("Overview")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Articles", len(filtered_df))
col2.metric("Sources", filtered_df["source_system"].nunique() if not filtered_df.empty else 0)
col3.metric("Domains", filtered_df["domain"].nunique() if not filtered_df.empty else 0)
col4.metric("Countries", filtered_df[country_col].nunique() if not filtered_df.empty else 0)

# -----------------------------
# Charts
# -----------------------------
left_col, right_col = st.columns(2)

with left_col:
    st.subheader("Articles by Source")
    if not filtered_df.empty:
        fig, ax = plt.subplots()
        filtered_df["source_system"].value_counts().plot(kind="bar", ax=ax)
        ax.set_xlabel("Source")
        ax.set_ylabel("Count")
        ax.set_title("Articles by Source")
        plt.xticks(rotation=0)
        st.pyplot(fig)
    else:
        st.info("No data available for this chart.")

with right_col:
    st.subheader("Articles Over Time")
    time_df = filtered_df.dropna(subset=["published_at"]).copy()

    if not time_df.empty:
        daily_counts = time_df.set_index("published_at").resample("D").size()
        fig, ax = plt.subplots()
        daily_counts.plot(ax=ax)
        ax.set_xlabel("Date")
        ax.set_ylabel("Articles")
        ax.set_title("Articles Over Time")
        st.pyplot(fig)
    else:
        st.info("No valid date data available.")

# -----------------------------
# Tables
# -----------------------------
col3, col4 = st.columns(2)

with col3:
    st.subheader("Top Domains")
    if not filtered_df.empty:
        top_domains = filtered_df["domain"].value_counts().head(10).reset_index()
        top_domains.columns = ["Domain", "Count"]
        st.dataframe(top_domains, use_container_width=True)
    else:
        st.info("No domain data available.")

with col4:
    st.subheader("Top Countries")
    if not filtered_df.empty:
        top_countries = filtered_df[country_col].value_counts().head(10).reset_index()
        top_countries.columns = ["Country", "Count"]
        st.dataframe(top_countries, use_container_width=True)
    else:
        st.info("No country data available.")

# -----------------------------
# Articles Table
# -----------------------------
st.subheader("Articles")

display_cols = [
    "published_at",
    "source_system",
    "source_name",
    "title",
    "category",
    country_col,
    "domain",
    "url"
]
display_cols = [col for col in display_cols if col in filtered_df.columns]

if not filtered_df.empty:
    table_df = filtered_df[display_cols].copy()
    if country_col in table_df.columns:
        table_df = table_df.rename(columns={country_col: "country"})
    st.dataframe(table_df, use_container_width=True)
else:
    st.info("No articles match the selected filters.")

# -----------------------------
# ML Section
# -----------------------------
st.subheader("Machine Learning: Country Prediction")

if os.path.exists(METRICS_PATH):
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)

    m1, m2, m3 = st.columns(3)
    m1.metric("Model Accuracy", round(metrics.get("accuracy", 0), 3))
    m2.metric("Train Size", metrics.get("train_size", 0))
    m3.metric("Test Size", metrics.get("test_size", 0))
else:
    st.info("Run analysis/predict_country.py first to generate ML outputs.")

if os.path.exists(REPORT_PATH):
    st.markdown("**Classification Report**")
    report_df = pd.read_csv(REPORT_PATH)
    st.dataframe(report_df, use_container_width=True)

if os.path.exists(CONF_MATRIX_PATH):
    st.markdown("**Confusion Matrix**")
    st.image(CONF_MATRIX_PATH, use_container_width=True)

if os.path.exists(PREDICTIONS_PATH):
    st.markdown("**Test Predictions**")
    pred_df = pd.read_csv(PREDICTIONS_PATH)
    st.dataframe(pred_df.head(20), use_container_width=True)

if os.path.exists(SAMPLE_PRED_PATH):
    st.markdown("**Sample Predictions**")
    sample_df = pd.read_csv(SAMPLE_PRED_PATH)
    st.dataframe(sample_df, use_container_width=True)

# -----------------------------
# Download
# -----------------------------
st.subheader("Download Data")

csv = filtered_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download CSV",
    data=csv,
    file_name="filtered_news.csv",
    mime="text/csv"
)