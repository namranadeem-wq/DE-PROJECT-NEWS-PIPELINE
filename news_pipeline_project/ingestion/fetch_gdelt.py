import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="News Dashboard", layout="wide")

st.title("News Pipeline Dashboard")
st.write("Multi-source news data from NewsAPI + GDELT")

DATA_PATH = "data/combined_news.csv"

# -----------------------------
# Load data
# -----------------------------
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

if "source_name" not in df.columns:
    df["source_name"] = "Unknown"
df["source_name"] = df["source_name"].fillna("Unknown")

if "description" not in df.columns:
    df["description"] = "No description"
df["description"] = df["description"].fillna("No description")

if "domain" not in df.columns:
    df["domain"] = "Unknown"
df["domain"] = df["domain"].fillna("Unknown")

if "language" not in df.columns:
    df["language"] = "unknown"
df["language"] = df["language"].fillna("unknown")

if "social_image" not in df.columns:
    df["social_image"] = "N/A"
df["social_image"] = df["social_image"].fillna("N/A")

if "category" not in df.columns:
    df["category"] = "Unknown"
df["category"] = df["category"].fillna("Unknown").astype(str).str.strip()
df.loc[df["category"] == "", "category"] = "Unknown"

if "source_system" not in df.columns:
    df["source_system"] = "Unknown"
df["source_system"] = df["source_system"].fillna("Unknown").astype(str).str.strip()
df.loc[df["source_system"] == "", "source_system"] = "Unknown"

# -----------------------------
# Use topic_country if available
# -----------------------------
country_col = "topic_country" if "topic_country" in df.columns else "country"

if country_col not in df.columns:
    df[country_col] = "Unknown"

df[country_col] = df[country_col].fillna("Unknown").astype(str).str.strip()
df.loc[df[country_col] == "", country_col] = "Unknown"

# Optional normalization if using raw country
if country_col == "country":
    df[country_col] = df[country_col].replace({
        "us": "United States",
        "usa": "United States"
    })

# -----------------------------
# Country focus list
# -----------------------------
important_countries = [
    "Pakistan",
    "Iran",
    "Palestine",
    "Israel",
    "United States"
]

# -----------------------------
# Sidebar filters
# -----------------------------
st.sidebar.header("Filters")

source_options = sorted(df["source_system"].dropna().unique().tolist())
sources = st.sidebar.multiselect(
    "Source System",
    options=source_options,
    default=source_options
)

category_options = sorted(df["category"].dropna().unique().tolist())
categories = st.sidebar.multiselect(
    "Category",
    options=category_options,
    default=category_options
)

country_options = sorted(df[country_col].dropna().unique().tolist())
default_countries = [c for c in important_countries if c in country_options]

countries = st.sidebar.multiselect(
    "Country",
    options=country_options,
    default=default_countries if default_countries else country_options
)

search = st.sidebar.text_input("Search Title")

valid_dates = df["published_at"].dropna()
if not valid_dates.empty:
    min_date = valid_dates.min().date()
    max_date = valid_dates.max().date()

    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
else:
    date_range = None

focus_mode = st.sidebar.checkbox("Show only focus countries", value=False)

# -----------------------------
# Apply filters
# -----------------------------
filtered_df = df.copy()

if focus_mode:
    filtered_df = filtered_df[filtered_df[country_col].isin(important_countries)]

if sources:
    filtered_df = filtered_df[filtered_df["source_system"].isin(sources)]

if categories:
    filtered_df = filtered_df[filtered_df["category"].isin(categories)]

if countries:
    filtered_df = filtered_df[filtered_df[country_col].isin(countries)]

if search:
    filtered_df = filtered_df[
        filtered_df["title"].astype(str).str.contains(search, case=False, na=False)
    ]

if date_range and len(date_range) == 2:
    start = pd.to_datetime(date_range[0])
    end = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    filtered_df = filtered_df[
        (filtered_df["published_at"] >= start) &
        (filtered_df["published_at"] <= end)
    ]

# -----------------------------
# KPIs
# -----------------------------
st.subheader("Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Articles", len(filtered_df))
col2.metric("Sources", filtered_df["source_system"].nunique())
col3.metric("Domains", filtered_df["domain"].nunique())
col4.metric("Countries", filtered_df[country_col].nunique())

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
        weekly = time_df.set_index("published_at").resample("W").size()
        fig, ax = plt.subplots()
        weekly.plot(ax=ax)
        ax.set_xlabel("Week")
        ax.set_ylabel("Articles")
        ax.set_title("Weekly Articles Over Time")
        st.pyplot(fig)
    else:
        st.info("No valid date data available.")

# -----------------------------
# Top tables
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
# Focus country summary
# -----------------------------
st.subheader("Focus Country Summary")

focus_df = filtered_df[filtered_df[country_col].isin(important_countries)]

if not focus_df.empty:
    focus_counts = focus_df[country_col].value_counts().reset_index()
    focus_counts.columns = ["Country", "Article Count"]
    st.dataframe(focus_counts, use_container_width=True)
else:
    st.info("No articles found for the selected focus countries.")

# -----------------------------
# Articles table
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