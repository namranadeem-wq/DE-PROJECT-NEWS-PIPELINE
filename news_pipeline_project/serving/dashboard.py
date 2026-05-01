from pathlib import Path
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_OK = True
except:
    ARIMA_OK = False

st.set_page_config(layout="wide")
st.title("News Dashboard")

BASE = Path(__file__).resolve().parent.parent if Path(__file__).parent.name == "serving" else Path.cwd()

paths = [
    BASE / "data/processed/cleaned_news.csv",
    BASE / "data/cleaned_news.csv",
    BASE / "data/processed/combined_news.csv",
    BASE / "data/combined_news.csv",
]

DATA_PATH = next((p for p in paths if p.exists()), None)
if DATA_PATH is None:
    st.error("No CSV found")
    st.stop()

df = pd.read_csv(DATA_PATH)

for col in ["title", "description", "source_system", "country", "published_at"]:
    if col not in df.columns:
        df[col] = "Unknown"

df["title"] = df["title"].fillna("")
df["description"] = df["description"].fillna("")
df["source_system"] = df["source_system"].fillna("Unknown")
df["country"] = df["country"].fillna("Unknown")
df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
df = df.dropna(subset=["published_at"])

df["text"] = df["title"].astype(str) * 2 + " " + df["description"].astype(str)

# -----------------------------
# Category rules
# -----------------------------
def assign_category(row):
    text = f"{row['title']} {row['description']}".lower()

    rules = {
        "geopolitics": ["war", "election", "government", "china", "russia", "gaza", "ukraine", "minister", "border"],
        "technology": ["ai", "technology", "software", "data", "app", "cyber", "chip", "robot"],
        "weather": ["weather", "rain", "storm", "flood", "climate", "temperature", "forecast"],
        "business": ["market", "stock", "economy", "finance", "bank", "inflation", "business", "trade"],
        "health": ["health", "doctor", "covid", "vaccine", "disease", "hospital"],
        "sports": ["cricket", "football", "match", "team", "player", "tournament"],
        "entertainment": ["movie", "film", "music", "actor", "celebrity"],
    }

    scores = {cat: 0 for cat in rules}

    for cat, words in rules.items():
        for word in words:
            if word in text:
                scores[cat] += 1

    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"

# -----------------------------
# Country rules
# -----------------------------
def infer_country_from_text(text):
    text = str(text).lower()

    rules = {
        "Pakistan": [
            "pakistan", "pakistani", "islamabad", "lahore", "karachi",
            "peshawar", "quetta", "punjab", "sindh", "kpk", "balochistan",
            "pti", "pml-n", "ppp", "imran khan", "shehbaz", "bilawal"
        ],
        "India": ["india", "indian", "delhi", "mumbai", "modi", "bjp"],
        "United States": ["united states", "america", "american", "washington", "trump", "biden"],
        "United Kingdom": ["united kingdom", "britain", "british", "london"],
        "China": ["china", "chinese", "beijing"],
        "Russia": ["russia", "russian", "moscow"],
        "Ukraine": ["ukraine", "ukrainian", "kyiv"],
        "Iran": ["iran", "iranian", "tehran"],
        "Israel": ["israel", "israeli", "tel aviv"],
        "Gaza": ["gaza", "hamas", "palestine", "palestinian"],
    }

    scores = {country: 0 for country in rules}

    for country, words in rules.items():
        for word in words:
            if word in text:
                scores[country] += 1

    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "Unknown"

df["category"] = df.apply(assign_category, axis=1)
df["inferred_country"] = df["text"].apply(infer_country_from_text)

df["country_for_ml"] = df["inferred_country"]
df.loc[df["country_for_ml"] == "Unknown", "country_for_ml"] = df["country"]

# -----------------------------
# Filters
# -----------------------------
st.sidebar.header("Filters")

sources = sorted(df["source_system"].unique())
cats = sorted(df["category"].unique())
countries = sorted(df["country_for_ml"].unique())

sf = st.sidebar.multiselect("Source", sources, sources)
cf = st.sidebar.multiselect("Category", cats, cats)
ctf = st.sidebar.multiselect("Country", countries, [])

filtered = df[df["source_system"].isin(sf) & df["category"].isin(cf)]

if ctf:
    filtered = filtered[filtered["country_for_ml"].isin(ctf)]

if filtered.empty:
    st.warning("No articles match filters.")
    st.stop()

# -----------------------------
# Metrics
# -----------------------------
m1, m2, m3, m4 = st.columns(4)
m1.metric("Articles", len(filtered))
m2.metric("Sources", filtered["source_system"].nunique())
m3.metric("Countries", filtered["country_for_ml"].nunique())
m4.metric("Categories", filtered["category"].nunique())

# -----------------------------
# Small charts
# -----------------------------
c1, c2 = st.columns(2)

with c1:
    fig, ax = plt.subplots(figsize=(4, 2.5))
    filtered["source_system"].value_counts().plot(kind="bar", ax=ax)
    ax.set_title("Sources")
    ax.set_ylabel("Articles")
    st.pyplot(fig, use_container_width=False)

with c2:
    fig, ax = plt.subplots(figsize=(4, 2.5))
    filtered["category"].value_counts().plot(kind="bar", ax=ax)
    ax.set_title("Categories")
    ax.set_ylabel("Articles")
    st.pyplot(fig, use_container_width=False)

# -----------------------------
# ARIMA
# -----------------------------
st.subheader("ARIMA Forecast")

daily = filtered.set_index("published_at").resample("D").size()

fig, ax = plt.subplots(figsize=(5, 3))
daily.plot(ax=ax, label="Actual", marker="o")

if ARIMA_OK and len(daily) >= 7:
    try:
        fc = ARIMA(daily, order=(1, 1, 1)).fit().forecast(5)
        fc.plot(ax=ax, label="Forecast", marker="o")
    except Exception as e:
        st.info(f"ARIMA could not run: {e}")

ax.set_title("Actual vs Forecasted News Volume")
ax.set_xlabel("Date")
ax.set_ylabel("Articles")
ax.legend()
st.pyplot(fig, use_container_width=False)

# -----------------------------
# TF-IDF by category
# -----------------------------
st.subheader("TF-IDF N-grams by Category")

sel_cat = st.selectbox("Select Category", sorted(filtered["category"].unique()))
cat_df = filtered[filtered["category"] == sel_cat]

if len(cat_df) >= 3:
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=1000)
    X = vec.fit_transform(cat_df["text"])
    scores = X.mean(axis=0).A1
    terms = vec.get_feature_names_out()

    tfidf_df = pd.DataFrame({"ngram": terms, "score": scores})
    tfidf_df = tfidf_df.sort_values("score", ascending=False).head(10)

    st.dataframe(tfidf_df, use_container_width=True)

    fig, ax = plt.subplots(figsize=(5, 3))
    tfidf_df.set_index("ngram")["score"].sort_values().plot(kind="barh", ax=ax)
    ax.set_title(f"Top N-grams: {sel_cat}")
    ax.set_xlabel("TF-IDF Score")
    st.pyplot(fig, use_container_width=False)
else:
    st.info("Not enough articles in this category.")

# -----------------------------
# ML helper
# -----------------------------
def train_model(data, target):
    temp = data.copy()
    temp = temp[
        (temp[target].notna())
        & (temp[target] != "Unknown")
        & (temp[target] != "general")
        & (temp["text"].str.strip() != "")
    ]

    counts = temp[target].value_counts()
    temp = temp[temp[target].isin(counts[counts >= 5].index)]

    if len(temp) < 20 or temp[target].nunique() < 2:
        return None

    temp = temp.groupby(target, group_keys=False).apply(
        lambda x: x.sample(min(len(x), 200), random_state=42)
    )

    X = temp["text"]
    y = temp[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=5000,
            min_df=2,
            max_df=0.9
        )),
        ("clf", LinearSVC())
    ])

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    return model, y_test, pred

# -----------------------------
# ML Predictions
# -----------------------------
st.subheader("ML Predictions")

country_res = train_model(df, "country_for_ml")
category_res = train_model(df, "category")

p1, p2 = st.columns(2)

with p1:
    st.markdown("### Country Prediction")

    if country_res:
        country_model, y_test, pred = country_res
        st.metric("Accuracy", round(accuracy_score(y_test, pred), 3))

        fig, ax = plt.subplots(figsize=(4, 3))
        ConfusionMatrixDisplay.from_predictions(
            y_test, pred, ax=ax, xticks_rotation=45
        )
        ax.set_title("Country Confusion Matrix")
        st.pyplot(fig, use_container_width=False)
    else:
        st.info("Not enough country labels for ML.")

with p2:
    st.markdown("### Category Prediction")

    if category_res:
        category_model, y_test, pred = category_res
        st.metric("Accuracy", round(accuracy_score(y_test, pred), 3))

        fig, ax = plt.subplots(figsize=(4, 3))
        ConfusionMatrixDisplay.from_predictions(
            y_test, pred, ax=ax, xticks_rotation=45
        )
        ax.set_title("Category Confusion Matrix")
        st.pyplot(fig, use_container_width=False)
    else:
        st.info("Not enough category labels for ML.")

# -----------------------------
# Try prediction
# -----------------------------
st.subheader("Try Prediction")

txt = st.text_area(
    "Enter article text",
    "Pakistan government invests in AI technology and business growth"
)

if st.button("Predict"):
    rule_country = infer_country_from_text(txt)

    if rule_country != "Unknown":
        st.success(f"Predicted Country: {rule_country}")
    elif country_res:
        st.success(f"Predicted Country: {country_res[0].predict([txt])[0]}")

    if category_res:
        st.success(f"Predicted Category: {category_res[0].predict([txt])[0]}")
    else:
        st.success(f"Predicted Category: {assign_category({'title': txt, 'description': ''})}")

# -----------------------------
# Articles
# -----------------------------
st.subheader("Articles")

st.dataframe(
    filtered[[
        "published_at",
        "source_system",
        "category",
        "country_for_ml",
        "title"
    ]],
    use_container_width=True
)