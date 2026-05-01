import os
import requests
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("NEWS_API_KEY")


def infer_country(text):
    text = str(text).lower()

    if "pakistan" in text:
        return "Pakistan"
    elif "india" in text:
        return "India"
    elif "china" in text:
        return "China"
    elif "turkey" in text:
        return "Turkey"
    elif "iran" in text:
        return "Iran"
    elif "israel" in text:
        return "Israel"
    elif "gaza" in text or "palestine" in text:
        return "Palestine"
    elif "ukraine" in text:
        return "Ukraine"
    elif "russia" in text:
        return "Russia"
    elif "syria" in text:
        return "Syria"
    elif (
        "usa" in text
        or "united states" in text
        or "america" in text
        or "washington" in text
    ):
        return "United States"
    else:
        return "Other"


def build_dataframe_from_articles(articles, language="en", category="general"):
    fetched_at = datetime.now(timezone.utc).isoformat()
    rows = []

    for article in articles:
        title = article.get("title") or ""
        description = article.get("description") or ""
        full_text = f"{title} {description}"

        rows.append({
            "source_name": article.get("source", {}).get("name"),
            "author": article.get("author"),
            "title": title,
            "description": description,
            "url": article.get("url"),
            "published_at": article.get("publishedAt"),
            "fetched_at": fetched_at,
            "country": infer_country(full_text),
            "category": category,
            "domain": "Unknown",
            "language": language,
            "social_image": article.get("urlToImage"),
            "source_system": "NewsAPI"
        })

    return pd.DataFrame(rows)


def fetch_news_everything():
    url = "https://newsapi.org/v2/everything"
    params = {
        "apiKey": API_KEY,
        "q": (
            "war OR conflict OR politics OR world OR diplomacy OR military "
            "OR attack OR missile OR border OR Gaza OR Israel OR Iran "
            "OR Pakistan OR India OR China OR Russia OR Ukraine"
        ),
        "language": "en",
        "from": "2026-03-01",
        "to": "2026-04-23",
        "sortBy": "publishedAt",
        "pageSize": 100,
        "page": 1
    }

    print("Trying NewsAPI /v2/everything ...")
    response = requests.get(url, params=params, timeout=30)

    if response.status_code != 200:
        print("Everything endpoint status:", response.status_code)
        try:
            print(response.json())
        except Exception:
            print(response.text)

    response.raise_for_status()

    data = response.json()
    articles = data.get("articles", [])
    return build_dataframe_from_articles(articles, language="en", category="broad_news")


def fetch_news_top_headlines_broad():
    url = "https://newsapi.org/v2/top-headlines"
    all_articles = []

    query_buckets = [
        "war OR conflict OR attack OR military",
        "Gaza OR Israel OR Palestine",
        "Iran OR Pakistan OR India OR China",
        "Russia OR Ukraine",
        "world OR diplomacy OR politics"
    ]

    categories = ["general", "business", "technology", "health"]

    print("Falling back to broad top-headlines ...")

    # query-based pulls
    for q in query_buckets:
        params = {
            "apiKey": API_KEY,
            "q": q,
            "language": "en",
            "pageSize": 100,
            "page": 1
        }

        response = requests.get(url, params=params, timeout=30)
        if response.status_code == 200:
            data = response.json()
            all_articles.extend(data.get("articles", []))
        else:
            print(f"Query bucket failed: {q} | status {response.status_code}")

    # category-based pulls
    for category in categories:
        params = {
            "apiKey": API_KEY,
            "category": category,
            "country": "us",
            "pageSize": 100,
            "page": 1
        }

        response = requests.get(url, params=params, timeout=30)
        if response.status_code == 200:
            data = response.json()
            all_articles.extend(data.get("articles", []))
        else:
            print(f"Category failed: {category} | status {response.status_code}")

    df = build_dataframe_from_articles(all_articles, language="en", category="broad_news")
    df = df.drop_duplicates(subset=["url"])
    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    try:
        df = fetch_news_everything()
    except Exception as e:
        print("Everything endpoint failed:", e)
        df = fetch_news_top_headlines_broad()

    if df.empty:
        print("No data returned.")
    else:
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
        df["published_at"] = df["published_at"].dt.tz_localize(None)

        df = df.dropna(subset=["url"])
        df = df.drop_duplicates(subset=["url"])
        df = df.sort_values(by="published_at", ascending=False)

        print("Shape:", df.shape)
        print("\nCountry counts:")
        print(df["country"].value_counts())
        print("\nSource counts:")
        print(df["source_name"].value_counts().head(10))

    output_path = "data/processed_news.csv"
    df.to_csv(output_path, index=False)

    print(f"\nSaved to {output_path}")