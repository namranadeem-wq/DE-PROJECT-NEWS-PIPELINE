import os
import requests
import pandas as pd
from datetime import datetime, timezone


API_KEY = "ec1632e7c7b749259b3cc62c464bbf2e"


def fetch_worldnews_data(query="war OR conflict OR Gaza OR Israel OR Iran OR Pakistan", max_results=50):
    """
    Fetch news from World News API
    """

    url = "https://api.worldnewsapi.com/search-news"

    params = {
        "api-key": API_KEY,
        "text": query,
        "language": "en",
        "number": max_results,
        "sort": "publish-time"
    }

    print("Fetching from World News API...")

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()

    articles = data.get("news", [])

    rows = []
    fetched_at = datetime.now(timezone.utc).isoformat()

    for article in articles:
        rows.append({
            "source_name": article.get("source"),
            "title": article.get("title"),
            "description": article.get("text"),
            "url": article.get("url"),
            "published_at": article.get("publish_date"),
            "fetched_at": fetched_at,
            "country": article.get("country"),
            "category": "geopolitics",
            "domain": article.get("url"),
            "language": article.get("language"),
            "social_image": article.get("image"),
            "source_system": "WorldNewsAPI"
        })

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    df = fetch_worldnews_data(
        query="war OR conflict OR Gaza OR Israel OR Iran OR Pakistan",
        max_results=50
    )

    if df.empty:
        print("No data returned.")
    else:
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
        df["published_at"] = df["published_at"].dt.tz_localize(None)

        df = df.dropna(subset=["url"])
        df = df.drop_duplicates(subset=["url"])
        df = df.sort_values(by="published_at", ascending=False)

        print(df.head())
        print("Shape:", df.shape)

    output_path = "data/processed_worldnews.csv"
    df.to_csv(output_path, index=False)

    print(f"Saved to {output_path}")