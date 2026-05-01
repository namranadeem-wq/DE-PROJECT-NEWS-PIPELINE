from fastapi import FastAPI
import sqlite3
import pandas as pd

app = FastAPI(title="News Pipeline API")


# -----------------------------
# GET ALL ARTICLES
# -----------------------------
@app.get("/articles")
def get_articles(limit: int = 10):
    conn = sqlite3.connect("data/news.db")

    query = f"""
    SELECT * FROM articles
    ORDER BY published_at DESC
    LIMIT {limit}
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    return df.to_dict(orient="records")


# -----------------------------
# GET BY CATEGORY
# -----------------------------
@app.get("/articles/category/{category}")
def get_by_category(category: str):
    conn = sqlite3.connect("data/news.db")

    query = f"""
    SELECT * FROM articles
    WHERE category = '{category}'
    ORDER BY published_at DESC
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    return df.to_dict(orient="records")


# -----------------------------
# ANALYTICS ENDPOINT
# -----------------------------
@app.get("/analytics")
def get_analytics():
    conn = sqlite3.connect("data/news.db")

    query = """
    SELECT category, COUNT(*) as count
    FROM articles
    GROUP BY category
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    return df.to_dict(orient="records")