import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from logs.logger import setup_logger

logger = setup_logger()


def clean_data(
    input_path="data/combined_news.csv",        # ✅ FIXED
    output_path="data/cleaned_news.csv"         # ✅ FIXED
):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    logger.info("===== DATA CLEANING STARTED =====")
    logger.info(f"Rows before cleaning: {len(df)}")

    # -----------------------------
    # Drop useless column
    # -----------------------------
    if "topic_country" in df.columns:
        df = df.drop(columns=["topic_country"])
        logger.info("Dropped column: topic_country")

    # -----------------------------
    # Standardize missing text fields
    # -----------------------------
    text_cols = ["source_name", "author", "title", "description", "category"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].replace(["", "nan", "None"], pd.NA)

    # -----------------------------
    # Drop duplicate URLs
    # -----------------------------
    if "url" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=["url"])
        logger.info(f"Dropped duplicate URLs: {before - len(df)}")

    # -----------------------------
    # Drop rows missing critical fields
    # -----------------------------
    critical_cols = [c for c in ["title", "url"] if c in df.columns]
    if critical_cols:
        before = len(df)
        df = df.dropna(subset=critical_cols)
        logger.info(f"Dropped rows missing {critical_cols}: {before - len(df)}")

    # -----------------------------
    # Fill missing values
    # -----------------------------
    if "author" in df.columns:
        df["author"] = df["author"].fillna("Unknown")

    if "source_name" in df.columns:
        df["source_name"] = df["source_name"].fillna("Unknown")

    if "description" in df.columns:
        df["description"] = df["description"].fillna("No description")

    if "country" in df.columns:
        df["country"] = df["country"].fillna("Unknown")

    if "social_image" in df.columns:
        df["social_image"] = df["social_image"].fillna("N/A")

    # -----------------------------
    # Handle dates
    # -----------------------------
    if "published_at" in df.columns:
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")

        invalid = df["published_at"].isna().sum()
        logger.info(f"Invalid published_at values: {invalid}")

        before = len(df)
        df = df.dropna(subset=["published_at"])
        logger.info(f"Dropped rows with invalid dates: {before - len(df)}")

    # -----------------------------
    # Final shape
    # -----------------------------
    logger.info(f"Rows after cleaning: {len(df)}")

    # -----------------------------
    # Save cleaned file
    # -----------------------------
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info(f"Cleaned data saved to: {output_path}")
    logger.info("===== DATA CLEANING COMPLETE =====")

    return df


if __name__ == "__main__":
    clean_data()