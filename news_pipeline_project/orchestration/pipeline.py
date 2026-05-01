import os
import subprocess
from pathlib import Path

# ✅ Correct Prefect API (use 4200 OR remove this line completely)
os.environ["PREFECT_API_URL"] = "http://127.0.0.1:4200/api"

from prefect import flow, task

BASE_DIR = Path(__file__).resolve().parents[1]


# ✅ Safe command runner (prints errors)
def run_cmd(command):
    print(f"\nRunning: {command}")
    result = subprocess.run(
        command,
        shell=True,
        cwd=BASE_DIR,
        capture_output=True,
        text=True
    )

    print(result.stdout)
    if result.stderr:
        print("ERROR:", result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {command}")


# 🔹 Tasks
@task
def merge_data():
    run_cmd("python -m processing.merge_data")


@task
def clean_data():
    run_cmd("python -m processing.clean_data")


@task
def validate_data():
    run_cmd("python -m processing.validate_data")


@task
def analyze_data():
    run_cmd("python -m processing.analyze_data")


@task
def train_model():
    run_cmd("python -m analysis.predict_country")


# 🔹 Flow
@flow(name="news-pipeline")
def news_pipeline():
    merge_data()
    clean_data()
    validate_data()
    analyze_data()
    train_model()


if __name__ == "__main__":
    news_pipeline()