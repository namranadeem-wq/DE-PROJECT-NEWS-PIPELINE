import subprocess

commands = [
    "python ingestion/fetch_news.py",
    "python ingestion/fetch_gdelt.py",
    "python processing/merge_data.py",
    "python processing/clean_data.py",
    "python processing/validate_data.py",
    "python analysis/trend_analysis.py",
    "python analysis/predict_country.py",
]

for cmd in commands:
    print(f"\nRunning: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed: {cmd}")

print("\nPipeline completed successfully.")