"""
Configuration for SEAL-drip: A Lifelong-Updating Llama
Simple settings file to keep everything organized.
"""

import os
from datetime import datetime, timedelta

# === Model Configuration ===
MODEL_NAME = "llama3.1:8b-instruct-q4_K_M"  # Your specific Ollama model
OLLAMA_HOST = "http://localhost:11434"

# === Data Collection Settings ===
MAX_DAILY_ITEMS = 10000  # Max items to collect per day (from spec)
DATA_DIR = "data"
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# === Data Sources ===
WIKIPEDIA_RECENT_CHANGES_URL = "https://en.wikipedia.org/api/rest_v1/feed/featured/2024/01/01"
HACKERNEWS_TOP_URL = "https://hacker-news.firebaseio.com/v0/topstories.json"
HACKERNEWS_ITEM_URL = "https://hacker-news.firebaseio.com/v0/item/{}.json"
ARXIV_RSS_URL = "http://rss.arxiv.org/rss/cs"

# === Collection Parameters ===
HACKERNEWS_MAX_ITEMS = 30  # Top N stories to collect
ARXIV_MAX_ITEMS = 50       # Max papers per day
WIKIPEDIA_LOOKBACK_DAYS = 1  # How many days back to look

# === File Naming ===
def get_daily_filename(source: str, date: datetime = None) -> str:
    """Generate filename for daily data collection"""
    if date is None:
        date = datetime.now()
    date_str = date.strftime("%Y%m%d")
    return f"{source}_{date_str}.jsonl"

# === Directories Setup ===
REQUIRED_DIRS = [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR]

def ensure_directories():
    """Create required directories if they don't exist"""
    for dir_path in REQUIRED_DIRS:
        os.makedirs(dir_path, exist_ok=True) 