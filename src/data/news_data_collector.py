import requests
import pandas as pd
import os
from datetime import datetime

def fetch_news_data(date_range: str = "03012026-03312026") -> pd.DataFrame | None:
    """
    Fetches multi-page news from ForexNewsAPI and saves to local CSV.
    Each page request consumes 1 trial credit.
    date_range format: MMDDYYYY-MMDDYYYY
    """
    API_TOKEN = "7ojgraypnrwh9oxsbiuopipe6vvclu7qb6eura6t"
    BASE_URL = "https://forexnewsapi.com/api/v1"
    CURRENCY_PAIR = "EUR-USD"

    output_dir = "data/raw_sentiment"
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{output_dir}/news_{CURRENCY_PAIR}_{date_range}.csv"

    all_news_raw = []

    total_pages = 0

    params = {
        "currencypair": CURRENCY_PAIR,
        "items": 100,
        "page": 1,
        "date": date_range,
        "token": API_TOKEN
    }

    # 1. Fetch First Page
    print(f"--- Fetching Page 1 for {date_range} ---")
    try:
        res = requests.get(BASE_URL, params=params)
        res.raise_for_status()
        data = res.json()

        all_news_raw.extend(data.get("data", []))
        total_pages = data.get("total_pages", 1)
        print(f"Found total of {total_pages} pages.")

        # 2. Loop through remaining pages (Capped at 100 per API rules)
        for page in range(2, min(total_pages, 100) + 1):
            print(f"--- Fetching Page {page} of {total_pages} ---")
            params["page"] = page
            res = requests.get(BASE_URL, params=params)
            res.raise_for_status()
            page_data = res.json()
            all_news_raw.extend(page_data.get("data", []))

    except Exception as e:
        print(f"CRITICAL ERROR during pagination: {e}")

    if not all_news_raw:
        print("No news items collected.")
        return

    # 3. Extract the Key-Pairs you specified
    news_list = []
    for item in all_news_raw:
        news_list.append({
            "time": item.get("date"),
            "title": item.get("title"),
            "text": item.get("text"),
            "source": item.get("source_name"),
            "sentiment_label": item.get("sentiment"), # Positive/Negative/Neutral
            "url": item.get("news_url"),
            "topics": ",".join(item.get("topics", [])), # Join list for CSV compatibility
            "type": item.get("type"),
            "tickers": ",".join(item.get("tickers", []))
        })

    # 4. Final Persistence
    df = pd.DataFrame(news_list)
    df.to_csv(filename, index=False)

    print(f"\n--- SUCCESS ---")
    print(f"Total Headlines: {len(df)}")
    print(f"Saved to: {filename}")
    print(f"Credits Used: Approx. {min(total_pages, 100)}")

    return df

if __name__ == "__main__":
    # Example: Fetch March 2026 news
    fetch_news_data("03012026-03312026")
