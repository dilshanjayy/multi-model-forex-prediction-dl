import time
import pandas as pd
from datetime import datetime, timedelta
from src.data.news_data_collector import fetch_news_data


def generate_date_chunks(
    start_date: datetime, end_date: datetime, months_per_chunk: int = 3
):
    """Generates MMDDYYYY-MMDDYYYY strings in chunks."""
    current = start_date
    chunks = []

    while current < end_date:
        chunk_start = current
        # Add months
        year = current.year + (current.month + months_per_chunk - 1) // 12
        month = (current.month + months_per_chunk - 1) % 12 + 1
        chunk_end = datetime(year, month, 1) - timedelta(days=1)

        if chunk_end > end_date:
            chunk_end = end_date

        chunks.append(
            f"{chunk_start.strftime('%m%d%Y')}-{chunk_end.strftime('%m%d%Y')}"
        )
        current = chunk_end + timedelta(days=1)

    return chunks


def run_historical_collection():
    """
    Collects news from April 2021 to April 2026.
    Matches the project's historical market data range.
    """
    start_date = datetime(2021, 4, 1)
    end_date = datetime(2026, 4, 30)

    # 3-month chunks are efficient for the API
    date_ranges = generate_date_chunks(start_date, end_date, months_per_chunk=3)

    print(f"--- Starting Historical News Collection ({len(date_ranges)} chunks) ---")

    all_dfs = []

    for i, date_range in enumerate(date_ranges):
        print(f"\n[Chunk {i + 1}/{len(date_ranges)}] Processing: {date_range}")

        try:
            df = fetch_news_data(date_range)
            if df is not None and len(df) > 0:
                all_dfs.append(df)

            # Small sleep to be polite to the API
            time.sleep(1)

        except Exception as e:
            print(f"Failed chunk {date_range}: {e}")
            continue

    if all_dfs:
        full_df = pd.concat(all_dfs, ignore_index=True)
        # Drop duplicates just in case chunks overlapped
        full_df.drop_duplicates(subset=["time", "title"], inplace=True)
        full_df.sort_values("time", ascending=False, inplace=True)

        master_path = "data/raw_sentiment/news_EURUSD_Full_2021_2026.csv"
        full_df.to_csv(master_path, index=False)

        print("\n" + "=" * 40)
        print("HISTORICAL COLLECTION COMPLETE")
        print(f"Total Headlines: {len(full_df)}")
        print(f"Master File: {master_path}")
        print("=" * 40)


if __name__ == "__main__":
    run_historical_collection()
