import pandas as pd


def run_diagnostics(file_path):
    print(f"Loading {file_path}...")
    df = pd.read_csv(file_path)

    print("\n" + "=" * 40)
    print("      NEWS DATASET DIAGNOSTICS")
    print("=" * 40)

    # 1. Basic Stats
    print(f"Total Headlines: {len(df)}")

    # 2. Missing Values
    print("\n[1] Missing Values:")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "None found.")

    # 3. Duplicates
    print("\n[2] Duplicates:")
    exact_dupes = df.duplicated().sum()
    logical_dupes = df.duplicated(subset=["time", "title"]).sum()
    print(f"  Exact Duplicates: {exact_dupes}")
    print(f"  Logical Duplicates (Same Time + Title): {logical_dupes}")

    # 4. Time Range
    print("\n[3] Temporal Coverage (UTC):")
    df["time"] = pd.to_datetime(df["time"])
    print(f"  Start: {df['time'].min()}")
    print(f"  End:   {df['time'].max()}")

    # 5. Sentiment Distribution
    print("\n[4] Sentiment Distribution:")
    print(df["sentiment_label"].value_counts())

    # 6. Quality Checks
    print("\n[5] Content Quality Checks:")
    empty_titles = df["title"].apply(lambda x: str(x).strip() == "").sum()
    short_titles = df["title"].apply(lambda x: len(str(x)) < 10).sum()
    print(f"  Empty Titles: {empty_titles}")
    print(f"  Extremely Short Titles (<10 chars): {short_titles}")

    # 7. Sample of relevant tickers (if available)
    if "tickers" in df.columns:
        print("\n[6] Ticker Mentions (Top 5):")
        # Handle cases where tickers might be NaN or mixed types
        tickers = df["tickers"].dropna().astype(str).str.split(",").explode()
        print(tickers.value_counts().head(5))

    print("=" * 40)


if __name__ == "__main__":
    run_diagnostics("data/raw_sentiment/news_EURUSD_Full_2021_2026.csv")
