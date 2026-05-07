import pandas as pd
import os

def polish_news_dataset(input_path, output_path):
    print(f"Polishing {input_path}...")
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    df = pd.read_csv(input_path)
    initial_len = len(df)
    
    # 1. Boilerplate / Junk Text Detection
    # List of keywords that typically indicate marketing fluff or non-news content
    junk_keywords = [
        'join us', 'sign up', 'follow us', 'free guide', 'webinar', 
        'video analysis', 'youtube', 'facebook', 'twitter', 'telegram',
        'subscribe', 'check out', 'newsletter', 'watch now', 'dailyfx'
    ]
    
    def is_suspicious(row):
        title = str(row['title']).lower()
        text = str(row['text']).lower()
        source = str(row['source']).lower() if 'source' in row else ""
        
        # Rule A: Explicit Source Issue (DailyFX)
        if 'dailyfx' in source:
            return True
        
        # Rule B: Marketing Keywords in Text
        if any(key in text for key in junk_keywords):
            return True
            
        # Rule C: Text is too short to be useful (less than 40 chars)
        if len(text) < 40:
            return True
            
        # Rule D: Text is just a duplicate of the title (no extra info)
        if text.strip() == title.strip():
            return True
            
        return False

    # Apply the fix: For suspicious rows, replace text with title
    suspicious_mask = df.apply(is_suspicious, axis=1)
    df.loc[suspicious_mask, 'text'] = df.loc[suspicious_mask, 'title']
    corrected_count = suspicious_mask.sum()

    # 2. Fill remaining Missing Text with the Title (for those 22 rows)
    df['text'] = df['text'].fillna(df['title'])
    
    # 3. Drop Unnecessary Columns to keep the dataset lean
    cols_to_drop = ['source', 'sentiment_label', 'url', 'topics', 'type', 'tickers']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # 4. Remove Junk (Titles < 10 chars)
    # This filters out non-informative or corrupted headlines.
    df = df[df['title'].str.len() >= 10]
    removed = initial_len - len(df)
    
    # 3. Final Chronological Sort (Ascending for Time-Series)
    # Collector downloads newest first; training needs oldest first.
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time', ascending=True)
    
    # Save the cleaned version
    df.to_csv(output_path, index=False)
    
    print("\n" + "="*40)
    print("      POLISHING COMPLETE")
    print("="*40)
    print(f"Initial Rows:     {initial_len}")
    print(f"Removed Junk:     {removed}")
    print(f"Final Row Count:  {len(df)}")
    print(f"Cleaned File:     {output_path}")
    print("="*40)

if __name__ == "__main__":
    raw_file = 'data/raw_sentiment/news_EURUSD_Full_2021_2026.csv'
    clean_file = 'data/raw_sentiment/news_EURUSD_Cleaned.csv'
    polish_news_dataset(raw_file, clean_file)
