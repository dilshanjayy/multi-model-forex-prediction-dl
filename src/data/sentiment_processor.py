import os
import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# Set device: Use GPU if available for 10x speedup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global model cache to avoid reloading in live mode
SENTIMENT_MODEL = None
SENTIMENT_TOKENIZER = None

def get_sentiment_engine():
    """Singleton-style getter for FinBERT model."""
    global SENTIMENT_MODEL, SENTIMENT_TOKENIZER
    if SENTIMENT_MODEL is None:
        model_name = "ProsusAI/finbert"
        SENTIMENT_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        SENTIMENT_MODEL = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        SENTIMENT_MODEL.eval()
    return SENTIMENT_MODEL, SENTIMENT_TOKENIZER

def score_headlines(titles, texts):
    """
    Scores a list of headlines and returns raw probabilities.
    """
    model, tokenizer = get_sentiment_engine()
    
    combined = [f"{t}. {tx}" for t, tx in zip(titles, texts)]
    
    inputs = tokenizer(
        combined, 
        padding=True, 
        truncation=True, 
        max_length=512, 
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return probs.cpu().numpy()

def run_sentiment_analysis(input_path, batch_size=64):
    """
    Runs FinBERT on the cleaned news dataset and returns probabilities.
    """
    print(f"Loading dataset: {input_path}")
    df = pd.read_csv(input_path)
    
    # Concatenate Title and Text for maximum context
    df['combined_text'] = df['title'] + ". " + df['text']
    texts = df['combined_text'].tolist()

    print(f"Initializing FinBERT on {device}...")
    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()

    results = []

    print(f"Processing {len(texts)} headlines in batches of {batch_size}...")
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i : i + batch_size]
        
        # Tokenize
        inputs = tokenizer(
            batch_texts, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        ).to(device)

        # Inference
        with torch.no_grad():
            outputs = model(**inputs)
            # FinBERT output order: [Positive, Negative, Neutral]
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            results.append(probs.cpu().numpy())

    # Combine all batch results
    all_probs = np.vstack(results)
    
    # Add to dataframe
    df['sent_pos'] = all_probs[:, 0]
    df['sent_neg'] = all_probs[:, 1]
    df['sent_neu'] = all_probs[:, 2]
    
    # Calculate a single composite score (-1 to +1)
    # Score = Pos_prob - Neg_prob
    df['sentiment_score'] = df['sent_pos'] - df['sent_neg']
    
    return df

def aggregate_to_h1(df, output_path):
    """
    Resamples news sentiment to an hourly (H1) time-series.
    """
    print("Aggregating sentiment to Hourly (H1) time-series...")
    
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    # Aggregate by hour: Take the mean of sentiment scores
    # This handles hours with multiple news events naturally.
    h1_sentiment = df[['sent_pos', 'sent_neg', 'sent_neu', 'sentiment_score']].resample('H').mean()
    
    # Fill hours with NO news as Neutral (score = 0, neutral prob = 1.0)
    h1_sentiment['sent_neu'] = h1_sentiment['sent_neu'].fillna(1.0)
    h1_sentiment = h1_sentiment.fillna(0.0)
    
    # Save as Parquet for the ML Pipeline
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    h1_sentiment.reset_index().to_parquet(output_path, index=False)
    
    print(f"Aggregation complete. Final features saved to: {output_path}")
    print(f"Total Hours in Time-Series: {len(h1_sentiment)}")

if __name__ == "__main__":
    # Local paths (will be adjusted by user if running on Kaggle)
    input_file = 'data/raw_sentiment/news_EURUSD_Cleaned.csv'
    output_file = 'data/raw_sentiment/news_with_sentiment_scores.csv'
    
    # 1. Run FinBERT Scoring
    # This is the heavy lifting part intended for Kaggle/Colab GPU
    enriched_df = run_sentiment_analysis(input_file, batch_size=128)
    
    # 2. Save the RAW scores (Future-Proof for any timeframe)
    enriched_df.to_csv(output_file, index=False)
    print(f"\n--- SUCCESS ---")
    print(f"Full enriched dataset saved to: {output_file}")
    print("You can now download this file and aggregate it to any timeframe (M15, H1, etc.) locally.")
