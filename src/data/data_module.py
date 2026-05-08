import pandas as pd
import os

class DataModule:
    """
    Industry-standard DataModule for handling Feature Store components.
    Responsible for joining technical features, sentiment, and targets.
    """
    def __init__(self, processed_dir: str):
        self.processed_dir = processed_dir

    def load_component(self, name: str) -> pd.DataFrame:
        """Loads a parquet component by name."""
        path = os.path.join(self.processed_dir, f"{name}.parquet")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Component {name} not found at {path}")
        return pd.read_parquet(path)

    def prepare_sentiment_component(self, target_index: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Loads the raw sentiment scores and resamples them to match 
        the target market data timeframe and index.
        """
        path = "data/raw_sentiment/news_with_sentiment_scores.csv"
        if not os.path.exists(path):
            print(f"Warning: Sentiment file not found at {path}. Returning empty neutral dataframe.")
            # Return a neutral dataframe matching the index
            neutral_df = pd.DataFrame(index=target_index)
            neutral_df['sent_pos'] = 0.0
            neutral_df['sent_neg'] = 0.0
            neutral_df['sent_neu'] = 1.0
            neutral_df['sentiment_score'] = 0.0
            return neutral_df

        print(f"Loading raw sentiment from {path}...")
        df = pd.read_csv(path)
        df['time'] = pd.to_datetime(df['time'], utc=True).dt.floor("s")
        df.set_index('time', inplace=True)

        # 1. Resample to the target frequency (detected from target_index)
        # We take the mean of sentiment in each bucket
        # Fallback to '1h' if frequency can't be inferred (standard for H1 data)
        freq = target_index.inferred_freq
        if freq is None:
            freq = "1h"
        
        # Standardize 'H' to '1h' for modern Pandas compatibility
        if freq == "H":
            freq = "1h"

        resampled = df[['sent_pos', 'sent_neg', 'sent_neu', 'sentiment_score']].resample(freq).mean()

        # 2. Reindex to match the EXACT market bars (fills gaps with NaN)
        # This ensures we have a sentiment value for every single price bar
        resampled = resampled.reindex(target_index)

        # 3. Fill NaNs (Empty hours/periods) as Neutral
        resampled['sent_neu'] = resampled['sent_neu'].fillna(1.0)
        resampled = resampled.fillna(0.0)

        print(f"Sentiment resampled to {freq}: {len(resampled)} rows aligned.")
        return resampled

    def prepare_dataset(self, components: list = ["technical_features", "targets"]) -> pd.DataFrame:
        """
        Joins multiple components into a single DataFrame on the 'time' index.
        Automatically handles 'sentiment' by resampling raw scores.
        """
        base_df = None
        sentiment_requested = "sentiment" in components
        
        # Filter out virtual components from the disk-loading loop
        disk_components = [c for c in components if c != "sentiment"]

        for comp in disk_components:
            df = self.load_component(comp)
            # FORCE UTC and normalize precision to 'ns' to ensure cross-OS compatibility
            df["time"] = pd.to_datetime(df["time"], utc=True).dt.floor("s") 
            df.set_index("time", inplace=True)

            if base_df is None:
                base_df = df
            else:
                # Join on time index to ensure alignment
                base_df = base_df.join(df, how="inner")

        if base_df is None or len(base_df) == 0:
            raise ValueError(f"Dataset preparation failed. Check if component timestamps match.")

        # Handle the virtual 'sentiment' component
        if sentiment_requested:
            sentiment_df = self.prepare_sentiment_component(base_df.index)
            base_df = base_df.join(sentiment_df, how="inner")

        base_df.sort_index(inplace=True)
        print(f"Dataset prepared successfully: {len(base_df)} rows.")
        return base_df

    @staticmethod
    def save_features(feature_dict: dict, output_dir: str):
        """Saves a dictionary of DataFrames to the Feature Store in Parquet format."""
        os.makedirs(output_dir, exist_ok=True)
        for name, df in feature_dict.items():
            path = os.path.join(output_dir, f"{name}.parquet")
            df.to_parquet(path, index=False)
            print(f"Saved component: {name} ({len(df)} rows) -> {path}")
