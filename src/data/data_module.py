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

    def prepare_dataset(self, components: list = ["technical_features", "targets"]) -> pd.DataFrame:
        """
        Joins multiple components into a single DataFrame on the 'time' index.
        """
        base_df = None

        for comp in components:
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
            raise ValueError(f"Dataset preparation failed. Final rows: {0 if base_df is None else len(base_df)}. Check if component timestamps match.")

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
