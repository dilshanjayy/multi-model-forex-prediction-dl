import pandas as pd
from src.preprocessing.feature_engineering import engineer_technical_features

file = "GBPUSD_H1_20210101_20260101"

df = pd.read_csv(f"data/raw_market/{file}.csv")

clean_df = engineer_technical_features(df)

clean_df.to_csv(f"{file}_technical.csv", index=False)
