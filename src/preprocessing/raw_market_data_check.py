import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/raw_market/EURUSD_H1_20200101_20260131.csv")
df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)

plt.figure(figsize=(12,6))
plt.plot(df['time'], df['close'])
plt.title("EURUSD Raw Data Check")
plt.show()
