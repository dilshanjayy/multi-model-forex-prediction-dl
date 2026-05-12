import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_tbm_plot():
    # Load actual processed data
    feat_df = pd.read_parquet('data/processed_market/technical_features.parquet')
    meta_df = pd.read_parquet('data/processed_market/metadata.parquet')
    
    # Merge them on index to avoid duplicate columns
    df = pd.merge(meta_df, feat_df, left_index=True, right_index=True)
    
    # Drop duplicate 'time' column if it exists as time_x and time_y
    if 'time_x' in df.columns:
        df['time'] = df['time_x']
        df = df.drop(columns=['time_x', 'time_y'])

    # Reset index if time is the index, or use time column
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], utc=True)
    else:
        df = df.reset_index()
        df['time'] = pd.to_datetime(df['time'], utc=True)

    # Find a good example window where a barrier is hit
    # Let's just pick a specific starting index where volatility was decent
    start_idx = 500
    
    # We will plot 24 hours of data to show before, during, and after the horizon
    plot_window = df.iloc[start_idx-5 : start_idx+20].copy()
    
    entry_row = df.iloc[start_idx]
    entry_time = entry_row['time']
    entry_price = entry_row['Close']
    atr = entry_row['ATRr_14']
    multiplier = 3.0
    horizon = 12

    upper_barrier = entry_price + (atr * multiplier)
    lower_barrier = entry_price - (atr * multiplier)
    
    # The exact time of the vertical barrier
    end_row = df.iloc[start_idx + horizon]
    end_time = end_row['time']

    # Create the plot
    plt.style.use('dark_background') # Matches your dashboard aesthetic
    plt.figure(figsize=(10, 6))
    
    # Plot the price line
    plt.plot(plot_window['time'], plot_window['Close'], color='#58a6ff', linewidth=2.5, label='EUR/USD Close Price')
    
    # Draw Barriers starting from entry time
    plt.hlines(y=upper_barrier, xmin=entry_time, xmax=end_time, color='#3fb950', linestyle='dashed', linewidth=2, label='Upper Barrier (+3.0x ATR)')
    plt.hlines(y=lower_barrier, xmin=entry_time, xmax=end_time, color='#f85149', linestyle='dashed', linewidth=2, label='Lower Barrier (-3.0x ATR)')
    plt.vlines(x=end_time, ymin=lower_barrier, ymax=upper_barrier, color='#8b949e', linestyle='dotted', linewidth=2, label='Vertical Barrier (12 Hours)')

    # Mark the entry point
    plt.scatter(entry_time, entry_price, color='#ffffff', s=100, zorder=5, label='Trade Entry (t=0)')

    # Formatting
    plt.title('Triple Barrier Method (TBM) - Real Data Example', fontsize=16, pad=20, color='white')
    plt.xlabel('Time (UTC)', fontsize=12, color='#c9d1d9')
    plt.ylabel('Price', fontsize=12, color='#c9d1d9')
    plt.legend(loc='best', frameon=True, facecolor='#0d1117', edgecolor='#30363d')
    plt.grid(True, alpha=0.1, color='#8b949e')
    
    # Ensure report directory exists
    os.makedirs('report', exist_ok=True)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('report/tbm_real_graph.png', dpi=300, bbox_inches='tight', facecolor='#0d1117')
    print("Graph successfully generated and saved to report/tbm_real_graph.png")

if __name__ == '__main__':
    generate_tbm_plot()
