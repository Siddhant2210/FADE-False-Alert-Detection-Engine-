import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import os
import glob
from datetime import datetime

def get_latest_csv(pattern):
    """
    Get the latest CSV file matching the pattern.
    
    Args:
        pattern (str): Glob pattern to match files
    
    Returns:
        str: Path to the latest file, or None if no files found
    """
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getctime)

def load_ohlcv_data(csv_path):
    """
    Load OHLCV data from CSV file.
    
    Args:
        csv_path (str): Path to CSV file
    
    Returns:
        pd.DataFrame: OHLCV data with datetime index
    """
    try:
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        
        # Convert OHLCV columns to numeric
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        print(f"✓ Loaded {len(df)} records from {os.path.basename(csv_path)}")
        return df
    except Exception as e:
        print(f"✗ Error loading {csv_path}: {str(e)}")
        return None

def plot_candlestick(ax, data, title, color_up='green', color_down='red'):
    """
    Plot candlestick chart on given axes.
    
    Args:
        ax: Matplotlib axes object
        data (pd.DataFrame): OHLCV data with columns Open, High, Low, Close
        title (str): Title for the subplot
        color_up (str): Color for up candles
        color_down (str): Color for down candles
    """
    if data is None or data.empty:
        ax.text(0.5, 0.5, 'No data available', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=12)
        ax.set_title(title)
        return
    
    # Ensure we have required columns (handle both lowercase and uppercase)
    required_cols = ['Open', 'High', 'Low', 'Close']
    data_cols = data.columns.tolist()
    
    # Normalize column names to handle case variations
    for col in required_cols:
        if col not in data.columns:
            # Check for case-insensitive match
            for data_col in data_cols:
                if data_col.lower() == col.lower():
                    data[col] = data[data_col]
                    break
    
    # Extract OHLC data and drop NaN values
    df_clean = data[['Open', 'High', 'Low', 'Close']].copy()
    df_clean = df_clean.dropna()
    
    if df_clean.empty:
        ax.text(0.5, 0.5, 'No valid data available', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=12)
        ax.set_title(title)
        return
    
    opens = df_clean['Open']
    highs = df_clean['High']
    lows = df_clean['Low']
    closes = df_clean['Close']
    
    # X-axis positions
    x_pos = range(len(opens))
    width = 0.6
    
    # Plot high-low lines
    for i, (high, low) in enumerate(zip(highs, lows)):
        ax.plot([i, i], [low, high], color='black', linewidth=0.5)
    
    # Plot candles
    for i, (open_price, close_price, high, low) in enumerate(zip(opens, closes, highs, lows)):
        color = color_up if close_price >= open_price else color_down
        body_height = abs(close_price - open_price)
        body_bottom = min(open_price, close_price)
        
        rect = Rectangle((i - width/2, body_bottom), width, body_height,
                         facecolor=color, edgecolor='black', linewidth=0.5)
        ax.add_patch(rect)
    
    # Configure axes
    ax.set_xlim(-1, len(opens))
    ax.set_ylim(min(lows) * 0.99, max(highs) * 1.01)
    ax.set_xlabel('Time Period')
    ax.set_ylabel('Price (₹)')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Set x-axis ticks (show every 10th label to avoid crowding)
    tick_step = max(1, len(opens) // 10)
    ax.set_xticks(range(0, len(opens), tick_step))
    
    if len(opens) > 0:
        index_labels = [str(idx).split(' ')[-1][:5] for idx in opens.index]  # Show time
        ax.set_xticklabels([index_labels[i] if i < len(index_labels) else '' 
                           for i in range(0, len(opens), tick_step)], rotation=45)
    
    # Add legend
    up_patch = mpatches.Patch(color=color_up, label='Up')
    down_patch = mpatches.Patch(color=color_down, label='Down')
    ax.legend(handles=[up_patch, down_patch], loc='upper left', fontsize=9)

def plot_ohlcv_data(data_5m, data_1h, output_dir=None):
    """
    Create separate candlestick plots for 5m and 1h data.
    
    Args:
        data_5m (pd.DataFrame): 5-minute OHLCV data
        data_1h (pd.DataFrame): 1-hour OHLCV data
        output_dir (str): Optional directory path to save the plots
    """
    # Plot 5-minute data
    fig1, ax1 = plt.subplots(figsize=(14, 6))
    plot_candlestick(ax1, data_5m, '5-Minute Candlestick Chart (RELIANCE.NS)',
                     color_up='green', color_down='red')
    plt.tight_layout()
    
    if output_dir:
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_5m = os.path.join(output_dir, f"candlestick_5m_{timestamp}.png")
            fig1.savefig(file_5m, dpi=150, bbox_inches='tight')
            print(f"✓ 5m plot saved to: {file_5m}")
        except Exception as e:
            print(f"✗ Error saving 5m plot: {str(e)}")
    
    plt.show()
    
    # Plot 1-hour data
    fig2, ax2 = plt.subplots(figsize=(14, 6))
    plot_candlestick(ax2, data_1h, '1-Hour Candlestick Chart (RELIANCE.NS)',
                     color_up='green', color_down='red')
    plt.tight_layout()
    
    if output_dir:
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_1h = os.path.join(output_dir, f"candlestick_1h_{timestamp}.png")
            fig2.savefig(file_1h, dpi=150, bbox_inches='tight')
            print(f"✓ 1h plot saved to: {file_1h}")
        except Exception as e:
            print(f"✗ Error saving 1h plot: {str(e)}")
    
    plt.show()

def main():
    """Main function to load data and plot candlestick charts."""
    
    print("=" * 60)
    print("OHLCV Candlestick Plotter")
    print("=" * 60)
    
    # Define data directory
    data_dir = "./ohlcv_data"
    
    # Find latest CSV files
    print("\nSearching for latest data files...")
    csv_5m = get_latest_csv(os.path.join(data_dir, "*_5m_*.csv"))
    csv_1h = get_latest_csv(os.path.join(data_dir, "*_1h_*.csv"))
    
    if not csv_5m:
        print("✗ No 5-minute data file found")
        return
    
    if not csv_1h:
        print("✗ No 1-hour data file found")
        return
    
    print(f"\n5-minute file: {os.path.basename(csv_5m)}")
    print(f"1-hour file: {os.path.basename(csv_1h)}")
    
    # Load data
    print("\nLoading data...")
    data_5m = load_ohlcv_data(csv_5m)
    data_1h = load_ohlcv_data(csv_1h)
    
    if data_5m is None or data_1h is None:
        print("✗ Failed to load one or more data files")
        return
    
    # Generate plots
    print("\nGenerating candlestick charts...")
    
    plot_ohlcv_data(data_5m, data_1h, data_dir)
    
    print("\n" + "=" * 60)
    print("✓ Plotting complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
