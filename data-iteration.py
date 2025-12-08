import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os


SYMBOL = "RELIANCE.NS"  
INTERVALS = ["1m", "5m", "1h"]
OUTPUT_DIR = "./ohlcv_data"


if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created directory: {OUTPUT_DIR}")

def fetch_ohlcv_data(symbol, interval, period="30d"):
  
    try:
        print(f"\nFetching {interval} data for {symbol}...")
        data = yf.download(symbol, interval=interval, period=period, progress=False)
        
        if data.empty:
            print(f"  ⚠ No data found for {symbol} with interval {interval}")
            return None
        
        print(f"  ✓ Fetched {len(data)} records for {interval}")
        return data
    
    except Exception as e:
        print(f"  ✗ Error fetching {interval} data: {str(e)}")
        return None

def save_to_csv(dataframe, symbol, interval, output_dir):
    """
    Save OHLCV dataframe to CSV file.
    
    Args:
        dataframe (pd.DataFrame): OHLCV data
        symbol (str): Stock symbol
        interval (str): Data interval
        output_dir (str): Output directory path
    
    Returns:
        str: Path to saved CSV file
    """
    if dataframe is None or dataframe.empty:
        return None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{symbol.replace('.', '_')}_{interval}_{timestamp}.csv"
    filepath = os.path.join(output_dir, filename)
    
    try:
        dataframe.to_csv(filepath)
        print(f"  ✓ Saved to: {filepath}")
        return filepath
    
    except Exception as e:
        print(f"  ✗ Error saving CSV: {str(e)}")
        return None

def iterate_and_process_ohlcv(symbol, intervals, output_dir):
    
    print("=" * 60)
    print(f"OHLCV Data Fetcher - {symbol}")
    print("=" * 60)
    
    results = {}
    
    for interval in intervals:
        print(f"\n[{interval.upper()}] Processing...")
        
        # Fetch data
        data = fetch_ohlcv_data(symbol, interval)
        
        if data is not None:
            # Save to CSV
            filepath = save_to_csv(data, symbol, interval, output_dir)
            
            if filepath:
                # Handle both regular and MultiIndex columns
                columns = [str(col) for col in data.columns]
                results[interval] = {
                    'filepath': filepath,
                    'records': len(data),
                    'columns': columns
                }
                
                # Display data preview
                print(f"\n  Data Preview ({interval}):")
                print(f"  {data.head(3).to_string()}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for interval, info in results.items():
        print(f"\n{interval.upper()}:")
        print(f"  Records: {info['records']}")
        print(f"  File: {info['filepath']}")
        print(f"  Columns: {', '.join(info['columns'])}")
    
    if not results:
        print("\n⚠ No data was successfully processed.")
    else:
        print(f"\n✓ Successfully processed {len(results)} interval(s)")

if __name__ == "__main__":
    # Run the data iteration and processing
    iterate_and_process_ohlcv(SYMBOL, INTERVALS, OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60)
