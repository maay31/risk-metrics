"""
Step 1: Fetch Crypto Data
Get daily closing prices for BTC-USD and ETH-USD for the last 365 days
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def fetch_crypto_prices(tickers, days=365):
    """
    Fetch daily closing prices for cryptocurrencies
    
    Args:
        tickers: List of ticker symbols (e.g., ['BTC-USD', 'ETH-USD'])
        days: Number of days of historical data
        
    Returns:
        DataFrame with closing prices for each ticker
    """
    print(f"Fetching {days} days of data for {', '.join(tickers)}...")
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Fetch data for all tickers
    data = yf.download(tickers, start=start_date, end=end_date, progress=True)
    
    # Extract closing prices
    if len(tickers) == 1:
        prices = pd.DataFrame({tickers[0]: data['Close']})
    else:
        prices = data['Close']
    
    # Clean data
    prices = prices.dropna()
    
    print(f"\n✅ Successfully fetched {len(prices)} days of data")
    print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    
    return prices


if __name__ == "__main__":
    # Fetch BTC and ETH data
    tickers = ['BTC-USD', 'ETH-USD']
    prices_df = fetch_crypto_prices(tickers, days=365)
    
    # Display summary
    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)
    print(f"\nShape: {prices_df.shape[0]} rows × {prices_df.shape[1]} columns")
    print(f"\nFirst 5 rows:")
    print(prices_df.head())
    print(f"\nLast 5 rows:")
    print(prices_df.tail())
    print(f"\nBasic Statistics:")
    print(prices_df.describe())
    
    # Save to CSV
    prices_df.to_csv('crypto_prices.csv')
    print(f"\n💾 Data saved to 'crypto_prices.csv'")
