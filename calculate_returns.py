"""
Step 2: Calculate Daily Returns
Calculate daily percentage returns for BTC-USD and ETH-USD
"""

import pandas as pd
from fetch_data import fetch_crypto_prices

def calculate_daily_returns(prices_df):
    """
    Calculate daily percentage returns from price data
    
    Args:
        prices_df: DataFrame with closing prices
        
    Returns:
        DataFrame with daily returns (percent change)
    """
    print("Calculating daily returns...")
    
    # Calculate percentage change (daily returns)
    returns_df = prices_df.pct_change()
    
    # Drop missing values (first row will be NaN)
    returns_df = returns_df.dropna()
    
    print(f"✅ Calculated returns for {len(returns_df)} days")
    
    return returns_df


if __name__ == "__main__":
    # Fetch price data
    tickers = ['BTC-USD', 'ETH-USD']
    prices_df = fetch_crypto_prices(tickers, days=365)
    
    # Calculate returns
    returns_df = calculate_daily_returns(prices_df)
    
    # Display summary
    print("\n" + "="*60)
    print("DAILY RETURNS SUMMARY")
    print("="*60)
    print(f"\nShape: {returns_df.shape[0]} rows × {returns_df.shape[1]} columns")
    print(f"\nFirst 10 returns:")
    print(returns_df.head(10))
    print(f"\nLast 10 returns:")
    print(returns_df.tail(10))
    
    print(f"\nStatistics (in decimal form):")
    print(returns_df.describe())
    
    print(f"\nMean Daily Returns (%):")
    for col in returns_df.columns:
        print(f"  {col}: {returns_df[col].mean() * 100:.4f}%")
    
    print(f"\nDaily Volatility (Standard Deviation %):")
    for col in returns_df.columns:
        print(f"  {col}: {returns_df[col].std() * 100:.4f}%")
    
    # Save to CSV
    returns_df.to_csv('crypto_returns.csv')
    print(f"\n💾 Returns saved to 'crypto_returns.csv'")
