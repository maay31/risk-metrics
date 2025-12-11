"""
RiskMetrics - Value at Risk (VaR) Calculator
Using J.P. Morgan's Variance-Covariance Method
"""

import numpy as np
import pandas as pd
from scipy import stats
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


class RiskMetrics:
    """Calculate Value at Risk using the Variance-Covariance method"""
    
    def __init__(self, returns, confidence_level=0.95):
        """
        Initialize RiskMetrics calculator
        
        Args:
            returns: pandas DataFrame or Series of historical returns
            confidence_level: Confidence level for VaR (default 95%)
        """
        self.returns = returns if isinstance(returns, pd.DataFrame) else pd.Series(returns)
        self.confidence_level = confidence_level
        self.z_score = stats.norm.ppf(confidence_level)
        
    def calculate_var(self, portfolio_value, time_horizon=1):
        """
        Calculate Value at Risk
        
        Args:
            portfolio_value: Current portfolio value in dollars
            time_horizon: Time horizon in days (default 1)
            
        Returns:
            VaR in dollars
        """
        # Calculate mean and standard deviation of returns
        mean_return = self.returns.mean()
        std_return = self.returns.std()
        
        # VaR calculation: portfolio_value * (mean - z_score * std) * sqrt(time_horizon)
        var = portfolio_value * (mean_return - self.z_score * std_return) * np.sqrt(time_horizon)
        
        return abs(var)
    
    def calculate_portfolio_var(self, weights, portfolio_value, time_horizon=1):
        """
        Calculate VaR for a multi-asset portfolio
        
        Args:
            weights: Array of portfolio weights (must sum to 1)
            portfolio_value: Total portfolio value
            time_horizon: Time horizon in days
            
        Returns:
            Portfolio VaR in dollars
        """
        if not isinstance(self.returns, pd.DataFrame):
            raise ValueError("Multi-asset VaR requires DataFrame of returns")
        
        weights = np.array(weights)
        
        # Calculate portfolio return statistics
        mean_returns = self.returns.mean().values
        cov_matrix = self.returns.cov().values
        
        # Portfolio mean and std
        portfolio_mean = np.dot(weights, mean_returns)
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        # VaR calculation
        var = portfolio_value * (portfolio_mean - self.z_score * portfolio_std) * np.sqrt(time_horizon)
        
        return abs(var)
    
    def get_summary(self, portfolio_value, time_horizon=1):
        """
        Get a summary of risk metrics
        
        Returns:
            Dictionary with risk statistics
        """
        var = self.calculate_var(portfolio_value, time_horizon)
        
        return {
            'VaR': var,
            'Confidence Level': f"{self.confidence_level * 100}%",
            'Time Horizon': f"{time_horizon} day(s)",
            'Portfolio Value': portfolio_value,
            'Max Expected Loss': var,
            'Interpretation': f"With {self.confidence_level * 100}% confidence, "
                            f"you will not lose more than ${var:,.2f} in {time_horizon} day(s)"
        }


def example_single_asset():
    """Example: Single asset VaR calculation"""
    print("=" * 60)
    print("EXAMPLE 1: Single Asset VaR")
    print("=" * 60)
    
    # Generate sample returns (normally distributed)
    np.random.seed(42)
    daily_returns = np.random.normal(0.001, 0.02, 252)  # ~1 year of daily returns
    
    # Calculate VaR
    rm = RiskMetrics(daily_returns, confidence_level=0.95)
    portfolio_value = 1_000_000  # $1 million portfolio
    
    summary = rm.get_summary(portfolio_value, time_horizon=1)
    
    print(f"\nPortfolio Value: ${summary['Portfolio Value']:,.2f}")
    print(f"Confidence Level: {summary['Confidence Level']}")
    print(f"Time Horizon: {summary['Time Horizon']}")
    print(f"\n📊 Value at Risk (VaR): ${summary['VaR']:,.2f}")
    print(f"\n💡 {summary['Interpretation']}")
    print()


def example_portfolio():
    """Example: Multi-asset portfolio VaR"""
    print("=" * 60)
    print("EXAMPLE 2: Multi-Asset Portfolio VaR")
    print("=" * 60)
    
    # Generate sample returns for 3 assets
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    
    returns_data = {
        'Stock_A': np.random.normal(0.0008, 0.015, 252),
        'Stock_B': np.random.normal(0.0010, 0.020, 252),
        'Stock_C': np.random.normal(0.0005, 0.012, 252)
    }
    
    returns_df = pd.DataFrame(returns_data, index=dates)
    
    # Portfolio weights (must sum to 1)
    weights = [0.4, 0.35, 0.25]  # 40%, 35%, 25%
    portfolio_value = 1_000_000
    
    # Calculate portfolio VaR
    rm = RiskMetrics(returns_df, confidence_level=0.95)
    var = rm.calculate_portfolio_var(weights, portfolio_value, time_horizon=1)
    
    print(f"\nPortfolio Composition:")
    for i, (asset, weight) in enumerate(zip(returns_df.columns, weights)):
        print(f"  {asset}: {weight*100}% (${portfolio_value * weight:,.2f})")
    
    print(f"\nTotal Portfolio Value: ${portfolio_value:,.2f}")
    print(f"Confidence Level: 95%")
    print(f"Time Horizon: 1 day")
    print(f"\n📊 Portfolio VaR: ${var:,.2f}")
    print(f"\n💡 With 95% confidence, this portfolio will not lose more than ${var:,.2f} in one day")
    print()


def fetch_crypto_data(ticker, period='1y'):
    """
    Fetch cryptocurrency data from Yahoo Finance
    
    Args:
        ticker: Crypto ticker (e.g., 'BTC-USD', 'ETH-USD')
        period: Time period ('1mo', '3mo', '6mo', '1y', '2y', '5y')
        
    Returns:
        DataFrame with price data and calculated returns
    """
    print(f"Fetching {ticker} data...")
    data = yf.download(ticker, period=period, progress=False)
    data['Returns'] = data['Close'].pct_change().dropna()
    return data


def plot_var_analysis(returns, var, portfolio_value, confidence_level=0.95):
    """
    Visualize the VaR analysis with distribution plot
    
    Args:
        returns: Series of returns
        var: Calculated VaR value
        portfolio_value: Portfolio value
        confidence_level: Confidence level
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Returns distribution
    ax1.hist(returns, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(returns.mean(), color='green', linestyle='--', linewidth=2, label='Mean')
    ax1.axvline(returns.mean() - stats.norm.ppf(confidence_level) * returns.std(), 
                color='red', linestyle='--', linewidth=2, label=f'{confidence_level*100}% VaR Threshold')
    ax1.set_xlabel('Daily Returns')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Returns')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot 2: VaR visualization
    var_pct = (var / portfolio_value) * 100
    ax2.bar(['Portfolio Value', 'VaR (Max Loss)'], 
            [portfolio_value, var], 
            color=['green', 'red'], alpha=0.7)
    ax2.set_ylabel('Value ($)')
    ax2.set_title(f'Value at Risk ({confidence_level*100}% Confidence)')
    ax2.text(0, portfolio_value/2, f'${portfolio_value:,.0f}', 
             ha='center', va='center', fontsize=12, fontweight='bold')
    ax2.text(1, var/2, f'${var:,.0f}\n({var_pct:.2f}%)', 
             ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    ax2.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('var_analysis.png', dpi=150, bbox_inches='tight')
    print("\n📊 Chart saved as 'var_analysis.png'")
    st.pyplot()


def example_crypto_var():
    """Example: Real crypto VaR using yfinance"""
    print("=" * 60)
    print("CRYPTO VaR ANALYSIS - Real Market Data")
    print("=" * 60)
    
    # Fetch Bitcoin data
    ticker = 'BTC-USD'
    data = fetch_crypto_data(ticker, period='1y')
    returns = data['Returns'].dropna()
    
    # Portfolio setup
    portfolio_value = 100_000  # $100k portfolio
    confidence_level = 0.95
    
    # Calculate VaR
    rm = RiskMetrics(returns, confidence_level=confidence_level)
    summary = rm.get_summary(portfolio_value, time_horizon=1)
    
    # Display results
    print(f"\n{'Asset:':<20} {ticker}")
    print(f"{'Data Points:':<20} {len(returns)} days")
    print(f"{'Portfolio Value:':<20} ${summary['Portfolio Value']:,.2f}")
    print(f"{'Confidence Level:':<20} {summary['Confidence Level']}")
    print(f"{'Time Horizon:':<20} {summary['Time Horizon']}")
    print(f"\n{'Mean Daily Return:':<20} {returns.mean()*100:.4f}%")
    print(f"{'Daily Volatility:':<20} {returns.std()*100:.4f}%")
    print(f"\n{'📊 Value at Risk:':<20} ${summary['VaR']:,.2f}")
    print(f"\n💡 {summary['Interpretation']}")
    
    # Plot analysis
    plot_var_analysis(returns, summary['VaR'], portfolio_value, confidence_level)
    print()


def example_crypto_portfolio():
    """Example: Multi-crypto portfolio VaR"""
    print("=" * 60)
    print("CRYPTO PORTFOLIO VaR - Real Market Data")
    print("=" * 60)
    
    # Fetch data for multiple cryptos
    tickers = ['BTC-USD', 'ETH-USD', 'BNB-USD']
    returns_dict = {}
    
    for ticker in tickers:
        data = fetch_crypto_data(ticker, period='1y')
        returns_dict[ticker] = data['Returns'].dropna()
    
    # Align dates and create DataFrame
    returns_df = pd.DataFrame(returns_dict).dropna()
    
    # Portfolio setup
    weights = [0.5, 0.3, 0.2]  # 50% BTC, 30% ETH, 20% BNB
    portfolio_value = 100_000
    confidence_level = 0.95
    
    # Calculate portfolio VaR
    rm = RiskMetrics(returns_df, confidence_level=confidence_level)
    var = rm.calculate_portfolio_var(weights, portfolio_value, time_horizon=1)
    
    # Display results
    print(f"\n{'Portfolio Composition:'}")
    for ticker, weight in zip(tickers, weights):
        print(f"  {ticker:<10} {weight*100:>5.1f}% (${portfolio_value * weight:>10,.2f})")
    
    print(f"\n{'Total Portfolio:':<20} ${portfolio_value:,.2f}")
    print(f"{'Confidence Level:':<20} {confidence_level*100}%")
    print(f"{'Time Horizon:':<20} 1 day")
    print(f"{'Data Points:':<20} {len(returns_df)} days")
    
    # Portfolio statistics
    portfolio_returns = (returns_df * weights).sum(axis=1)
    print(f"\n{'Mean Daily Return:':<20} {portfolio_returns.mean()*100:.4f}%")
    print(f"{'Daily Volatility:':<20} {portfolio_returns.std()*100:.4f}%")
    
    print(f"\n{'📊 Portfolio VaR:':<20} ${var:,.2f}")
    print(f"\n💡 With {confidence_level*100}% confidence, this portfolio will not lose more than ${var:,.2f} in one day")
    
    # Correlation matrix
    print(f"\n{'Correlation Matrix:'}")
    print(returns_df.corr().round(3))
    print()


if __name__ == "__main__":
    # Run crypto examples with real market data
    example_crypto_var()
    example_crypto_portfolio()


