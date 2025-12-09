"""
Step 4: Visualize Value at Risk
Create distribution plot with VaR cutoff line
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from fetch_data import fetch_crypto_prices
from calculate_returns import calculate_daily_returns
from calculate_var import calculate_portfolio_var


def plot_var_distribution(returns_df, portfolio_values, confidence_level=0.95):
    """
    Plot portfolio returns distribution with VaR cutoff line
    
    Args:
        returns_df: DataFrame with daily returns
        portfolio_values: Dictionary with asset values
        confidence_level: Confidence level for VaR
    """
    # Calculate portfolio returns
    total_value = sum(portfolio_values.values())
    weights = np.array([portfolio_values[col] / total_value for col in returns_df.columns])
    portfolio_returns = (returns_df * weights).sum(axis=1)
    
    # Calculate VaR threshold
    z_score = stats.norm.ppf(confidence_level)
    portfolio_mean = portfolio_returns.mean()
    portfolio_std = portfolio_returns.std()
    var_threshold = portfolio_mean - z_score * portfolio_std
    
    # Create the plot
    plt.figure(figsize=(12, 7))
    
    # Plot histogram
    n, bins, patches = plt.hist(portfolio_returns, bins=50, alpha=0.7, 
                                 color='skyblue', edgecolor='black', density=True)
    
    # Overlay normal distribution curve
    mu = portfolio_returns.mean()
    sigma = portfolio_returns.std()
    x = np.linspace(portfolio_returns.min(), portfolio_returns.max(), 100)
    plt.plot(x, stats.norm.pdf(x, mu, sigma), 'b-', linewidth=2, 
             label='Normal Distribution')
    
    # Draw mean line
    plt.axvline(portfolio_mean, color='green', linestyle='--', linewidth=2, 
                label=f'Mean Return: {portfolio_mean*100:.3f}%')
    
    # Draw VaR cutoff line (RED DASHED)
    plt.axvline(var_threshold, color='red', linestyle='--', linewidth=2.5, 
                label=f'95% VaR Cutoff: {var_threshold*100:.3f}%')
    
    # Shade the VaR region (5% tail)
    plt.axvspan(portfolio_returns.min(), var_threshold, alpha=0.2, color='red', 
                label=f'Risk Region ({(1-confidence_level)*100}%)')
    
    # Labels and title
    plt.xlabel('Daily Portfolio Returns', fontsize=12, fontweight='bold')
    plt.ylabel('Probability Density', fontsize=12, fontweight='bold')
    plt.title('Crypto Portfolio RiskMetrics', fontsize=16, fontweight='bold')
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(alpha=0.3)
    
    # Add text box with statistics
    stats_text = f'Portfolio Statistics:\n'
    stats_text += f'Mean: {portfolio_mean*100:.4f}%\n'
    stats_text += f'Std Dev: {portfolio_std*100:.4f}%\n'
    stats_text += f'VaR (95%): {var_threshold*100:.4f}%\n'
    stats_text += f'Days: {len(portfolio_returns)}'
    
    plt.text(0.98, 0.97, stats_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('var_distribution.png', dpi=150, bbox_inches='tight')
    print("\n📊 Chart saved as 'var_distribution.png'")
    plt.show()


if __name__ == "__main__":
    # Fetch data
    tickers = ['BTC-USD', 'ETH-USD']
    prices_df = fetch_crypto_prices(tickers, days=365)
    
    # Calculate returns
    returns_df = calculate_daily_returns(prices_df)
    
    # Define portfolio
    portfolio = {
        'BTC-USD': 1000,  # $1,000 in Bitcoin
        'ETH-USD': 1000   # $1,000 in Ethereum
    }
    
    # Calculate and display VaR
    var_results = calculate_portfolio_var(returns_df, portfolio, confidence_level=0.95)
    
    # Create visualization
    print("\n" + "="*60)
    print("CREATING VISUALIZATION")
    print("="*60)
    plot_var_distribution(returns_df, portfolio, confidence_level=0.95)
