"""
Step 3: Calculate Value at Risk (VaR)
Using J.P. Morgan's RiskMetrics Parametric Method (Variance-Covariance)
"""

import numpy as np
import pandas as pd
from scipy import stats
from fetch_data import fetch_crypto_prices
from calculate_returns import calculate_daily_returns


def calculate_portfolio_var(returns_df, portfolio_values, confidence_level=0.95):
    """
    Calculate Value at Risk using the Variance-Covariance method
    
    Args:
        returns_df: DataFrame with daily returns for each asset
        portfolio_values: Dictionary with asset values (e.g., {'BTC-USD': 1000, 'ETH-USD': 1000})
        confidence_level: Confidence level (default 95%)
        
    Returns:
        Dictionary with VaR calculation details
    """
    print("\n" + "="*60)
    print("RISKMETRICS VAR CALCULATION")
    print("="*60)
    
    # Step 1: Portfolio setup
    total_portfolio_value = sum(portfolio_values.values())
    weights = np.array([portfolio_values[col] / total_portfolio_value for col in returns_df.columns])
    
    print(f"\nPortfolio Composition:")
    for i, col in enumerate(returns_df.columns):
        print(f"  {col}: ${portfolio_values[col]:,.2f} ({weights[i]*100:.1f}%)")
    print(f"  Total: ${total_portfolio_value:,.2f}")
    
    # Step 2: Calculate Covariance Matrix
    cov_matrix = returns_df.cov()
    print(f"\n📊 Covariance Matrix:")
    print(cov_matrix)
    
    # Step 3: Calculate Portfolio Variance and Standard Deviation
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix.values, weights))
    portfolio_std = np.sqrt(portfolio_variance)
    
    print(f"\n📈 Portfolio Statistics:")
    print(f"  Portfolio Variance: {portfolio_variance:.8f}")
    print(f"  Portfolio Std Dev (Daily): {portfolio_std:.6f} ({portfolio_std*100:.4f}%)")
    
    # Step 4: Calculate mean portfolio return
    mean_returns = returns_df.mean().values
    portfolio_mean = np.dot(weights, mean_returns)
    print(f"  Portfolio Mean Return: {portfolio_mean:.6f} ({portfolio_mean*100:.4f}%)")
    
    # Step 5: Calculate z-score for confidence level
    z_score = stats.norm.ppf(confidence_level)
    print(f"\n🎯 Confidence Level: {confidence_level*100}%")
    print(f"  Z-Score: {z_score:.4f}")
    
    # Step 6: Calculate VaR (for 1 day / 24 hours)
    # VaR = Portfolio Value × (z-score × portfolio_std - portfolio_mean)
    var_percentage = z_score * portfolio_std - portfolio_mean
    var_dollar = total_portfolio_value * var_percentage
    
    print(f"\n💰 VALUE AT RISK (24 hours):")
    print(f"  VaR (percentage): {var_percentage*100:.4f}%")
    print(f"  VaR (dollar): ${var_dollar:,.2f}")
    
    print(f"\n💡 INTERPRETATION:")
    print(f"  With {confidence_level*100}% confidence, your portfolio will NOT lose")
    print(f"  more than ${var_dollar:,.2f} in the next 24 hours.")
    print(f"\n  There is a {(1-confidence_level)*100}% chance you could lose more than this amount.")
    
    # Correlation matrix
    corr_matrix = returns_df.corr()
    print(f"\n🔗 Correlation Matrix:")
    print(corr_matrix)
    
    return {
        'portfolio_value': total_portfolio_value,
        'weights': weights,
        'covariance_matrix': cov_matrix,
        'portfolio_std': portfolio_std,
        'portfolio_mean': portfolio_mean,
        'z_score': z_score,
        'var_percentage': var_percentage,
        'var_dollar': var_dollar,
        'confidence_level': confidence_level,
        'correlation_matrix': corr_matrix
    }


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
    
    # Calculate VaR
    var_results = calculate_portfolio_var(returns_df, portfolio, confidence_level=0.95)
    
    # Save results
    results_summary = pd.DataFrame({
        'Metric': [
            'Total Portfolio Value',
            'Portfolio Daily Volatility (%)',
            'Portfolio Mean Return (%)',
            'Confidence Level (%)',
            'Z-Score',
            'VaR (24 hours) $',
            'VaR (24 hours) %'
        ],
        'Value': [
            f"${var_results['portfolio_value']:,.2f}",
            f"{var_results['portfolio_std']*100:.4f}%",
            f"{var_results['portfolio_mean']*100:.4f}%",
            f"{var_results['confidence_level']*100}%",
            f"{var_results['z_score']:.4f}",
            f"${var_results['var_dollar']:,.2f}",
            f"{var_results['var_percentage']*100:.4f}%"
        ]
    })
    
    results_summary.to_csv('var_results.csv', index=False)
    print(f"\n💾 Results saved to 'var_results.csv'")
