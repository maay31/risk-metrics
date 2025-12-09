"""
Step 5: Analyze Portfolio Diversification
Calculate correlation and provide diversification recommendations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fetch_data import fetch_crypto_prices
from calculate_returns import calculate_daily_returns


def analyze_correlation(returns_df):
    """
    Analyze correlation between assets and assess diversification
    
    Args:
        returns_df: DataFrame with daily returns for each asset
        
    Returns:
        Dictionary with correlation analysis
    """
    print("\n" + "="*60)
    print("PORTFOLIO DIVERSIFICATION ANALYSIS")
    print("="*60)
    
    # Calculate correlation matrix
    corr_matrix = returns_df.corr()
    
    print("\n📊 CORRELATION MATRIX:")
    print(corr_matrix)
    print()
    
    # Extract correlation between BTC and ETH
    btc_eth_corr = corr_matrix.iloc[0, 1]
    
    print(f"🔗 BTC-ETH Correlation: {btc_eth_corr:.4f}")
    print()
    
    # Interpret correlation strength
    print("📈 CORRELATION INTERPRETATION:")
    if abs(btc_eth_corr) >= 0.8:
        strength = "VERY HIGH"
        color = "🔴"
    elif abs(btc_eth_corr) >= 0.6:
        strength = "HIGH"
        color = "🟠"
    elif abs(btc_eth_corr) >= 0.4:
        strength = "MODERATE"
        color = "🟡"
    elif abs(btc_eth_corr) >= 0.2:
        strength = "LOW"
        color = "🟢"
    else:
        strength = "VERY LOW"
        color = "🟢"
    
    print(f"  {color} Correlation Strength: {strength}")
    print(f"  Direction: {'Positive (move together)' if btc_eth_corr > 0 else 'Negative (move opposite)'}")
    print()
    
    # Diversification assessment
    print("💡 DIVERSIFICATION ASSESSMENT:")
    print()
    
    if btc_eth_corr >= 0.8:
        assessment = "POORLY DIVERSIFIED ❌"
        explanation = """
  Your portfolio is POORLY DIVERSIFIED. BTC and ETH move very closely together.
  
  ⚠️  RISK: When Bitcoin drops, Ethereum will likely drop too.
  ⚠️  IMPACT: You're essentially doubling down on the same risk.
  
  RECOMMENDATION:
  • Add assets with LOW correlation (< 0.5) to BTC/ETH
  • Consider: Stablecoins, commodities, or different crypto sectors
  • Example alternatives: BNB, SOL, ADA, or traditional assets
        """
    elif btc_eth_corr >= 0.6:
        assessment = "MODERATELY DIVERSIFIED ⚠️"
        explanation = """
  Your portfolio has MODERATE diversification. BTC and ETH still move together often.
  
  ⚠️  RISK: Significant overlap in price movements.
  
  RECOMMENDATION:
  • Consider adding 1-2 assets with lower correlation
  • This will reduce portfolio volatility
  • Look for assets in different crypto sectors
        """
    else:
        assessment = "WELL DIVERSIFIED ✅"
        explanation = """
  Your portfolio is WELL DIVERSIFIED. BTC and ETH have relatively independent movements.
  
  ✅ BENEFIT: When one asset drops, the other may not follow.
  ✅ IMPACT: Lower overall portfolio risk.
  
  RECOMMENDATION:
  • Your current diversification is good
  • Continue monitoring correlation over time
  • Consider rebalancing if correlation increases
        """
    
    print(f"  Status: {assessment}")
    print(explanation)
    
    # Calculate portfolio metrics
    print("\n📊 PORTFOLIO RISK METRICS:")
    individual_vols = returns_df.std() * 100
    print(f"\n  Individual Asset Volatility (Daily %):")
    for col in returns_df.columns:
        print(f"    {col}: {individual_vols[col]:.4f}%")
    
    # Portfolio volatility with equal weights
    weights = np.array([0.5, 0.5])
    cov_matrix = returns_df.cov()
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix.values, weights))
    portfolio_vol = np.sqrt(portfolio_variance) * 100
    
    print(f"\n  Portfolio Volatility (50/50 split): {portfolio_vol:.4f}%")
    
    # Diversification benefit
    avg_individual_vol = individual_vols.mean()
    diversification_benefit = ((avg_individual_vol - portfolio_vol) / avg_individual_vol) * 100
    
    print(f"  Average Individual Volatility: {avg_individual_vol:.4f}%")
    print(f"  Diversification Benefit: {diversification_benefit:.2f}%")
    
    if diversification_benefit > 0:
        print(f"\n  ✅ Your portfolio volatility is {diversification_benefit:.2f}% lower than")
        print(f"     holding a single asset due to diversification.")
    else:
        print(f"\n  ⚠️  Limited diversification benefit due to high correlation.")
    
    return {
        'correlation_matrix': corr_matrix,
        'btc_eth_correlation': btc_eth_corr,
        'strength': strength,
        'assessment': assessment,
        'portfolio_volatility': portfolio_vol,
        'diversification_benefit': diversification_benefit
    }


def plot_correlation_heatmap(returns_df):
    """
    Create a visual heatmap of the correlation matrix
    
    Args:
        returns_df: DataFrame with daily returns
    """
    corr_matrix = returns_df.corr()
    
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn_r', center=0,
                square=True, linewidths=2, cbar_kws={"shrink": 0.8},
                vmin=-1, vmax=1, fmt='.4f', annot_kws={'size': 14, 'weight': 'bold'})
    
    plt.title('Correlation Matrix - Crypto Portfolio', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Assets', fontsize=12, fontweight='bold')
    plt.ylabel('Assets', fontsize=12, fontweight='bold')
    
    # Add interpretation text
    interpretation = """
    Correlation Guide:
    • 1.0 = Perfect positive correlation (move together)
    • 0.0 = No correlation (independent)
    • -1.0 = Perfect negative correlation (move opposite)
    
    For diversification: Look for correlations < 0.5
    """
    
    plt.text(0.5, -0.15, interpretation, transform=plt.gca().transAxes,
             fontsize=9, verticalalignment='top', horizontalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight')
    print("\n📊 Correlation heatmap saved as 'correlation_heatmap.png'")
    plt.show()


if __name__ == "__main__":
    # Fetch data
    tickers = ['BTC-USD', 'ETH-USD']
    prices_df = fetch_crypto_prices(tickers, days=365)
    
    # Calculate returns
    returns_df = calculate_daily_returns(prices_df)
    
    # Analyze correlation and diversification
    analysis = analyze_correlation(returns_df)
    
    # Create visualization
    print("\n" + "="*60)
    print("CREATING CORRELATION HEATMAP")
    print("="*60)
    plot_correlation_heatmap(returns_df)
    
    # Save analysis to file
    summary = pd.DataFrame({
        'Metric': [
            'BTC-ETH Correlation',
            'Correlation Strength',
            'Diversification Status',
            'Portfolio Volatility (%)',
            'Diversification Benefit (%)'
        ],
        'Value': [
            f"{analysis['btc_eth_correlation']:.4f}",
            analysis['strength'],
            analysis['assessment'],
            f"{analysis['portfolio_volatility']:.4f}%",
            f"{analysis['diversification_benefit']:.2f}%"
        ]
    })
    
    summary.to_csv('diversification_analysis.csv', index=False)
    print("\n💾 Analysis saved to 'diversification_analysis.csv'")
