# RiskMetrics - Value at Risk Calculator

Implementation of J.P. Morgan's RiskMetrics methodology using the **Variance-Covariance method** to calculate Value at Risk (VaR).

## What is Value at Risk?

VaR answers: **"With X% confidence, I will not lose more than $Y in one day"**

For example, a 95% VaR of $50,000 means:
- There's a 95% chance your loss won't exceed $50,000
- There's a 5% chance you could lose more than $50,000

## The Math

The Variance-Covariance method assumes returns follow a **Normal Distribution** (bell curve):

```
VaR = Portfolio Value × (μ - z × σ) × √t
```

Where:
- `μ` = mean return
- `z` = z-score for confidence level (1.645 for 95%)
- `σ` = standard deviation of returns
- `t` = time horizon in days

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

Run the crypto VaR analysis with real market data:

```bash
python risk_metrics.py
```

This will:
- Fetch real Bitcoin, Ethereum, and BNB data from Yahoo Finance
- Calculate VaR for single asset and portfolio
- Generate visualization charts
- Show correlation analysis

### Crypto VaR with Real Data

```python
from risk_metrics import RiskMetrics, fetch_crypto_data

# Fetch Bitcoin data
data = fetch_crypto_data('BTC-USD', period='1y')
returns = data['Returns'].dropna()

# Calculate VaR
rm = RiskMetrics(returns, confidence_level=0.95)
var = rm.calculate_var(portfolio_value=100_000, time_horizon=1)

print(f"Bitcoin VaR: ${var:,.2f}")
```

### Crypto Portfolio VaR

```python
from risk_metrics import RiskMetrics, fetch_crypto_data
import pandas as pd

# Fetch multiple cryptos
tickers = ['BTC-USD', 'ETH-USD', 'BNB-USD']
returns_dict = {}
for ticker in tickers:
    data = fetch_crypto_data(ticker, period='1y')
    returns_dict[ticker] = data['Returns'].dropna()

returns_df = pd.DataFrame(returns_dict).dropna()

# Portfolio weights (must sum to 1)
weights = [0.5, 0.3, 0.2]  # 50% BTC, 30% ETH, 20% BNB

rm = RiskMetrics(returns_df, confidence_level=0.95)
var = rm.calculate_portfolio_var(weights, portfolio_value=100_000)
```

## Key Features

- ✅ **Real crypto data** via yfinance (Bitcoin, Ethereum, etc.)
- ✅ Single asset VaR calculation
- ✅ Multi-asset portfolio VaR with correlation analysis
- ✅ Configurable confidence levels (90%, 95%, 99%)
- ✅ Multiple time horizons
- ✅ **Visualization charts** with matplotlib
- ✅ Based on J.P. Morgan RiskMetrics methodology

## Libraries Used (Industry Standard)

- **yfinance** - Free crypto/stock market data
- **numpy & pandas** - Mathematical calculations and data handling
- **scipy** - Statistical z-score calculations
- **matplotlib** - Chart visualization

## Assumptions

- Returns follow a **normal distribution**
- Historical volatility predicts future volatility
- Linear relationship between risk factors
