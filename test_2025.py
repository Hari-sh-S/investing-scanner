"""
Local test for 2025 with EMA 100 regime filter, GOLDBEES uncorrelated.
Uses real cached data to identify issues.
"""

import datetime
import pandas as pd
import numpy as np
from portfolio_engine import PortfolioEngine, DataCache
from nifty_universe import get_universe

print("=" * 70)
print("LOCAL TEST: 2025 BACKTEST WITH REGIME FILTER")
print("=" * 70)

# Settings matching user's configuration
START_DATE = datetime.date(2025, 1, 1)
END_DATE = datetime.date(2025, 12, 13)  # Today
INITIAL_CAPITAL = 100000
SCORING_FORMULA = "6 Month Performance / 6 Month Volatility"
NUM_STOCKS = 5

# Regime filter config (EMA 100, Go Cash)
REGIME_CONFIG = {
    'type': 'EMA',
    'value': 100,
    'action': 'Go Cash',
    'index': 'NIFTY 100'
}

# Uncorrelated asset config (GOLDBEES 100%)
UNCORRELATED_CONFIG = {
    'asset': 'GOLDBEES',
    'allocation_pct': 100
}

REBAL_CONFIG = {
    'frequency': 'Monthly',
    'date': 1,
    'alt_day': 'Next Day'
}

# Get universe
print("\n[1] Loading universe...")
universe = get_universe('NIFTY 100')
print(f"   Universe: NIFTY 100 with {len(universe)} stocks")

# Create engine
print("\n[2] Creating portfolio engine...")
engine = PortfolioEngine(
    universe,
    START_DATE,
    END_DATE,
    initial_capital=INITIAL_CAPITAL
)

# Fetch data from cache
print("\n[3] Loading data from cache...")
def progress_cb(current, total, ticker):
    if current % 20 == 0 or current == total:
        print(f"   Loading: {current}/{total}")

success = engine.fetch_data(progress_callback=progress_cb)
print(f"   Loaded {len(engine.data)} stocks with data")

if len(engine.data) < 10:
    print("   [WARN] Not enough stocks loaded. Check cache.")

# Show date ranges
print("\n[4] Checking data date ranges...")
date_issues = []
for ticker, df in list(engine.data.items())[:5]:
    if not df.empty:
        print(f"   {ticker}: {df.index[0].date()} to {df.index[-1].date()} ({len(df)} days)")
        if df.index[-1].date() < END_DATE - datetime.timedelta(days=30):
            date_issues.append(ticker)

if date_issues:
    print(f"   [WARN] Stale data for: {date_issues}")

# Run backtest
print("\n[5] Running backtest...")
print(f"   Formula: {SCORING_FORMULA}")
print(f"   Regime: EMA 100 on NIFTY 100, Action: Go Cash")
print(f"   Uncorrelated: GOLDBEES 100%")
print(f"   Rebalance: Monthly on day 1")

try:
    engine.run_rebalance_strategy(
        scoring_formula=SCORING_FORMULA,
        num_stocks=NUM_STOCKS,
        rebal_config=REBAL_CONFIG,
        exit_rank=None,
        regime_config=REGIME_CONFIG,
        uncorrelated_config=UNCORRELATED_CONFIG,
        reinvest_profits=True
    )
    print("   [OK] Backtest completed")
except Exception as e:
    print(f"   [FAIL] Backtest failed: {e}")
    import traceback
    traceback.print_exc()

# Analyze results
print("\n[6] Analyzing results...")
if engine.portfolio_df is not None and not engine.portfolio_df.empty:
    pf = engine.portfolio_df
    
    # Monthly summary
    print("\n   Monthly Portfolio Values:")
    monthly = pf.resample('ME').last()
    for idx, row in monthly.iterrows():
        print(f"   {idx.strftime('%Y-%m')}: ${row['Portfolio Value']:,.0f} | "
              f"Positions: {row['Positions']:.0f} | Regime: {'ON' if row.get('Regime_Active', False) else 'OFF'}")
    
    # Check for missing months
    print("\n   Checking for gaps...")
    expected_months = pd.date_range(start=START_DATE, end=END_DATE, freq='ME')
    actual_months = set(monthly.index.strftime('%Y-%m'))
    expected_set = set(expected_months.strftime('%Y-%m'))
    missing = expected_set - actual_months
    if missing:
        print(f"   [WARN] Missing months: {sorted(missing)}")
    else:
        print("   [OK] All months have data")
        
else:
    print("   [FAIL] No portfolio data generated")

# Check trades
print("\n[7] Analyzing trades...")
if engine.trades_df is not None and not engine.trades_df.empty:
    trades = engine.trades_df
    print(f"   Total trades: {len(trades)}")
    
    # Group by month
    trades['Month'] = pd.to_datetime(trades['Date']).dt.to_period('M')
    monthly_trades = trades.groupby('Month').size()
    print("\n   Trades per month:")
    for month, count in monthly_trades.items():
        print(f"   {month}: {count} trades")
    
    # Check for months with no trades
    all_months = pd.period_range(start='2025-01', end='2025-12', freq='M')
    months_with_trades = set(monthly_trades.index)
    months_without = set(all_months) - months_with_trades
    if months_without:
        print(f"\n   [ISSUE] Months without trades: {sorted([str(m) for m in months_without])}")
    
    # Show sample trades
    print("\n   Sample trades (first 10):")
    sample = trades.head(10)[['Date', 'Ticker', 'Action', 'Shares', 'Price', 'Value']]
    for _, t in sample.iterrows():
        print(f"   {t['Date'].strftime('%Y-%m-%d') if hasattr(t['Date'], 'strftime') else t['Date']} | "
              f"{t['Action']:4} | {t['Ticker']:12} | {t['Shares']:5.0f} @ {t['Price']:8.2f} = {t['Value']:10.2f}")
else:
    print("   [FAIL] No trades generated")

# Check metrics
print("\n[8] Performance metrics...")
try:
    metrics = engine.get_metrics()
    if metrics:
        print(f"   Total Return: {metrics.get('Total Return %', 0):.1f}%")
        print(f"   CAGR: {metrics.get('CAGR %', 0):.1f}%")
        print(f"   Max Drawdown: {metrics.get('Max Drawdown %', 0):.1f}%")
        print(f"   Sharpe: {metrics.get('Sharpe Ratio', 0):.2f}")
        print(f"   Win Rate: {metrics.get('Win Rate %', 0):.1f}%")
    else:
        print("   [WARN] Metrics not available")
except Exception as e:
    print(f"   [FAIL] Metrics error: {e}")

print("\n" + "=" * 70)
print("LOCAL TEST COMPLETE")
print("=" * 70)
