"""
Local test with user's exact parameters from the screenshot.
"""

import datetime
import pandas as pd
from portfolio_engine import PortfolioEngine
from nifty_universe import get_universe

print("=" * 70)
print("LOCAL TEST: User's Exact Parameters")
print("=" * 70)

# Settings from screenshot
START_DATE = datetime.date(2025, 1, 1)
END_DATE = datetime.date(2025, 11, 28)  # Data ends Nov 28
INITIAL_CAPITAL = 100000
SCORING_FORMULA = "(6 Month Performance) / (6 Month Volatility)"
NUM_STOCKS = 3
EXIT_RANK = None  # "All" means None

# Regime filter: EMA 100, Half Portfolio, NIFTY 50
REGIME_CONFIG = {
    'type': 'EMA',
    'value': 100,
    'action': 'Half Portfolio',
    'index': 'NIFTY 50'
}

# Uncorrelated: GOLDBEES 100%
UNCORRELATED_CONFIG = {
    'asset': 'GOLDBEES',
    'allocation_pct': 100
}

REBAL_CONFIG = {
    'frequency': 'Monthly',
    'date': 1,
    'alt_day': 'Next Day'
}

# Get universe (NIFTY LARGECAP100 = NIFTY 100)
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

# Fetch data
print("\n[3] Loading data from cache...")
def progress_cb(current, total, ticker):
    if current % 25 == 0 or current == total:
        print(f"   Loading: {current}/{total}")

success = engine.fetch_data(progress_callback=progress_cb)
print(f"   Loaded {len(engine.data)} stocks")

# Run backtest
print("\n[4] Running backtest...")
print(f"   Formula: {SCORING_FORMULA}")
print(f"   Stocks: {NUM_STOCKS}")
print(f"   Regime: EMA 100 on NIFTY 50, Action: Half Portfolio")
print(f"   Uncorrelated: GOLDBEES 100%")
print(f"   Rebalance: Monthly on day 1")

try:
    engine.run_rebalance_strategy(
        scoring_formula=SCORING_FORMULA,
        num_stocks=NUM_STOCKS,
        rebal_config=REBAL_CONFIG,
        exit_rank=EXIT_RANK,
        regime_config=REGIME_CONFIG,
        uncorrelated_config=UNCORRELATED_CONFIG,
        reinvest_profits=True
    )
    print("   [OK] Backtest completed")
except Exception as e:
    print(f"   [FAIL] Backtest failed: {e}")
    import traceback
    traceback.print_exc()

# Results
print("\n[5] Results Summary...")
if engine.portfolio_df is not None and not engine.portfolio_df.empty:
    pf = engine.portfolio_df
    
    # Monthly summary
    print("\n   Monthly Portfolio Values:")
    monthly = pf.resample('ME').last()
    for idx, row in monthly.iterrows():
        regime_str = 'HALF' if row.get('Regime_Active', False) else 'FULL'
        print(f"   {idx.strftime('%Y-%m')}: ${row['Portfolio Value']:,.0f} | "
              f"Positions: {row['Positions']:.0f} | {regime_str}")
    
    # Final metrics
    print("\n[6] Performance Metrics...")
    metrics = engine.get_metrics()
    if metrics:
        print(f"   Total Return: {metrics.get('Total Return %', 0):.1f}%")
        print(f"   CAGR: {metrics.get('CAGR %', 0):.1f}%")
        print(f"   Max Drawdown: {metrics.get('Max Drawdown %', 0):.1f}%")
        print(f"   Sharpe: {metrics.get('Sharpe Ratio', 0):.2f}")
        print(f"   Win Rate: {metrics.get('Win Rate %', 0):.1f}%")
        print(f"   Total Trades: {metrics.get('Total Trades', 0)}")
else:
    print("   [FAIL] No results")

# Trade analysis
print("\n[7] Trade History...")
if engine.trades_df is not None and not engine.trades_df.empty:
    trades = engine.trades_df
    print(f"   Total trades: {len(trades)}")
    
    # Show all months
    trades['Month'] = pd.to_datetime(trades['Date']).dt.to_period('M')
    monthly_trades = trades.groupby('Month').size()
    print("\n   Trades per month:")
    for month, count in monthly_trades.items():
        print(f"   {month}: {count} trades")
    
    # Check for gaps
    all_months = pd.period_range(start='2024-01', end='2024-12', freq='M')
    months_with_trades = set(monthly_trades.index)
    months_without = set(all_months) - months_with_trades
    if months_without:
        print(f"\n   [WARN] Months without trades: {sorted([str(m) for m in months_without])}")
    else:
        print("\n   [OK] All months have trades")

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
