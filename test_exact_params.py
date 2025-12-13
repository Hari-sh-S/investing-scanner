"""
Test exact parameters from user's screenshots.
NIFTY LARGEMID250, 5 stocks, Exit Rank 8, EMA 100 Go Cash, GOLDBEES 100%
"""

import datetime
import pandas as pd
from portfolio_engine import PortfolioEngine
from nifty_universe import get_universe

print("=" * 70)
print("EXACT USER PARAMETERS TEST")
print("=" * 70)

# EXACT settings from screenshots
START_DATE = datetime.date(2025, 1, 1)
END_DATE = datetime.date(2025, 11, 28)  # Use cached data end date
INITIAL_CAPITAL = 100000
SCORING_FORMULA = "(6 Month Performance / 6 Month Volatility)"
NUM_STOCKS = 5
EXIT_RANK = 8  # User has Exit Rank = 8

# Regime filter: EMA 100, Go Cash, NIFTY 100
REGIME_CONFIG = {
    'type': 'EMA',
    'value': 100,
    'action': 'Go Cash',  # GO CASH not Half Portfolio
    'index': 'NIFTY 100'
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

# Get universe - NIFTY LARGEMID250
print("\n[1] Loading universe...")
universe = get_universe('NIFTY LARGEMID250')
print(f"   Universe: NIFTY LARGEMID250 with {len(universe)} stocks")

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
success = engine.fetch_data()
print(f"   Loaded {len(engine.data)} stocks")

# Check a sample stock's columns
if engine.data:
    sample_ticker = list(engine.data.keys())[0]
    sample_df = engine.data[sample_ticker]
    print(f"\n   Sample: {sample_ticker}")
    print(f"   Columns type: {type(sample_df.columns)}")
    print(f"   Sample columns: {list(sample_df.columns)[:10]}")
    if isinstance(sample_df.columns[0], tuple):
        print("   [ERROR] MultiIndex columns still present!")
    else:
        print("   [OK] Columns are flat strings")

# Run backtest
print("\n[4] Running backtest...")
print(f"   Formula: {SCORING_FORMULA}")
print(f"   Stocks: {NUM_STOCKS}, Exit Rank: {EXIT_RANK}")
print(f"   Regime: EMA 100 on NIFTY 100, Action: Go Cash")
print(f"   Uncorrelated: GOLDBEES 100%")

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
print("\n[5] Monthly Results...")
if engine.portfolio_df is not None and not engine.portfolio_df.empty:
    pf = engine.portfolio_df.resample('ME').last()
    
    for idx, row in pf.iterrows():
        regime_str = 'REGIME' if row.get('Regime_Active', False) else 'NORMAL'
        print(f"   {idx.strftime('%Y-%m')}: ${row['Portfolio Value']:,.0f} | "
              f"Positions: {row['Positions']:.0f} | {regime_str}")
else:
    print("   [FAIL] No results")

# Trade analysis
print("\n[6] Trades by Month...")
if engine.trades_df is not None and not engine.trades_df.empty:
    trades = engine.trades_df.copy()
    trades['Month'] = pd.to_datetime(trades['Date']).dt.strftime('%Y-%m')
    
    for month in ['2025-01', '2025-02', '2025-03', '2025-04', '2025-05', '2025-06', 
                  '2025-07', '2025-08', '2025-09', '2025-10', '2025-11']:
        month_trades = trades[trades['Month'] == month]
        if len(month_trades) > 0:
            tickers = month_trades['Ticker'].unique().tolist()
            print(f"   {month}: {len(month_trades)} trades - {tickers[:3]}...")
        else:
            print(f"   {month}: NO TRADES [ISSUE]")
    
    # Show rounded trade list format
    print("\n[7] Trade History (like user sees)...")
    if hasattr(engine, 'get_completed_trades'):
        completed = engine.get_completed_trades()
    else:
        # Reconstruct from trades_df
        buys = trades[trades['Action'] == 'BUY'][['Date', 'Ticker', 'Price', 'Shares']].rename(
            columns={'Date': 'Buy_Date', 'Price': 'Buy_Price'})
        sells = trades[trades['Action'] == 'SELL'][['Date', 'Ticker', 'Price']].rename(
            columns={'Date': 'Exit_Date', 'Price': 'Exit_Price'})
        # Simple display
        print("   First 5 BUY trades:")
        print(trades[trades['Action'] == 'BUY'].head()[['Date', 'Ticker', 'Shares', 'Price']])
else:
    print("   [FAIL] No trades")

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
