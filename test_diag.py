"""Quick diagnostic - check if indicators exist."""
import datetime
from portfolio_engine import PortfolioEngine
from nifty_universe import get_universe

# Load a small test
engine = PortfolioEngine(
    get_universe('NIFTY 50')[:5],  # Just 5 stocks
    datetime.date(2025, 1, 1),
    datetime.date(2025, 12, 13),
    initial_capital=100000
)

print("Loading data...")
engine.fetch_data()

print(f"\nLoaded {len(engine.data)} stocks")

# Check first stock
if engine.data:
    ticker = list(engine.data.keys())[0]
    df = engine.data[ticker]
    print(f"\n{ticker} columns BEFORE indicators:")
    print(f"  {list(df.columns)}")
    
    # Calculate indicators
    print("\nCalculating indicators...")
    engine.calculate_indicators_for_formula("6 Month Performance / 6 Month Volatility")
    
    # Check again
    df = engine.data[ticker]
    print(f"\n{ticker} columns AFTER indicators:")
    print(f"  Shape: {df.shape}")
    indicator_cols = [c for c in df.columns if 'Performance' in c or 'Volatility' in c]
    print(f"  Indicator columns: {indicator_cols}")
    
    # Check May 2 values
    if '6 Month Performance' in df.columns:
        may_data = df.loc[df.index >= '2025-05-01'][:5]
        print(f"\n{ticker} May 2025 data:")
        print(may_data[['Close', '6 Month Performance', '6 Month Volatility']].to_string())
    else:
        print("\n  [ERROR] 6 Month Performance column NOT FOUND!")
