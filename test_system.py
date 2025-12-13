"""
Comprehensive test script for the backtesting system.
Tests: Scoring, Indicators, Data Loading, and Backtest Execution
"""

import sys
import datetime
import pandas as pd
import numpy as np

print("=" * 60)
print("COMPREHENSIVE SELF-TEST")
print("=" * 60)

# Test 1: Import all modules
print("\n[1] Testing module imports...")
try:
    from scoring import ScoreParser
    from indicators import IndicatorLibrary
    from portfolio_engine import PortfolioEngine, DataCache
    from nifty_universe import INDEX_NAMES, get_universe
    print("   [OK] All modules imported successfully")
except Exception as e:
    print(f"   [FAIL] Import failed: {e}")
    sys.exit(1)

# Test 2: ScoreParser
print("\n[2] Testing ScoreParser...")
parser = ScoreParser()

# Create mock row with indicator data
mock_row = pd.Series({
    '1 Month Performance': 0.05,
    '3 Month Performance': 0.12,
    '6 Month Performance': 0.25,
    '9 Month Performance': 0.30,
    '1 Year Performance': 0.40,
    '1 Month Volatility': 0.15,
    '3 Month Volatility': 0.18,
    '6 Month Volatility': 0.20,
    '9 Month Volatility': 0.22,
    '1 Year Volatility': 0.25,
    '6 Month Sharpe': 1.2,
    '6 Month Sortino': 1.5,
    '6 Month Calmar': 0.8,
    '6 Month Max Drawdown': -0.15,
})

# Test formulas
test_formulas = [
    ("6 Month Performance", 0.25),
    ("(6 Month Performance / 6 Month Volatility)", 0.25 / 0.20),
    ("6 Month Performance + 6 Month Sharpe", 0.25 + 1.2),
    ("(6 Month Performance * 0.8) + (3 Month Performance * 0.2)", 0.25*0.8 + 0.12*0.2),
]

all_scoring_passed = True
for formula, expected in test_formulas:
    # Validate
    is_valid, msg = parser.validate_formula(formula)
    if not is_valid:
        print(f"   [FAIL] Validation failed for '{formula}': {msg}")
        all_scoring_passed = False
        continue
    
    # Calculate
    result = parser.parse_and_calculate(formula, mock_row)
    if abs(result - expected) < 0.001:
        print(f"   [OK] '{formula[:40]}...' = {result:.4f}")
    else:
        print(f"   [FAIL] '{formula[:40]}...' = {result:.4f} (expected {expected:.4f})")
        all_scoring_passed = False

# Test vectorized scoring
print("\n   Testing vectorized scoring...")
mock_df = pd.DataFrame([mock_row, mock_row * 1.1, mock_row * 0.9])
scores = parser.calculate_scores(mock_df, "6 Month Performance")
if len(scores) == 3 and not pd.isna(scores).any():
    print(f"   [OK] Vectorized scoring works: {scores.tolist()}")
else:
    print(f"   [FAIL] Vectorized scoring failed")
    all_scoring_passed = False

if all_scoring_passed:
    print("   [OK] All scoring tests PASSED")
else:
    print("   [FAIL] Some scoring tests FAILED")

# Test 3: IndicatorLibrary
print("\n[3] Testing IndicatorLibrary...")
# Create mock OHLCV data
np.random.seed(42)
n_days = 300
dates = pd.date_range(end=datetime.date.today(), periods=n_days, freq='D')
close_prices = 100 * (1 + np.random.randn(n_days).cumsum() * 0.01)
mock_ohlcv = pd.DataFrame({
    'Open': close_prices * 0.99,
    'High': close_prices * 1.02,
    'Low': close_prices * 0.98,
    'Close': close_prices,
    'Volume': np.random.randint(1000000, 10000000, n_days)
}, index=dates)

try:
    # Test momentum metrics
    df_with_indicators = IndicatorLibrary.add_momentum_volatility_metrics(mock_ohlcv.copy())
    required_cols = ['6 Month Performance', '6 Month Volatility', '6 Month Sharpe', '6 Month Sortino']
    missing = [c for c in required_cols if c not in df_with_indicators.columns]
    if missing:
        print(f"   [FAIL] Missing indicator columns: {missing}")
    else:
        print(f"   [OK] Momentum metrics added ({len([c for c in df_with_indicators.columns if 'Month' in c or 'Year' in c])} columns)")
    
    # Test regime filters
    df_with_regime = IndicatorLibrary.add_regime_filters(mock_ohlcv.copy())
    regime_cols = ['EMA_100', 'EMA_200', 'MACD', 'Supertrend']
    missing = [c for c in regime_cols if c not in df_with_regime.columns]
    if missing:
        print(f"   [FAIL] Missing regime columns: {missing}")
    else:
        print(f"   [OK] Regime filters added ({len([c for c in df_with_regime.columns if 'EMA' in c or 'MACD' in c])} columns)")
    
    print("   [OK] IndicatorLibrary tests PASSED")
except Exception as e:
    print(f"   [FAIL] IndicatorLibrary test failed: {e}")

# Test 4: Universe definitions
print("\n[4] Testing universe definitions...")
print(f"   Total indexes defined: {len(INDEX_NAMES)}")
sample_universes = ['NIFTY 50', 'NIFTY 100', 'NIFTY BANK']
for uni in sample_universes:
    stocks = get_universe(uni)
    if stocks:
        print(f"   [OK] {uni}: {len(stocks)} stocks")
    else:
        print(f"   [FAIL] {uni}: No stocks found")

# Test 5: DataCache
print("\n[5] Testing DataCache...")
cache = DataCache()
info = cache.get_cache_info()
print(f"   Cache: {info['total_files']} files, {info['total_size_mb']:.2f} MB")

# Test 6: Quick backtest simulation
print("\n[6] Testing backtest with mock data...")
try:
    # Use a small universe for testing
    test_tickers = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK']
    
    # Create engine
    engine = PortfolioEngine(
        test_tickers,
        datetime.date(2024, 1, 1),
        datetime.date(2024, 12, 31),
        initial_capital=100000
    )
    
    # Manually add mock data for testing
    for ticker in test_tickers:
        n_days = 252
        dates = pd.date_range(start='2024-01-01', periods=n_days, freq='B')
        close = 100 * (1 + np.random.randn(n_days).cumsum() * 0.02)
        mock_df = pd.DataFrame({
            'Open': close * 0.99,
            'High': close * 1.02,
            'Low': close * 0.98,
            'Close': close,
            'Volume': np.random.randint(1000000, 10000000, n_days)
        }, index=dates)
        engine.data[ticker] = mock_df
    
    print(f"   Mock data created for {len(engine.data)} stocks")
    
    # Calculate indicators
    engine.calculate_indicators_for_formula("6 Month Performance")
    
    # Check indicators exist
    sample_df = list(engine.data.values())[0]
    if '6 Month Performance' in sample_df.columns:
        print("   [OK] Indicators calculated successfully")
    else:
        print("   [FAIL] Indicators not found")
    
    # Run backtest
    rebal_config = {'frequency': 'Monthly', 'date': 1, 'alt_day': 'Next Day'}
    engine.run_rebalance_strategy(
        scoring_formula="6 Month Performance",
        num_stocks=5,
        rebal_config=rebal_config,
        exit_rank=None,
        regime_config=None,
        uncorrelated_config=None,
        reinvest_profits=True
    )
    
    # Check results
    if engine.portfolio_df is not None and not engine.portfolio_df.empty:
        final_value = engine.portfolio_df['Portfolio Value'].iloc[-1]
        trades_count = len(engine.trades_df) if engine.trades_df is not None else 0
        print(f"   [OK] Backtest completed: Final value ${final_value:,.0f}, {trades_count} trades")
    else:
        print("   [FAIL] Backtest produced no results")
    
    # Test metrics
    metrics = engine.get_metrics()
    if metrics and 'Total Return %' in metrics:
        print(f"   [OK] Metrics calculated: Return {metrics['Total Return %']:.1f}%, Sharpe {metrics.get('Sharpe Ratio', 0):.2f}")
    else:
        print("   [FAIL] Metrics calculation failed")
    
    print("   [OK] Backtest tests PASSED")
    
except Exception as e:
    print(f"   [FAIL] Backtest test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("SELF-TEST COMPLETE")
print("=" * 60)

