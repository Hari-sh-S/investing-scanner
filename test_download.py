"""
Test script for different download approaches
Run locally to find the most reliable fast method
"""
import time
import yfinance as yf
import pandas as pd

TEST_TICKERS = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'HINDUNILVR', 'ITC', 'SBIN', 'BHARTIARTL', 'KOTAKBANK']

def test_sequential():
    """Original sequential download - baseline"""
    print("\n=== Testing SEQUENTIAL download ===")
    start = time.time()
    success = 0
    
    for ticker in TEST_TICKERS:
        try:
            df = yf.download(f"{ticker}.NS", period="1y", interval="1d", progress=False, auto_adjust=True)
            if not df.empty:
                success += 1
                print(f"  [OK] {ticker}: {len(df)} rows")
            else:
                print(f"  [FAIL] {ticker}: empty")
        except Exception as e:
            print(f"  [FAIL] {ticker}: {e}")
    
    elapsed = time.time() - start
    print(f"\nSequential: {success}/{len(TEST_TICKERS)} in {elapsed:.1f}s ({elapsed/len(TEST_TICKERS):.2f}s/stock)")
    return success, elapsed


def test_batch_download():
    """yfinance batch download - downloads multiple at once"""
    print("\n=== Testing BATCH download ===")
    start = time.time()
    
    tickers_ns = [f"{t}.NS" for t in TEST_TICKERS]
    try:
        df = yf.download(tickers_ns, period="1y", interval="1d", progress=False, auto_adjust=True, group_by='ticker')
        
        success = 0
        for ticker in TEST_TICKERS:
            ticker_ns = f"{ticker}.NS"
            try:
                if ticker_ns in df.columns.get_level_values(0):
                    ticker_df = df[ticker_ns]
                    if not ticker_df.empty and len(ticker_df.dropna()) > 50:
                        success += 1
                        print(f"  [OK] {ticker}: {len(ticker_df.dropna())} rows")
                    else:
                        print(f"  [FAIL] {ticker}: insufficient data")
                else:
                    print(f"  [FAIL] {ticker}: not in result")
            except Exception as e:
                print(f"  [FAIL] {ticker}: {e}")
    except Exception as e:
        print(f"  Batch download failed: {e}")
        success = 0
    
    elapsed = time.time() - start
    print(f"\nBatch: {success}/{len(TEST_TICKERS)} in {elapsed:.1f}s ({elapsed/len(TEST_TICKERS):.2f}s/stock)")
    return success, elapsed


def test_batch_small_groups():
    """Download in small batches of 5"""
    print("\n=== Testing SMALL BATCH download (5 at a time) ===")
    start = time.time()
    success = 0
    batch_size = 5
    
    for i in range(0, len(TEST_TICKERS), batch_size):
        batch = TEST_TICKERS[i:i+batch_size]
        tickers_ns = [f"{t}.NS" for t in batch]
        
        try:
            df = yf.download(tickers_ns, period="1y", interval="1d", progress=False, auto_adjust=True, group_by='ticker')
            
            for ticker in batch:
                ticker_ns = f"{ticker}.NS"
                try:
                    if len(tickers_ns) == 1:
                        if not df.empty:
                            success += 1
                            print(f"  [OK] {ticker}: {len(df)} rows")
                    elif ticker_ns in df.columns.get_level_values(0):
                        ticker_df = df[ticker_ns]
                        if not ticker_df.empty and len(ticker_df.dropna()) > 50:
                            success += 1
                            print(f"  [OK] {ticker}: {len(ticker_df.dropna())} rows")
                except:
                    pass
        except Exception as e:
            print(f"  Batch {i//batch_size + 1} failed: {e}")
        
        time.sleep(0.5)
    
    elapsed = time.time() - start
    print(f"\nSmall Batch: {success}/{len(TEST_TICKERS)} in {elapsed:.1f}s ({elapsed/len(TEST_TICKERS):.2f}s/stock)")
    return success, elapsed


if __name__ == "__main__":
    print("=" * 60)
    print("DOWNLOAD METHOD COMPARISON TEST")
    print("=" * 60)
    
    results = {}
    
    seq_success, seq_time = test_sequential()
    results['Sequential'] = (seq_success, seq_time)
    
    time.sleep(2)
    
    batch_success, batch_time = test_batch_download()
    results['Batch'] = (batch_success, batch_time)
    
    time.sleep(2)
    
    small_success, small_time = test_batch_small_groups()
    results['SmallBatch'] = (small_success, small_time)
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for method, (success, elapsed) in results.items():
        reliability = success / len(TEST_TICKERS) * 100
        speed = elapsed / len(TEST_TICKERS)
        status = "RELIABLE" if reliability >= 90 else "UNRELIABLE"
        print(f"{method:15} | {success}/{len(TEST_TICKERS)} ({reliability:.0f}%) | {elapsed:.1f}s | {speed:.2f}s/stock | {status}")
    
    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    
    best = None
    best_score = 0
    for method, (success, elapsed) in results.items():
        if success >= len(TEST_TICKERS) * 0.9:
            score = success * 100 - elapsed
            if score > best_score:
                best = method
                best_score = score
    
    if best:
        print(f"Best method: {best}")
    else:
        print("No reliable method found - stick with Sequential")
