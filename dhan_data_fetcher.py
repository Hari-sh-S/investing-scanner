"""
Dhan API Data Fetcher

Fetches historical OHLC data from Dhan API for NSE stocks.
Uses the dhanhq SDK and handles:
- Security ID mapping (symbol -> Dhan security ID)
- Custom epoch timestamp conversion (from 1980-01-01 IST)
- Rate limiting to avoid API throttling
"""

import os
import time
import pandas as pd
import requests
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional, Callable
from dotenv import load_dotenv

load_dotenv()

# Dhan uses epoch from 1980-01-01 00:00:00 IST
DHAN_EPOCH = datetime(1980, 1, 1, 0, 0, 0)

# Cache for security ID mappings
SECURITY_ID_CACHE: dict[str, int] = {}
INSTRUMENTS_CACHE_PATH = Path("dhan_instruments_cache.csv")


def _convert_dhan_timestamp(dhan_ts) -> datetime:
    """Convert Dhan's timestamp to datetime.
    
    Dhan may return timestamps in different formats:
    - Integer/float: seconds since 1980-01-01 00:00:00 IST
    - String: date string like '2024-01-15'
    """
    # Handle string input
    if isinstance(dhan_ts, str):
        # Try parsing as date string first
        try:
            return datetime.strptime(dhan_ts, '%Y-%m-%d')
        except ValueError:
            pass
        # Try as numeric string
        try:
            dhan_ts = int(float(dhan_ts))
        except ValueError:
            # Return current date as fallback
            return datetime.now()
    
    # Handle numeric timestamp (Dhan's custom epoch from 1980)
    return DHAN_EPOCH + timedelta(seconds=int(dhan_ts))


def _download_instruments_list() -> pd.DataFrame:
    """Download and cache Dhan's instruments list."""
    # Dhan provides CSV at this URL
    url = "https://images.dhan.co/api-data/api-scrip-master.csv"
    
    try:
        print("Downloading Dhan instruments list...")
        df = pd.read_csv(url)
        
        # Filter for NSE equity segment
        # SEM_EXM_EXCH_ID: NSE = 1, BSE = 2
        # SEM_SEGMENT: EQ = Equity
        nse_eq = df[(df['SEM_EXM_EXCH_ID'] == 'NSE') & (df['SEM_SEGMENT'] == 'E')]
        
        # Cache it
        nse_eq.to_csv(INSTRUMENTS_CACHE_PATH, index=False)
        print(f"Cached {len(nse_eq)} NSE equity instruments")
        
        return nse_eq
    except Exception as e:
        print(f"Error downloading instruments: {e}")
        
        # Try to use cached version
        if INSTRUMENTS_CACHE_PATH.exists():
            print("Using cached instruments list")
            return pd.read_csv(INSTRUMENTS_CACHE_PATH)
        
        raise ValueError("Could not fetch Dhan instruments list")


def get_dhan_security_id(symbol: str) -> Optional[int]:
    """Get Dhan security ID for an NSE symbol.
    
    Args:
        symbol: NSE symbol (e.g., 'RELIANCE', 'TCS')
        
    Returns:
        Dhan security ID or None if not found
    """
    global SECURITY_ID_CACHE
    
    # Check cache first
    if symbol in SECURITY_ID_CACHE:
        return SECURITY_ID_CACHE[symbol]
    
    # Load instruments if cache is empty
    if not SECURITY_ID_CACHE:
        try:
            if INSTRUMENTS_CACHE_PATH.exists():
                # Check if cache is fresh (less than 7 days old)
                cache_age = time.time() - INSTRUMENTS_CACHE_PATH.stat().st_mtime
                if cache_age < 7 * 24 * 3600:  # 7 days
                    df = pd.read_csv(INSTRUMENTS_CACHE_PATH)
                else:
                    df = _download_instruments_list()
            else:
                df = _download_instruments_list()
            
            # Build security ID cache
            # SEM_TRADING_SYMBOL is the trading symbol
            # SEM_SMST_SECURITY_ID is the security ID
            for _, row in df.iterrows():
                trading_symbol = str(row.get('SEM_TRADING_SYMBOL', '')).strip()
                security_id = row.get('SEM_SMST_SECURITY_ID')
                if trading_symbol and pd.notna(security_id):
                    SECURITY_ID_CACHE[trading_symbol] = int(security_id)
                    
            print(f"Loaded {len(SECURITY_ID_CACHE)} security ID mappings")
            
        except Exception as e:
            print(f"Error loading security IDs: {e}")
            return None
    
    return SECURITY_ID_CACHE.get(symbol)


def fetch_historical_data(
    symbol: str,
    from_date: date,
    to_date: date
) -> Optional[pd.DataFrame]:
    """Fetch historical daily OHLC data for a symbol from Dhan API.
    
    Args:
        symbol: NSE symbol (e.g., 'RELIANCE')
        from_date: Start date
        to_date: End date
        
    Returns:
        DataFrame with columns: Date, Open, High, Low, Close, Volume
        Returns None if fetch fails
    """
    from config import get_dhan_client, validate_credentials
    
    try:
        validate_credentials()
    except ValueError as e:
        print(f"Dhan credentials error: {e}")
        return None
    
    # Get security ID
    security_id = get_dhan_security_id(symbol)
    if security_id is None:
        print(f"Could not find security ID for {symbol}")
        return None
    
    try:
        # Get Dhan client
        dhan = get_dhan_client()
        
        # Fetch historical daily data
        # Using the historical_daily_data method from dhanhq SDK
        response = dhan.historical_daily_data(
            security_id=str(security_id),
            exchange_segment='NSE_EQ',
            instrument_type='EQUITY',
            from_date=from_date.strftime('%Y-%m-%d'),
            to_date=to_date.strftime('%Y-%m-%d')
        )
        
        if response.get('status') != 'success':
            print(f"Dhan API error for {symbol}: {response.get('remarks', 'Unknown error')}")
            return None
        
        data = response.get('data', [])
        if not data:
            print(f"No data returned for {symbol}")
            return None
        
        # Convert to DataFrame
        records = []
        for candle in data:
            # Dhan returns: [timestamp, open, high, low, close, volume]
            ts = _convert_dhan_timestamp(candle[0])
            records.append({
                'Date': ts.date(),
                'Open': float(candle[1]),
                'High': float(candle[2]),
                'Low': float(candle[3]),
                'Close': float(candle[4]),
                'Volume': int(candle[5]) if len(candle) > 5 else 0
            })
        
        df = pd.DataFrame(records)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        
        return df
        
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None


def fetch_all_universe_data(
    symbols: list[str],
    from_date: date,
    to_date: date,
    progress_callback: Optional[Callable[[int, int, str, float], None]] = None,
    delay_seconds: float = 0.5
) -> dict[str, pd.DataFrame]:
    """Fetch historical data for all symbols in a universe.
    
    Args:
        symbols: List of NSE symbols
        from_date: Start date
        to_date: End date
        progress_callback: Callback function(current, total, symbol, remaining_seconds)
        delay_seconds: Delay between API calls to avoid rate limiting
        
    Returns:
        Dictionary mapping symbol -> DataFrame
    """
    results = {}
    start_time = time.time()
    
    for i, symbol in enumerate(symbols):
        # Calculate ETA
        elapsed = time.time() - start_time
        avg_time = elapsed / (i + 1) if i > 0 else delay_seconds
        remaining = avg_time * (len(symbols) - i - 1)
        
        if progress_callback:
            progress_callback(i + 1, len(symbols), symbol, remaining)
        
        # Fetch data
        df = fetch_historical_data(symbol, from_date, to_date)
        if df is not None and not df.empty:
            results[symbol] = df
            print(f"✓ {symbol}: {len(df)} days")
        else:
            print(f"✗ {symbol}: No data")
        
        # Rate limiting delay (except for last symbol)
        if i < len(symbols) - 1:
            time.sleep(delay_seconds)
    
    return results


# Test function
if __name__ == "__main__":
    # Test with a single symbol
    print("Testing Dhan data fetch...")
    
    test_symbol = "RELIANCE"
    test_from = date(2024, 1, 1)
    test_to = date(2024, 12, 31)
    
    df = fetch_historical_data(test_symbol, test_from, test_to)
    
    if df is not None:
        print(f"\nFetched {len(df)} days of data for {test_symbol}")
        print(df.head())
        print(df.tail())
    else:
        print("Failed to fetch data")
