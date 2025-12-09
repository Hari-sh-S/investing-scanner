"""
NSE Index Constituents Fetcher
Downloads live index constituents from NSE India website
"""

import requests
import json
import time
from pathlib import Path

# NSE requires proper headers to access API
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Referer': 'https://www.nseindia.com/market-data/live-market-indices',
}

# All NSE Indices to fetch
NSE_INDICES = [
    # Broad Market
    "NIFTY 50",
    "NIFTY NEXT 50", 
    "NIFTY 100",
    "NIFTY 200",
    "NIFTY 500",
    "NIFTY TOTAL MARKET",
    
    # Cap-based
    "NIFTY MIDCAP 50",
    "NIFTY MIDCAP 100",
    "NIFTY MIDCAP 150",
    "NIFTY SMALLCAP 50",
    "NIFTY SMALLCAP 100",
    "NIFTY SMALLCAP 250",
    "NIFTY MICROCAP 250",
    "NIFTY LARGEMIDCAP 250",
    "NIFTY MIDSMALLCAP 400",
    
    # Sectoral
    "NIFTY BANK",
    "NIFTY FINANCIAL SERVICES",
    "NIFTY IT",
    "NIFTY PHARMA",
    "NIFTY AUTO",
    "NIFTY FMCG",
    "NIFTY METAL",
    "NIFTY REALTY",
    "NIFTY ENERGY",
    "NIFTY CONSUMPTION",
    "NIFTY MEDIA",
    "NIFTY HEALTHCARE INDEX",
    "NIFTY OIL & GAS",
    
    # Thematic
    "NIFTY PSU BANK",
    "NIFTY PRIVATE BANK",
    "NIFTY PSE",
    "NIFTY COMMODITIES",
    "NIFTY CPSE",
    "NIFTY MNC",
    "NIFTY INFRASTRUCTURE",
    "NIFTY INDIA DIGITAL",
    "NIFTY INDIA CONSUMPTION",
    "NIFTY INDIA MANUFACTURING",
    "NIFTY INDIA DEFENCE",
    
    # Strategy
    "NIFTY ALPHA 50",
    "NIFTY50 VALUE 20",
    "NIFTY GROWTH SECTORS 15",
    "NIFTY100 QUALITY 30",
    "NIFTY50 EQUAL WEIGHT",
    "NIFTY100 EQUAL WEIGHT",
    "NIFTY100 LOW VOLATILITY 30",
    "NIFTY DIVIDEND OPPORTUNITIES 50",
]

CACHE_FILE = Path("nse_universe_cache.json")


def get_session():
    """Create a session with NSE cookies."""
    session = requests.Session()
    session.headers.update(HEADERS)
    
    # First hit the main page to get cookies
    try:
        response = session.get('https://www.nseindia.com', timeout=10)
        time.sleep(0.5)  # Small delay to avoid rate limiting
    except Exception as e:
        print(f"Error getting NSE session: {e}")
    
    return session


def fetch_index_constituents(session, index_name):
    """Fetch constituents for a single index."""
    url = f"https://www.nseindia.com/api/equity-stockIndices?index={requests.utils.quote(index_name)}"
    
    try:
        response = session.get(url, timeout=15)
        if response.status_code == 200:
            data = response.json()
            
            # Extract stock symbols from the data
            stocks = []
            if 'data' in data:
                for stock in data['data']:
                    symbol = stock.get('symbol', '')
                    if symbol and symbol != index_name:
                        stocks.append(symbol)
            
            return stocks
        else:
            print(f"Error fetching {index_name}: HTTP {response.status_code}")
            return None
    except Exception as e:
        print(f"Error fetching {index_name}: {e}")
        return None


def fetch_all_indices(progress_callback=None):
    """Fetch constituents for all indices."""
    session = get_session()
    results = {}
    
    total = len(NSE_INDICES)
    for i, index_name in enumerate(NSE_INDICES):
        if progress_callback:
            progress_callback(i / total, f"Fetching {index_name}...")
        
        stocks = fetch_index_constituents(session, index_name)
        if stocks:
            results[index_name] = stocks
            print(f"✓ {index_name}: {len(stocks)} stocks")
        else:
            print(f"✗ {index_name}: Failed to fetch")
        
        time.sleep(0.3)  # Rate limiting
    
    return results


def save_to_cache(data):
    """Save fetched data to cache file."""
    cache_data = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'universes': data
    }
    
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache_data, f, indent=2)
    
    print(f"Saved {len(data)} universes to {CACHE_FILE}")


def load_from_cache():
    """Load data from cache file."""
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, 'r') as f:
                data = json.load(f)
                return data.get('universes', {}), data.get('timestamp', 'Unknown')
        except Exception as e:
            print(f"Error loading cache: {e}")
    return {}, None


def get_universe(name):
    """Get a universe by name, from cache or hardcoded fallback."""
    cached, timestamp = load_from_cache()
    if name in cached:
        return cached[name]
    
    # Fallback to hardcoded
    from nifty_universe import UNIVERSES
    return UNIVERSES.get(name, [])


def get_all_universe_names():
    """Get all available universe names from cache."""
    cached, _ = load_from_cache()
    if cached:
        return sorted(cached.keys())
    
    # Fallback
    return NSE_INDICES


def refresh_universes(progress_callback=None):
    """Refresh all universe data from NSE."""
    print("Refreshing universe data from NSE India...")
    data = fetch_all_indices(progress_callback)
    
    if data:
        save_to_cache(data)
        return True, f"Updated {len(data)} universes"
    else:
        return False, "Failed to fetch data from NSE"


if __name__ == "__main__":
    print("NSE Universe Fetcher")
    print("=" * 50)
    
    success, message = refresh_universes()
    print(f"\nResult: {message}")
    
    # Show summary
    cached, timestamp = load_from_cache()
    if cached:
        print(f"\nCached at: {timestamp}")
        print(f"Total universes: {len(cached)}")
        
        # Show stock counts
        print("\nUniverse stock counts:")
        for name, stocks in sorted(cached.items()):
            print(f"  {name}: {len(stocks)} stocks")
