"""
Fetch current constituents for all Broad Market Indices from NSE API
and generate historical snapshots from changes data.
"""

import json
import requests
from pathlib import Path
from datetime import datetime
import time

# Paths
DATA_DIR = Path(__file__).parent / "data"
CHANGES_DIR = DATA_DIR / "changes"
SNAPSHOTS_DIR = DATA_DIR / "index_constituents"
CURRENT_DIR = DATA_DIR / "current_constituents"

# All 19 Broad Market Indices - NSE API index names
INDICES = {
    "nifty50": "NIFTY 50",
    "niftynext50": "NIFTY NEXT 50",
    "nifty100": "NIFTY 100",
    "nifty200": "NIFTY 200",
    "nifty500": "NIFTY 500",
    "niftymidcap150": "NIFTY MIDCAP 150",
    "niftymidcap50": "NIFTY MIDCAP 50",
    "niftymidselect": "NIFTY MIDCAP SELECT",
    "niftymidcap100": "NIFTY MIDCAP 100",
    "niftysmlcap250": "NIFTY SMLCAP 250",
    "niftysmlcap50": "NIFTY SMLCAP 50",
    "niftysmlcap100": "NIFTY SMLCAP 100",
    "niftylargemid250": "NIFTY LARGEMIDCAP 250",
    "niftymidsml400": "NIFTY MIDSMALLCAP 400",
    "nifty500multicap": "NIFTY500 MULTICAP 50:25:25",
    "niftymicrocap250": "NIFTY MICROCAP 250",
    "niftytotalmkt": "NIFTY TOTAL MARKET",
    "nifty500lmseql": "NIFTY500 EQUAL WEIGHT",
    "niftyfpi150": "NIFTY FPI 150",
}

# Headers for NSE API
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'application/json',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://www.nseindia.com/',
}


def get_nse_session():
    """Create a session with NSE cookies."""
    session = requests.Session()
    session.headers.update(HEADERS)
    # Get cookies from main page first
    try:
        session.get('https://www.nseindia.com/', timeout=10)
    except:
        pass
    return session


def fetch_index_constituents(session, index_name: str) -> list:
    """Fetch current constituents for an index from NSE API."""
    # NSE API endpoint for index constituents
    url = f"https://www.nseindia.com/api/equity-stockIndices?index={index_name.replace(' ', '%20')}"
    
    try:
        response = session.get(url, timeout=30)
        if response.status_code == 200:
            data = response.json()
            stocks = data.get('data', [])
            # Extract symbols (skip the index itself which is first entry)
            symbols = [stock['symbol'] for stock in stocks[1:] if 'symbol' in stock]
            return symbols
    except Exception as e:
        print(f"  Error fetching {index_name}: {e}")
    return []


def save_current_constituents(index_key: str, index_name: str, symbols: list):
    """Save current constituents to JSON file."""
    CURRENT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = CURRENT_DIR / f"{index_key}.json"
    
    data = {
        "index": index_name,
        "fetched_at": datetime.now().isoformat(),
        "count": len(symbols),
        "symbols": sorted(symbols)
    }
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"  Saved {len(symbols)} stocks to {output_file.name}")


def load_changes(index_key: str) -> list:
    """Load changes data for an index."""
    changes_file = CHANGES_DIR / f"{index_key}_changes.json"
    if changes_file.exists():
        with open(changes_file, 'r') as f:
            data = json.load(f)
            return data.get('changes', [])
    return []


def generate_snapshots(index_key: str, current_symbols: list, changes: list):
    """Generate quarterly snapshots by working backward from current."""
    if not current_symbols:
        print(f"  No current symbols for {index_key}, skipping")
        return
    
    output_dir = SNAPSHOTS_DIR / index_key
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Start with current constituents
    constituents = set(current_symbols)
    expected_count = len(current_symbols)
    
    # Save current snapshot (2026 Q1)
    save_snapshot(output_dir, "2026_Q1", index_key.upper(), constituents, expected_count)
    
    # Sort changes by date descending
    sorted_changes = sorted(changes, key=lambda x: x.get('effective_date', ''), reverse=True)
    
    snapshots_created = 1
    
    for change in sorted_changes:
        effective_date = change.get('effective_date', '')
        additions = set(change.get('additions', []))
        exclusions = set(change.get('exclusions', []))
        
        if not effective_date:
            continue
        
        # Skip empty changes or changes with bad data (like "SR")
        valid_additions = {s for s in additions if len(s) > 2 and s != 'SR'}
        valid_exclusions = {s for s in exclusions if len(s) > 2 and s != 'SR'}
        
        if not valid_additions and not valid_exclusions:
            continue
        
        # Reverse the change: remove additions, add exclusions
        constituents = constituents - valid_additions
        constituents = constituents.union(valid_exclusions)
        
        # Determine quarter for this snapshot
        try:
            dt = datetime.strptime(effective_date, "%Y-%m-%d")
            if dt.month <= 3:
                quarter = f"{dt.year}_Q1"
            elif dt.month <= 6:
                quarter = f"{dt.year}_Q2"
            elif dt.month <= 9:
                quarter = f"{dt.year}_Q3"
            else:
                quarter = f"{dt.year}_Q4"
            
            print(f"    {effective_date}: -{len(valid_additions)} +{len(valid_exclusions)} = {len(constituents)} stocks")
            save_snapshot(output_dir, quarter, index_key.upper(), constituents, expected_count)
            snapshots_created += 1
            
        except ValueError:
            continue
    
    print(f"  Created {snapshots_created} snapshots for {index_key}")


def save_snapshot(output_dir: Path, quarter: str, index_name: str, constituents: set, expected_count: int):
    """Save a snapshot to JSON file."""
    output_file = output_dir / f"{quarter}.json"
    
    count = len(constituents)
    status = "OK" if count == expected_count else f"WARN: {count} vs {expected_count}"
    
    data = {
        "index": index_name,
        "quarter": quarter,
        "count": count,
        "expected_count": expected_count,
        "symbols": sorted(list(constituents)),
        "generated_at": datetime.now().isoformat()
    }
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)


def main():
    print("="*60)
    print("Historical Constituent Snapshot Generator")
    print("="*60)
    
    # Create NSE session
    print("\nInitializing NSE session...")
    session = get_nse_session()
    time.sleep(2)  # Wait for cookies
    
    # Process each index
    for index_key, index_name in INDICES.items():
        print(f"\n{'='*40}")
        print(f"Processing: {index_name}")
        print('='*40)
        
        # Fetch current constituents
        print(f"  Fetching current constituents...")
        symbols = fetch_index_constituents(session, index_name)
        
        if symbols:
            save_current_constituents(index_key, index_name, symbols)
            
            # Load changes and generate snapshots
            changes = load_changes(index_key)
            if changes:
                print(f"  Found {len(changes)} change records")
                generate_snapshots(index_key, symbols, changes)
            else:
                print(f"  No changes data, saving current only")
                output_dir = SNAPSHOTS_DIR / index_key
                output_dir.mkdir(parents=True, exist_ok=True)
                save_snapshot(output_dir, "2026_Q1", index_key.upper(), set(symbols), len(symbols))
        else:
            print(f"  Failed to fetch constituents, trying fallback...")
            # Use existing changes file if available
            changes = load_changes(index_key)
            if changes:
                print(f"  Using changes data only (no current baseline)")
        
        time.sleep(1)  # Rate limiting
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == '__main__':
    main()
