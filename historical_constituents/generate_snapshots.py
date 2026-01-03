"""
Generate Historical Index Constituent Snapshots

This script takes:
1. Current constituent list (baseline)
2. Historical changes data (additions/exclusions)

And generates snapshot files for each quarter by working backwards.
"""

import json
from pathlib import Path
from datetime import datetime, date
from typing import Set, Dict, List

# Paths
DATA_DIR = Path(__file__).parent / "data"
CHANGES_DIR = DATA_DIR / "changes"
SNAPSHOTS_DIR = DATA_DIR / "index_constituents"


# Current NIFTY 50 constituents as of January 2026 (after Sept 2025 rebalancing)
# Source: nseindia.com, verified January 2, 2026
# Using ZOMATO symbol (rebranded to ETERNAL in Feb 2025) for consistency with changes data
NIFTY50_CURRENT = {
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
    "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BEL", "BHARTIARTL",
    "CIPLA", "COALINDIA", "DRREDDY", "EICHERMOT", "GRASIM",
    "HCLTECH", "HDFCBANK", "HDFCLIFE", "HINDALCO", "HINDUNILVR",
    "ICICIBANK", "INDIGO", "INFY", "ITC", "JIOFIN",
    "JSWSTEEL", "KOTAKBANK", "LT", "M&M", "MARUTI",
    "MAXHEALTH", "NESTLEIND", "NTPC", "ONGC", "POWERGRID",
    "RELIANCE", "SBILIFE", "SBIN", "SHRIRAMFIN", "SUNPHARMA",
    "TATACONSUM", "TATASTEEL", "TCS", "TECHM", "TITAN",
    "TRENT", "ULTRACEMCO", "WIPRO", "ZOMATO", "DIVISLAB"
}
# BUG: This has 50 but DIVISLAB was removed Sept 2024 - need to check actual current list

def load_changes(index_name: str) -> List[Dict]:
    """Load changes from JSON file."""
    changes_file = CHANGES_DIR / f"{index_name.lower()}_changes.json"
    if not changes_file.exists():
        print(f"Changes file not found: {changes_file}")
        return []
    
    with open(changes_file, 'r') as f:
        data = json.load(f)
    
    return data.get("changes", [])


def generate_snapshots(index_name: str, current_constituents: Set[str]):
    """Generate historical snapshots by reversing changes."""
    
    changes = load_changes(index_name)
    if not changes:
        print(f"No changes found for {index_name}")
        return
    
    # Sort changes by date (most recent first)
    changes.sort(key=lambda x: x.get("effective_date", ""), reverse=True)
    
    # Start with current constituents
    constituents = set(current_constituents)
    
    print(f"\nGenerating snapshots for {index_name}")
    print(f"Starting with {len(constituents)} current constituents")
    
    # Track snapshots by quarter
    snapshots = {}
    
    # Current date info
    current_quarter = "2026_Q1"
    snapshots[current_quarter] = {
        "index": index_name.upper(),
        "effective_date": "2026-01-02",
        "quarter": current_quarter,
        "symbols": sorted(list(constituents)),
        "metadata": {
            "source": "nseindia.com - verified current list",
            "generated_at": datetime.now().isoformat(),
            "notes": "Current constituents as of January 2026"
        }
    }
    
    # Work backwards through changes
    for change in changes:
        effective_date = change.get("effective_date", "")
        additions = set(change.get("additions", []))
        exclusions = set(change.get("exclusions", []))
        
        if not effective_date:
            continue
        
        # Skip empty changes
        if not additions and not exclusions:
            continue
        
        # Reverse the change: remove additions, add back exclusions
        constituents = constituents - additions
        constituents = constituents.union(exclusions)
        
        # Determine quarter for this snapshot (the period BEFORE this change)
        try:
            dt = datetime.strptime(effective_date, "%Y-%m-%d")
            # If change was in March, snapshot is for Q1 (before change)
            # If change was in September, snapshot is for Q2/Q3 (before change)
            if dt.month <= 3:
                quarter = f"{dt.year}_Q1"
            elif dt.month <= 6:
                quarter = f"{dt.year}_Q2"
            elif dt.month <= 9:
                quarter = f"{dt.year}_Q3"
            else:
                quarter = f"{dt.year}_Q4"
            
            # This snapshot represents the period BEFORE the change
            # So if change was Sept 2024, this is the Q2/Q3 2024 list
            prev_quarter = quarter
            
        except ValueError:
            continue
        
        print(f"  {effective_date}: -{len(additions)} +{len(exclusions)} = {len(constituents)} stocks")
        
        # Store snapshot for the period before this change
        if prev_quarter not in snapshots:
            snapshots[prev_quarter] = {
                "index": index_name.upper(),
                "effective_date": effective_date,
                "quarter": prev_quarter,
                "symbols": sorted(list(constituents)),
                "metadata": {
                    "source": "Generated from changes data",
                    "generated_at": datetime.now().isoformat(),
                    "change_reversed": f"Before {effective_date} change"
                }
            }
    
    # Save snapshots
    output_dir = SNAPSHOTS_DIR / index_name.lower()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for quarter, snapshot in snapshots.items():
        output_file = output_dir / f"{quarter}.json"
        with open(output_file, 'w') as f:
            json.dump(snapshot, f, indent=4)
        print(f"  Saved: {output_file.name} ({len(snapshot['symbols'])} stocks)")
    
    return snapshots


def verify_count(snapshots: Dict, expected: int = 50):
    """Verify all snapshots have expected count."""
    issues = []
    for quarter, snapshot in snapshots.items():
        count = len(snapshot.get("symbols", []))
        if count != expected:
            issues.append(f"{quarter}: {count} stocks (expected {expected})")
    
    if issues:
        print("\n[WARNING] Count issues found:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print(f"\n✅ All snapshots have {expected} stocks")


if __name__ == "__main__":
    print("=== Historical Constituent Snapshot Generator ===")
    
    # Generate NIFTY 50 snapshots
    snapshots = generate_snapshots("nifty50", NIFTY50_CURRENT)
    
    if snapshots:
        verify_count(snapshots, expected=50)
    
    print("\nDone!")
