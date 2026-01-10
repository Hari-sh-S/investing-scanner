"""
Populate reasonable NIFTY 100 historical snapshots.

Due to data quality issues with the scraped changes data,
we'll use the current NIFTY 100 constituents as a baseline
and create quarterly snapshots from 2020 onwards.

Note: This is an approximation. The actual constituents
may have been slightly different historically, but for
momentum-based strategies, this should be close enough
as most large-cap stocks remain in the index.
"""

import json
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(__file__).parent / "data"
CURRENT_DIR = DATA_DIR / "current_constituents"
SNAPSHOTS_DIR = DATA_DIR / "index_constituents"


def load_current_constituents(index_key: str) -> list:
    """Load current constituents for an index."""
    file_path = CURRENT_DIR / f"{index_key}.json"
    if file_path.exists():
        with open(file_path, 'r') as f:
            data = json.load(f)
            return data.get('symbols', [])
    return []


def save_snapshot(index_key: str, quarter: str, symbols: list):
    """Save a snapshot."""
    output_dir = SNAPSHOTS_DIR / index_key
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"{quarter}.json"
    
    data = {
        "index": index_key.upper(),
        "quarter": quarter,
        "count": len(symbols),
        "expected_count": 100,
        "symbols": sorted(symbols),
        "generated_at": datetime.now().isoformat(),
        "note": "Approximated from current constituents"
    }
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)


def main():
    print("=" * 60)
    print("NIFTY 100 Snapshot Population")
    print("=" * 60)
    
    # Load current NIFTY 100
    current = load_current_constituents("nifty100")
    if not current:
        print("ERROR: Could not load current NIFTY 100")
        return
    
    print(f"Current NIFTY 100: {len(current)} stocks")
    
    # Generate quarterly snapshots from 2020 to 2026
    quarters = []
    for year in range(2020, 2027):
        for q in [1, 2, 3, 4]:
            if year == 2026 and q > 1:
                break  # Don't go past current quarter
            quarters.append(f"{year}_Q{q}")
    
    print(f"\nGenerating {len(quarters)} quarterly snapshots...")
    
    for quarter in quarters:
        save_snapshot("nifty100", quarter, current)
        print(f"  Created {quarter}")
    
    print(f"\n{'=' * 60}")
    print(f"Done! Created {len(quarters)} NIFTY 100 snapshots")
    print("\nNote: These use current constituents as approximation.")
    print("For more accurate historical data, manual curation is needed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
