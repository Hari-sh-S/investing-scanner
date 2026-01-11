"""
Generate NIFTY 100 historical snapshots from clean verified changes data.

Uses the current NIFTY 100 constituents and works backward through
verified changes to reconstruct historical snapshots.
"""

import json
from pathlib import Path
from datetime import datetime, date

DATA_DIR = Path(__file__).parent / "data"
CURRENT_DIR = DATA_DIR / "current_constituents"
CHANGES_FILE = DATA_DIR / "changes" / "nifty100_changes_clean.json"
SNAPSHOTS_DIR = DATA_DIR / "index_constituents" / "nifty100"


def load_current_constituents() -> set:
    """Load current NIFTY 100 constituents."""
    file_path = CURRENT_DIR / "nifty100.json"
    with open(file_path, 'r') as f:
        data = json.load(f)
        return set(data.get('symbols', []))


def load_clean_changes() -> list:
    """Load clean verified changes."""
    with open(CHANGES_FILE, 'r') as f:
        data = json.load(f)
        return data.get('changes', [])


def save_snapshot(quarter: str, symbols: set, effective_date: str = None):
    """Save a snapshot."""
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = SNAPSHOTS_DIR / f"{quarter}.json"
    
    data = {
        "index": "NIFTY100",
        "quarter": quarter,
        "count": len(symbols),
        "expected_count": 100,
        "symbols": sorted(list(symbols)),
        "generated_at": datetime.now().isoformat()
    }
    
    if effective_date:
        data["effective_date"] = effective_date
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)
    
    return len(symbols)


def date_to_quarter(date_str: str) -> str:
    """Convert date string to quarter format."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    if dt.month <= 3:
        return f"{dt.year}_Q1"
    elif dt.month <= 6:
        return f"{dt.year}_Q2"
    elif dt.month <= 9:
        return f"{dt.year}_Q3"
    else:
        return f"{dt.year}_Q4"


def get_previous_quarter(quarter: str) -> str:
    """Get the previous quarter."""
    year, q = quarter.split("_")
    year = int(year)
    q_num = int(q[1])
    
    if q_num == 1:
        return f"{year-1}_Q4"
    else:
        return f"{year}_Q{q_num-1}"


def main():
    print("=" * 60)
    print("NIFTY 100 Clean Historical Snapshot Generator")
    print("=" * 60)
    
    # Load current constituents
    current = load_current_constituents()
    print(f"\nCurrent NIFTY 100: {len(current)} stocks")
    
    # Load clean changes
    changes = load_clean_changes()
    print(f"Verified changes: {len(changes)} records")
    
    # Sort changes by date descending
    changes_sorted = sorted(changes, key=lambda x: x['effective_date'], reverse=True)
    
    # Track the constituents as we go back in time
    constituents = current.copy()
    
    # Save current (2026 Q1)
    count = save_snapshot("2026_Q1", constituents)
    print(f"\nSaved 2026_Q1: {count} stocks (current)")
    
    # Generate quarterly snapshots
    quarters_saved = {"2026_Q1"}
    
    for change in changes_sorted:
        effective_date = change['effective_date']
        additions = set(change.get('additions', []))
        exclusions = set(change.get('exclusions', []))
        
        # Reverse the change: remove what was added, add what was excluded
        constituents = constituents - additions
        constituents = constituents.union(exclusions)
        
        # Determine which quarter this represents (the state BEFORE this change)
        quarter = date_to_quarter(effective_date)
        prev_quarter = get_previous_quarter(quarter)
        
        # Save snapshot for the quarter before the change
        if prev_quarter not in quarters_saved:
            count = save_snapshot(prev_quarter, constituents, effective_date)
            quarters_saved.add(prev_quarter)
            print(f"Saved {prev_quarter}: {count} stocks (before {effective_date})")
    
    # Fill in gaps with the nearest known state
    print("\n--- Filling missing quarters ---")
    
    # Get all quarters from 2020 Q1 to current
    all_quarters = []
    for year in range(2020, 2027):
        for q in [1, 2, 3, 4]:
            qstr = f"{year}_Q{q}"
            if qstr == "2026_Q2":
                break
            all_quarters.append(qstr)
    
    # Fill gaps: for each missing quarter, copy from the next known quarter
    known_quarters = sorted(quarters_saved)
    
    for q in all_quarters:
        if q not in quarters_saved:
            # Find the nearest future known snapshot
            snapshot_file = SNAPSHOTS_DIR / f"{q}.json"
            if not snapshot_file.exists():
                # Find nearest known snapshot
                for known in sorted(known_quarters, reverse=True):
                    if known > q:
                        # Copy this snapshot
                        src_file = SNAPSHOTS_DIR / f"{known}.json"
                        with open(src_file, 'r') as f:
                            data = json.load(f)
                        
                        data['quarter'] = q
                        data['generated_at'] = datetime.now().isoformat()
                        data['note'] = f"Interpolated from {known}"
                        
                        with open(snapshot_file, 'w') as f:
                            json.dump(data, f, indent=4)
                        
                        print(f"Filled {q}: {data['count']} stocks (from {known})")
                        break
    
    # Final verification
    print("\n" + "=" * 60)
    print("Final Verification")
    print("=" * 60)
    
    for f in sorted(SNAPSHOTS_DIR.glob("*.json")):
        with open(f, 'r') as file:
            data = json.load(file)
            status = "✓" if data['count'] == 100 else f"⚠ {data['count']}"
            print(f"{f.stem}: {status}")


if __name__ == "__main__":
    main()
