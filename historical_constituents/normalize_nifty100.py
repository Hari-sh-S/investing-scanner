"""
Normalize NIFTY 100 snapshots to exactly 100 stocks.

For snapshots with more than 100 stocks, keep only the stocks
that are also in current NIFTY 100 or in adjacent snapshots.
"""

import json
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
SNAPSHOTS_DIR = DATA_DIR / "index_constituents" / "nifty100"
CURRENT_FILE = DATA_DIR / "current_constituents" / "nifty100.json"


def main():
    # Load current constituents as reference
    with open(CURRENT_FILE, 'r') as f:
        current = set(json.load(f)['symbols'])
    
    print("Normalizing NIFTY 100 snapshots to 100 stocks each\n")
    
    # Process each snapshot
    for f in sorted(SNAPSHOTS_DIR.glob("*.json")):
        with open(f, 'r') as file:
            data = json.load(file)
        
        symbols = set(data['symbols'])
        count = len(symbols)
        
        if count == 100:
            print(f"{f.stem}: Already 100 stocks")
            continue
        
        if count > 100:
            # Keep stocks that are in current index (most likely correct)
            # Then keep oldest stocks to fill to 100
            in_current = symbols & current
            not_in_current = symbols - current
            
            # Keep all that are in current, remove extras from not_in_current
            excess = count - 100
            to_remove = list(not_in_current)[:excess]
            
            new_symbols = symbols - set(to_remove)
            
            data['symbols'] = sorted(list(new_symbols))
            data['count'] = len(new_symbols)
            data['normalized'] = True
            data['removed_stocks'] = to_remove
            
            with open(f, 'w') as file:
                json.dump(data, file, indent=4)
            
            print(f"{f.stem}: {count} -> {len(new_symbols)} (removed {to_remove})")
        
        elif count < 100:
            print(f"{f.stem}: Only {count} stocks (need to add {100 - count})")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
