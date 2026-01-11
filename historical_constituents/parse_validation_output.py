"""
Parse raw output from process_local_pdfs.py to extract potential NIFTY 100/Next 50 changes.
"""

import re
from pathlib import Path
import json

def parse_log_file(filename):
    try:
        with open(filename, 'r', encoding='utf-16') as f:
            content = f.read()
    except UnicodeError:
        # Fallback to default if not utf-16
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    
    # Split by file
    sections = content.split("--- ")
    
    events = []
    
    for section in sections:
        if not section.strip(): continue
        
        lines = section.split('\n')
        pdf_name = lines[0].strip(' -')
        
        # Extract date
        date_match = re.search(r'Date: (\d{4}-\d{2}-\d{2})', section)
        date_str = date_match.group(1) if date_match else "Unknown"
        
        # Check relevance
        if "NIFTY NEXT 50" not in section.upper() and "NIFTY 100" not in section.upper() and "NIFTY 50" not in section.upper():
            continue
            
        # Find tables and symbols
        # This is a bit loose because the log format is loose
        # We look for "Symbols: ..." lines
        
        # Try to infer context (exclusion/inclusion) by looking at text before symbols
        # Default assumption: First list is Exclusions, Second is Additions (standard NSE format)
        # But we check for keywords
        
        table_matches = list(re.finditer(r'Symbols: (.*?) \(\d+ total\)', section))
        
        if not table_matches:
            continue
            
        # Basic heuristic parsing
        found_symbols = []
        for m in table_matches:
            s_list = m.group(1).split(', ')
            found_symbols.append(s_list)
            
        events.append({
            'date': date_str,
            'file': pdf_name,
            'tables': found_symbols,
            'raw_section': section[:500] + "..." # Snippet
        })
        
    return events

def main():
    recent = parse_log_file("recent_years.txt") if Path("recent_years.txt").exists() else []
    old = parse_log_file("old_years.txt") if Path("old_years.txt").exists() else []
    
    all_events = sorted(recent + old, key=lambda x: x['date'], reverse=True)
    
    print(f"Found {len(all_events)} relevant PDF events related to NIFTY Indices")
    
    for e in all_events:
        print(f"\nDate: {e['date']} | File: {e['file']}")
        if "next 50" in e['raw_section'].lower() or "nifty 100" in e['raw_section'].lower() or "nifty next 50" in e['raw_section'].lower():
            print(f"  *** EXTRACTING POTENTIAL NIFTY NEXT 50/100 CHANGES ***")
            for i, tbl in enumerate(e['tables']):
                print(f"  Table {i+1} ({len(tbl)}): {', '.join(tbl)}")

if __name__ == "__main__":
    main()
