"""
Press Release Scraper for Nifty Indices
Extracts historical index constituent changes from niftyindices.com press releases

Usage:
    python scrape_press_releases.py --year 2024
    python scrape_press_releases.py --all
"""

import requests
from bs4 import BeautifulSoup
import re
import json
import os
from pathlib import Path
from datetime import datetime
import time
import pdfplumber
from io import BytesIO
from typing import List, Dict, Tuple, Optional


# Configuration
BASE_URL = "https://www.niftyindices.com/press-release"
PDF_URL_PREFIX = "https://www.niftyindices.com"
CHANGES_DIR = Path(__file__).parent / "data" / "changes"
RAW_PDF_DIR = Path(__file__).parent / "data" / "raw_pdfs"
SCREENSHOTS_DIR = Path(__file__).parent / "data" / "pdf_screenshots"
YEARS = list(range(2015, 2027))  # 2015 to 2026

# Setup requests session with proper headers
SESSION = requests.Session()
SESSION.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
})

# All 19 Broad Market Indices - NSE API names and variations
INDEX_NAME_MAP = {
    # Standard names -> normalized keys
    "NIFTY 50": "nifty50",
    "NIFTY50": "nifty50",
    "CNX NIFTY": "nifty50",  # Old name
    "NIFTY NEXT 50": "niftynext50",
    "NIFTY JUNIOR": "niftynext50",  # Old name
    "CNX NIFTY JUNIOR": "niftynext50",
    "NIFTY 100": "nifty100",
    "NIFTY 200": "nifty200",
    "NIFTY 500": "nifty500",
    "NIFTY MIDCAP 150": "niftymidcap150",
    "NIFTY MIDCAP 50": "niftymidcap50",
    "NIFTY MIDCAP SELECT": "niftymidselect",
    "NIFTY MID SELECT": "niftymidselect",
    "NIFTY MIDCAP 100": "niftymidcap100",
    "NIFTY SMALLCAP 250": "niftysmlcap250",
    "NIFTY SMLCAP 250": "niftysmlcap250",
    "NIFTY SMALLCAP 50": "niftysmlcap50",
    "NIFTY SMLCAP 50": "niftysmlcap50",
    "NIFTY SMALLCAP 100": "niftysmlcap100",
    "NIFTY SMLCAP 100": "niftysmlcap100",
    "NIFTY LARGEMIDCAP 250": "niftylargemid250",
    "NIFTY LARGEMID250": "niftylargemid250",
    "NIFTY MIDSMALLCAP 400": "niftymidsml400",
    "NIFTY MIDSML 400": "niftymidsml400",
    "NIFTY500 MULTICAP 50:25:25": "nifty500multicap",
    "NIFTY500 MULTICAP": "nifty500multicap",
    "NIFTY MICROCAP 250": "niftymicrocap250",
    "NIFTY MICROCAP250": "niftymicrocap250",
    "NIFTY TOTAL MARKET": "niftytotalmkt",
    "NIFTY TOTAL MKT": "niftytotalmkt",
    "NIFTY500 LARGEMIDSMALL EQUAL-CAP WEIGHTED": "nifty500lmseql",
    "NIFTY500 LMS EQL": "nifty500lmseql",
    "NIFTY INDIA FPI 150": "niftyfpi150",
    "NIFTY FPI 150": "niftyfpi150",
}


def get_press_releases(year: int) -> List[Dict]:
    """Fetch press release list for a given year."""
    url = f"{BASE_URL}?date={year}"
    
    try:
        response = SESSION.get(url, timeout=60)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        releases = []
        
        # Find all links that point to PDFs
        for link in soup.find_all('a', href=True):
            href = link['href']
            text = link.get_text(strip=True)
            
            # Filter for PDF links related to index changes
            if '.pdf' in href.lower():
                # Check if it's related to index changes
                keywords = ['change', 'replacement', 'indices', 'index', 'rebalancing', 
                           'inclusion', 'exclusion', 'revision', 'reconstitution']
                if any(kw in text.lower() for kw in keywords):
                    full_url = href if href.startswith('http') else PDF_URL_PREFIX + href
                    
                    # Extract date from URL or text
                    date_match = re.search(r'(\d{2})(\d{2})(\d{4})', href)
                    if date_match:
                        day, month, yr = date_match.groups()
                        date_str = f"{yr}-{month}-{day}"
                    else:
                        date_str = f"{year}-00-00"
                    
                    releases.append({
                        'title': text,
                        'url': full_url,
                        'date': date_str,
                        'year': year
                    })
        
        print(f"Found {len(releases)} index-related press releases for {year}")
        return releases
        
    except Exception as e:
        print(f"Error fetching {year}: {e}")
        return []


def download_pdf(url: str) -> Optional[bytes]:
    """Download PDF content."""
    try:
        response = SESSION.get(url, timeout=120)
        response.raise_for_status()
        return response.content
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return None


def extract_changes_from_pdf(pdf_content: bytes, press_release: Dict) -> Dict[str, Dict]:
    """
    Parse PDF and extract index constituent changes.
    NSE PDFs have tables with [Sr. No., Company Name, Symbol] structure.
    The context (Excluded/Included) comes from text before each table.
    Returns: {index_key: {'additions': [], 'exclusions': [], 'date': str}}
    """
    changes = {}
    
    try:
        with pdfplumber.open(BytesIO(pdf_content)) as pdf:
            # Process each page
            for page in pdf.pages:
                text = page.extract_text() or ""
                tables = page.extract_tables() or []
                
                # Find positions of key markers in text
                text_lower = text.lower()
                
                # Track current context (which index, excluded/included)
                current_index_key = None
                is_exclusion = False
                
                # Process tables on this page
                for table in tables:
                    if not table or len(table) < 2:
                        continue
                    
                    # Check if this is a stock table [Sr. No., Company Name, Symbol]
                    headers = [str(cell).lower().strip() if cell else '' for cell in table[0]]
                    
                    # Look for Symbol column
                    symbol_col = None
                    for i, h in enumerate(headers):
                        if 'symbol' in h:
                            symbol_col = i
                            break
                    
                    if symbol_col is None:
                        continue
                    
                    # Extract symbols from table
                    symbols = []
                    for row in table[1:]:
                        if row and len(row) > symbol_col and row[symbol_col]:
                            symbol = str(row[symbol_col]).strip().upper()
                            # Validate it looks like a stock symbol
                            if symbol and len(symbol) >= 2 and len(symbol) <= 20:
                                if re.match(r'^[A-Z0-9\-&]+$', symbol):
                                    symbols.append(symbol)
                    
                    if not symbols:
                        continue
                    
                    # Try to find context from text above table
                    # Look for index name and excluded/included keywords
                    for index_name, index_key in INDEX_NAME_MAP.items():
                        if index_name.upper() in text.upper() or index_name.replace(' ', '').upper() in text.upper():
                            # Check if this section is about exclusion or inclusion
                            # Look for patterns like "Excluded from NIFTY 50" or "following are excluded"
                            excl_patterns = ['excluded', 'exclusion', 'removed', 'will be removed', 'deletion']
                            incl_patterns = ['included', 'inclusion', 'added', 'will be added', 'addition']
                            
                            # Check which pattern comes first before this index mention
                            is_exclusion = any(p in text_lower for p in excl_patterns)
                            is_inclusion = any(p in text_lower for p in incl_patterns)
                            
                            # Default: first table after index name is usually exclusions
                            if index_key not in changes:
                                changes[index_key] = {
                                    'additions': [],
                                    'exclusions': [],
                                    'date': press_release['date'],
                                    'source': press_release['url']
                                }
                            
                            # Heuristic: if 'exclud' appears before 'includ' in text, first table is exclusions
                            excl_pos = text_lower.find('exclud')
                            incl_pos = text_lower.find('includ')
                            
                            if excl_pos >= 0 and (incl_pos < 0 or excl_pos < incl_pos):
                                # This is likely an exclusion table
                                changes[index_key]['exclusions'].extend(symbols)
                            elif incl_pos >= 0:
                                # This is likely an inclusion table
                                changes[index_key]['additions'].extend(symbols)
                            else:
                                # Can't determine, add to exclusions by default (first tables are usually exclusions)
                                if not changes[index_key]['exclusions']:
                                    changes[index_key]['exclusions'].extend(symbols)
                                else:
                                    changes[index_key]['additions'].extend(symbols)
                            break
    
    except Exception as e:
        print(f"Error parsing PDF: {e}")
    
    # Remove duplicates
    for index_key in changes:
        changes[index_key]['additions'] = list(set(changes[index_key]['additions']))
        changes[index_key]['exclusions'] = list(set(changes[index_key]['exclusions']))
    
    return changes


def extract_symbols(text: str) -> List[str]:
    """Extract stock symbols from text."""
    # Common patterns: "RELIANCE", "TCS LTD", "HDFC BANK LTD"
    # Clean and split
    text = re.sub(r'\s+', ' ', text.strip())
    # Remove common suffixes
    text = re.sub(r'\s+(LTD|LIMITED|CORP|CORPORATION|INC)\b', '', text, flags=re.IGNORECASE)
    
    # Split by common delimiters
    parts = re.split(r'[,\n;|]', text)
    
    symbols = []
    for part in parts:
        part = part.strip()
        if part and len(part) >= 2 and len(part) <= 20:
            # Normalize to uppercase
            symbol = part.upper().replace(' ', '')
            if re.match(r'^[A-Z0-9\-&]+$', symbol):
                symbols.append(symbol)
    
    return symbols


def parse_table_for_changes(table: List[List], press_release: Dict) -> Dict[str, Dict]:
    """Parse a table for index changes."""
    changes = {}
    
    # Try to identify table structure
    if not table or len(table) < 2:
        return changes
    
    headers = [str(cell).lower() if cell else '' for cell in table[0]]
    
    # Look for inclusion/exclusion columns
    add_col = None
    exc_col = None
    index_col = None
    
    for i, header in enumerate(headers):
        if 'inclus' in header or 'add' in header:
            add_col = i
        elif 'exclus' in header or 'remov' in header:
            exc_col = i
        elif 'index' in header or 'nifty' in header:
            index_col = i
    
    # Process rows
    for row in table[1:]:
        if not row or len(row) < 2:
            continue
        
        # Try to match index name
        for index_name, index_key in INDEX_NAME_MAP.items():
            row_text = ' '.join(str(cell) for cell in row if cell)
            if index_name.upper() in row_text.upper():
                additions = []
                exclusions = []
                
                if add_col is not None and add_col < len(row) and row[add_col]:
                    additions = extract_symbols(str(row[add_col]))
                if exc_col is not None and exc_col < len(row) and row[exc_col]:
                    exclusions = extract_symbols(str(row[exc_col]))
                
                if additions or exclusions:
                    changes[index_key] = {
                        'additions': additions,
                        'exclusions': exclusions,
                        'date': press_release['date'],
                        'source': press_release['url']
                    }
                break
    
    return changes


def save_changes(all_changes: Dict[str, List[Dict]]):
    """Save extracted changes to JSON files."""
    CHANGES_DIR.mkdir(parents=True, exist_ok=True)
    
    for index_key, changes in all_changes.items():
        output_file = CHANGES_DIR / f"{index_key}_changes.json"
        
        # Load existing data if present
        existing_data = {'index': index_key.upper(), 'changes': []}
        if output_file.exists():
            try:
                with open(output_file, 'r') as f:
                    existing_data = json.load(f)
            except:
                pass
        
        # Merge changes (avoid duplicates by date)
        existing_dates = {c.get('effective_date') for c in existing_data.get('changes', [])}
        
        for change in changes:
            if change.get('date') not in existing_dates:
                existing_data['changes'].append({
                    'effective_date': change.get('date'),
                    'additions': change.get('additions', []),
                    'exclusions': change.get('exclusions', []),
                    'source': change.get('source', '')
                })
        
        # Sort by date descending
        existing_data['changes'].sort(key=lambda x: x.get('effective_date', ''), reverse=True)
        existing_data['last_updated'] = datetime.now().strftime('%Y-%m-%d')
        
        with open(output_file, 'w') as f:
            json.dump(existing_data, f, indent=4)
        
        print(f"Saved {len(existing_data['changes'])} changes to {output_file.name}")


def scrape_year(year: int) -> Dict[str, List[Dict]]:
    """Scrape all press releases for a given year."""
    print(f"\n{'='*50}")
    print(f"Scraping year {year}")
    print('='*50)
    
    all_changes = {}
    
    releases = get_press_releases(year)
    
    for release in releases:
        print(f"\nProcessing: {release['title'][:60]}...")
        
        pdf_content = download_pdf(release['url'])
        if pdf_content:
            # Save raw PDF for reference
            RAW_PDF_DIR.mkdir(parents=True, exist_ok=True)
            pdf_name = f"{release['date']}_{release['url'].split('/')[-1]}"
            with open(RAW_PDF_DIR / pdf_name, 'wb') as f:
                f.write(pdf_content)
            
            changes = extract_changes_from_pdf(pdf_content, release)
            
            for index_key, data in changes.items():
                if index_key not in all_changes:
                    all_changes[index_key] = []
                all_changes[index_key].append(data)
                print(f"  Found changes for {index_key}: +{len(data.get('additions', []))} -{len(data.get('exclusions', []))}")
        
        time.sleep(1)  # Be respectful of the server
    
    return all_changes


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Scrape niftyindices.com press releases')
    parser.add_argument('--year', type=int, help='Specific year to scrape')
    parser.add_argument('--all', action='store_true', help='Scrape all years (2015-2026)')
    parser.add_argument('--test', action='store_true', help='Test mode - just print found URLs')
    
    args = parser.parse_args()
    
    if args.test:
        # Test mode - just list press releases
        for year in (YEARS if args.all else [args.year or 2024]):
            releases = get_press_releases(year)
            for r in releases:
                print(f"  {r['date']}: {r['title'][:50]}... -> {r['url']}")
        return
    
    all_changes = {}
    
    if args.all:
        for year in YEARS:
            year_changes = scrape_year(year)
            for index_key, changes in year_changes.items():
                if index_key not in all_changes:
                    all_changes[index_key] = []
                all_changes[index_key].extend(changes)
    elif args.year:
        all_changes = scrape_year(args.year)
    else:
        print("Please specify --year YYYY or --all")
        return
    
    # Save all changes
    save_changes(all_changes)
    
    print(f"\n{'='*50}")
    print("Summary")
    print('='*50)
    for index_key, changes in all_changes.items():
        print(f"  {index_key}: {len(changes)} change records")


if __name__ == '__main__':
    main()
