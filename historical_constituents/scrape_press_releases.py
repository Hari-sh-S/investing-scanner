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
    Returns: {index_key: {'additions': [], 'exclusions': [], 'date': str}}
    """
    changes = {}
    
    try:
        with pdfplumber.open(BytesIO(pdf_content)) as pdf:
            full_text = ""
            tables = []
            
            for page in pdf.pages:
                text = page.extract_text() or ""
                full_text += text + "\n"
                
                # Extract tables
                page_tables = page.extract_tables()
                for table in page_tables:
                    tables.append(table)
            
            # Parse the text and tables for each index
            current_index = None
            additions = []
            exclusions = []
            
            # Patterns to identify index sections
            for index_name, index_key in INDEX_NAME_MAP.items():
                # Look for index name in text followed by additions/exclusions
                pattern = re.compile(
                    rf'{re.escape(index_name)}.*?(?:inclusion|addition|include)s?.*?:\s*([\w\s,]+?)(?:exclusion|remove|exclude)s?.*?:\s*([\w\s,]+)',
                    re.IGNORECASE | re.DOTALL
                )
                
                match = pattern.search(full_text)
                if match:
                    add_text = match.group(1)
                    exc_text = match.group(2)
                    
                    # Extract stock symbols
                    add_symbols = extract_symbols(add_text)
                    exc_symbols = extract_symbols(exc_text)
                    
                    if add_symbols or exc_symbols:
                        if index_key not in changes:
                            changes[index_key] = {
                                'additions': [],
                                'exclusions': [],
                                'date': press_release['date'],
                                'source': press_release['url']
                            }
                        changes[index_key]['additions'].extend(add_symbols)
                        changes[index_key]['exclusions'].extend(exc_symbols)
            
            # Also try to parse tables
            for table in tables:
                if table and len(table) > 1:
                    parsed = parse_table_for_changes(table, press_release)
                    for index_key, data in parsed.items():
                        if index_key not in changes:
                            changes[index_key] = data
                        else:
                            changes[index_key]['additions'].extend(data.get('additions', []))
                            changes[index_key]['exclusions'].extend(data.get('exclusions', []))
    
    except Exception as e:
        print(f"Error parsing PDF: {e}")
    
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
