"""
Process local PDF files to extract NIFTY 100 and NIFTY NEXT 50 changes.
"""

import pdfplumber
import re
import os
from pathlib import Path
from typing import List, Dict, Optional
import json

# Configuration
RAW_PDF_DIR = Path(__file__).parent / "data" / "raw_pdfs"
INDEX_NAME_MAP = {
    "NIFTY 50": "nifty50",
    "NIFTY NEXT 50": "niftynext50",
    "NIFTY 100": "nifty100",
}

def extract_symbols(text: str) -> List[str]:
    """Extract stock symbols from text."""
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'\s+(LTD|LIMITED|CORP|CORPORATION|INC)\b', '', text, flags=re.IGNORECASE)
    parts = re.split(r'[,\n;|]', text)
    symbols = []
    for part in parts:
        part = part.strip()
        if part and len(part) >= 2 and len(part) <= 20:
            symbol = part.upper().replace(' ', '')
            if re.match(r'^[A-Z0-9\-&]+$', symbol):
                symbols.append(symbol)
    return symbols

def extract_changes_from_pdf(pdf_path: Path) -> Dict[str, Dict]:
    changes = {}
    print(f"Processing {pdf_path.name}...")
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            full_text = ""
            for page in pdf.pages:
                text = page.extract_text() or ""
                full_text += text + "\n"
                tables = page.extract_tables() or []
                
                text_lower = text.lower()
                
                for table in tables:
                    if not table or len(table) < 2:
                        continue
                        
                    # Check headers
                    headers = [str(cell).lower().strip() if cell else '' for cell in table[0]]
                    
                    symbol_col = -1
                    for i, h in enumerate(headers):
                        if 'symbol' in h:
                            symbol_col = i
                            break
                    
                    if symbol_col == -1:
                        continue
                        
                    symbols = []
                    for row in table[1:]:
                        if row and len(row) > symbol_col and row[symbol_col]:
                            symbol = str(row[symbol_col]).strip().upper()
                            if symbol and len(symbol) >= 2 and len(symbol) <= 20:
                                if re.match(r'^[A-Z0-9\-&]+$', symbol):
                                    symbols.append(symbol)
                    
                    if not symbols:
                        continue

                    # Context parsing
                    # We look for "Nifty 100" or "Nifty Next 50" around the table
                    # This is a bit heuristic. We can look at text before table if possible,
                    # but pdfplumber extracts text separately. 
                    
                    # Search in the extracted text for index names
                    for index_name, index_key in INDEX_NAME_MAP.items():
                        if index_name.upper() in text.upper():
                             # Check if exclusion or inclusion
                            is_exclusion = any(p in text_lower for p in ['excluded', 'exclusion', 'remove'])
                            is_inclusion = any(p in text_lower for p in ['included', 'inclusion', 'add'])
                            
                            # Refine logic: normally exclusion table comes first
                            # But let's just dump what we find for manual verification if ambiguous
                            
                            if index_key not in changes:
                                changes[index_key] = {'additions': [], 'exclusions': [], 'file': pdf_path.name}

                            # Basic heuristic: if we see "exclusion" and not "inclusion" nearby, it's exclusion
                            # Or if table has header "Include" or "Exclude"
                            
                            # Check table headers for clues
                            header_str = " ".join(headers)
                            if 'exclu' in header_str:
                                changes[index_key]['exclusions'].extend(symbols)
                            elif 'inclu' in header_str or 'add' in header_str:
                                changes[index_key]['additions'].extend(symbols)
                            else:
                                # Fallback to text context
                                # If both present, it's hard. 
                                # Let's assume exclusions usually listed first in NSE PDFs
                                # But we can't rely on that 100% without table position
                                pass
                                
                            # Simplification: For now, I will extract ALL symbols along with the index name found
                            # and print them for user validation, rather than trusting auto-classification fully
                            # because the user wants "revalidate... by going through each pdf"
                            
                            # actually, let's just store them and print them out clearly
                            pass

    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        
    return changes

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=str, help="Process only files containing this year (e.g. 2024)")
    args = parser.parse_args()

    if not RAW_PDF_DIR.exists():
        print(f"Directory not found: {RAW_PDF_DIR}")
        return

    pdf_files = sorted(list(RAW_PDF_DIR.glob("*.pdf")))
    
    if args.year:
        print(f"Filtering for year: {args.year}")
        pdf_files = [f for f in pdf_files if args.year in f.name]
        
    print(f"Found {len(pdf_files)} PDF files to process.")
    
    all_findings = []

    for pdf_file in pdf_files:
        try:
            with pdfplumber.open(pdf_file) as pdf:
                full_text = ""
                for page in pdf.pages:
                    full_text += (page.extract_text() or "") + "\n"
                
                # Simple keyword search first to see if relevant
                relevant = False
                for idx in INDEX_NAME_MAP.keys():
                    if idx.upper() in full_text.upper():
                        relevant = True
                        break
                
                if relevant:
                    # Extract tables to be more precise
                    findings = {'file': pdf_file.name, 'indices': {}}
                    
                    for page in pdf.pages:
                        tables = page.extract_tables()
                        text = page.extract_text() or ""
                        
                        for table in tables:
                            # Try to identify what this table is for
                            # Look for symbols
                            has_symbols = False
                            symbols = []
                            
                            # Flatten table to find symbols
                            for row in table:
                                for cell in row:
                                    if cell:
                                        s = str(cell).strip().upper()
                                        if len(s) > 2 and len(s) < 15 and s.isalpha():
                                            # potentially a symbol
                                            pass
                            
                            # Actually, rely on the `extract_changes_from_pdf` logic from previous script
                            # but improved for reporting
                            pass
                            
                    # Let's use a simpler text based approach for reporting
                    print(f"\n--- {pdf_file.name} ---")
                    
                    # Extract date from filename
                    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', pdf_file.name)
                    date_str = date_match.group(1) if date_match else "Unknown Date"
                    print(f"Date: {date_str}")

                    # Check for Nifty 100 / Nifty Next 50 mentions
                    for index_name in ["NIFTY 100", "NIFTY NEXT 50", "NIFTY 50"]: # Nifty 50 also relevant
                        if index_name in full_text.upper():
                            print(f"  Mentioned: {index_name}")
                            # TODO: Extract specific changes if possible
                            # For now, just flagging it is useful
                    
                    # Try to parse tables
                    for page in pdf.pages:
                        tables = page.extract_tables()
                        for table in tables:
                            if not table: continue
                            # Naive print of potential stock lists
                            # Check if header has 'Symbol'
                            headers = [str(c).lower() for c in table[0] if c]
                            if any('symbol' in h for h in headers):
                                # Print first few rows to see what it is
                                print(f"    Table found with headers: {headers}")
                                symbols = []
                                for row in table[1:]:
                                    # find symbol col
                                    for i, h in enumerate(headers):
                                        if 'symbol' in h and i < len(row) and row[i]:
                                            symbols.append(row[i])
                                if symbols:
                                    print(f"    Symbols: {', '.join(map(str, symbols[:5]))} ... ({len(symbols)} total)")

        except Exception as e:
            print(f"Error processing {pdf_file.name}: {e}")

if __name__ == "__main__":
    main()
