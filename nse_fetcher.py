"""
NSE Stock List Fetcher
Fetches live stock lists from NSE India official sources
"""

import requests
import pandas as pd
import json
from typing import List, Dict
import time

class NSEFetcher:
    """Fetch live stock lists from NSE India"""

    BASE_URL = "https://www.nseindia.com"

    # NSE Index URLs
    INDEX_URLS = {
        "NIFTY 50": "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050",
        "NIFTY NEXT 50": "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20NEXT%2050",
        "NIFTY 100": "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20100",
        "NIFTY 200": "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20200",
        "NIFTY 500": "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20500",
        "NIFTY MIDCAP 50": "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20MIDCAP%2050",
        "NIFTY MIDCAP 100": "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20MIDCAP%20100",
        "NIFTY MIDCAP 150": "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20MIDCAP%20150",
        "NIFTY SMALLCAP 50": "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20SMALLCAP%2050",
        "NIFTY SMALLCAP 100": "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20SMALLCAP%20100",
        "NIFTY SMALLCAP 250": "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20SMALLCAP%20250",
        "NIFTY BANK": "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20BANK",
        "NIFTY AUTO": "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20AUTO",
        "NIFTY FINANCIAL SERVICES": "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20FINANCIAL%20SERVICES",
        "NIFTY FMCG": "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20FMCG",
        "NIFTY IT": "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20IT",
        "NIFTY MEDIA": "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20MEDIA",
        "NIFTY METAL": "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20METAL",
        "NIFTY PHARMA": "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20PHARMA",
        "NIFTY PSU BANK": "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20PSU%20BANK",
        "NIFTY REALTY": "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20REALTY",
        "NIFTY HEALTHCARE": "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20HEALTHCARE%20INDEX",
    }

    def __init__(self):
        self.session = requests.Session()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Referer': 'https://www.nseindia.com/',
            'Connection': 'keep-alive',
        }

    def _get_cookies(self):
        """Get cookies from NSE homepage"""
        try:
            response = self.session.get(
                self.BASE_URL,
                headers=self.headers,
                timeout=10
            )
            return response.cookies
        except Exception as e:
            print(f"Error getting cookies: {e}")
            return None

    def fetch_index_stocks(self, index_name: str) -> List[str]:
        """
        Fetch stock symbols for a given index

        Args:
            index_name: Name of the index (e.g., "NIFTY 50")

        Returns:
            List of stock symbols
        """
        if index_name not in self.INDEX_URLS:
            print(f"Index {index_name} not found in available indices")
            return []

        try:
            # Fetch index data (cookies already set in fetch_all_indices)
            response = self.session.get(
                self.INDEX_URLS[index_name],
                headers=self.headers,
                timeout=15
            )

            if response.status_code == 200:
                data = response.json()
                stocks = []

                if 'data' in data:
                    for item in data['data']:
                        if 'symbol' in item and item['symbol'] != index_name:
                            stocks.append(item['symbol'])

                print(f"[OK] Fetched {len(stocks)} stocks from {index_name}")
                return stocks
            else:
                print(f"[FAIL] Failed to fetch {index_name}: Status {response.status_code}")
                return []

        except Exception as e:
            print(f"[ERROR] Error fetching {index_name}: {e}")
            return []

    def fetch_all_indices(self, indices: List[str] = None) -> Dict[str, List[str]]:
        """
        Fetch stocks for multiple indices in parallel

        Args:
            indices: List of index names to fetch. If None, fetches all available.

        Returns:
            Dictionary mapping index names to stock lists
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        if indices is None:
            indices = list(self.INDEX_URLS.keys())

        results = {}
        total = len(indices)

        # Get cookies once for all requests
        self._get_cookies()

        # Fetch all indices in parallel with 10 concurrent workers
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_index = {executor.submit(self.fetch_index_stocks, idx): idx for idx in indices}

            completed = 0
            for future in as_completed(future_to_index):
                index_name = future_to_index[future]
                completed += 1
                try:
                    stocks = future.result()
                    if stocks:
                        results[index_name] = stocks
                        print(f"[{completed}/{total}] Fetched {index_name}: {len(stocks)} stocks")
                    else:
                        print(f"[{completed}/{total}] Failed {index_name}")
                except Exception as e:
                    print(f"[{completed}/{total}] Error {index_name}: {e}")

        return results

    def generate_universe_file(self, output_file: str = "nifty_universe_updated.py"):
        """
        Generate updated universe file with live data from NSE

        Args:
            output_file: Path to output Python file
        """
        print("=" * 60)
        print("Fetching live stock data from NSE India...")
        print("=" * 60)

        # Fetch key indices
        key_indices = [
            "NIFTY 50",
            "NIFTY NEXT 50",
            "NIFTY 100",
            "NIFTY 200",
            "NIFTY 500",
            "NIFTY MIDCAP 50",
            "NIFTY MIDCAP 100",
            "NIFTY MIDCAP 150",
            "NIFTY SMALLCAP 50",
            "NIFTY SMALLCAP 100",
            "NIFTY SMALLCAP 250",
            "NIFTY BANK",
            "NIFTY AUTO",
            "NIFTY FINANCIAL SERVICES",
            "NIFTY FMCG",
            "NIFTY IT",
            "NIFTY MEDIA",
            "NIFTY METAL",
            "NIFTY PHARMA",
            "NIFTY PSU BANK",
            "NIFTY REALTY",
            "NIFTY HEALTHCARE"
        ]

        results = self.fetch_all_indices(key_indices)

        # Generate Python file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('"""\n')
            f.write('NIFTY Universe Data - Auto-generated from NSE India\n')
            f.write(f'Last Updated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
            f.write('Source: NSE Official API\n')
            f.write('"""\n\n')

            # Write each index
            for index_name, stocks in results.items():
                var_name = index_name.replace(" ", "_").replace("-", "_").upper()
                f.write(f"# {index_name} ({len(stocks)} stocks)\n")
                f.write(f"{var_name} = [\n")

                # Write stocks in formatted rows
                for i in range(0, len(stocks), 5):
                    batch = stocks[i:i+5]
                    formatted = ', '.join(f'"{s}"' for s in batch)
                    f.write(f"    {formatted},\n")

                f.write("]\n\n")

            # Add summary
            f.write("\n# Summary\n")
            f.write("UNIVERSE_STATS = {\n")
            for index_name, stocks in results.items():
                f.write(f'    "{index_name}": {len(stocks)},\n')
            f.write("}\n")

        print("=" * 60)
        print(f"[SUCCESS] Generated {output_file}")
        print("=" * 60)
        print("\nStock counts:")
        for index_name, stocks in results.items():
            print(f"  {index_name}: {len(stocks)} stocks")
        print("=" * 60)

        return results


def update_universe_file_inplace(universe_file: str = "nifty_universe.py"):
    """
    Update the universe file in-place, preserving UNIVERSES dict and helper functions

    Args:
        universe_file: Path to the nifty_universe.py file to update

    Returns:
        Dictionary of fetched indices and their stocks
    """
    import re
    import shutil

    # Backup the original file
    backup_file = universe_file.replace(".py", "_backup.py")
    shutil.copy(universe_file, backup_file)
    print(f"Created backup: {backup_file}")

    # Fetch latest data from NSE
    fetcher = NSEFetcher()
    key_indices = [
        "NIFTY 50", "NIFTY NEXT 50", "NIFTY 100", "NIFTY 200", "NIFTY 500",
        "NIFTY MIDCAP 50", "NIFTY MIDCAP 100", "NIFTY MIDCAP 150",
        "NIFTY SMALLCAP 50", "NIFTY SMALLCAP 100", "NIFTY SMALLCAP 250",
        "NIFTY BANK", "NIFTY AUTO", "NIFTY FINANCIAL SERVICES",
        "NIFTY FMCG", "NIFTY IT", "NIFTY MEDIA", "NIFTY METAL",
        "NIFTY PHARMA", "NIFTY PSU BANK", "NIFTY REALTY", "NIFTY HEALTHCARE"
    ]

    results = fetcher.fetch_all_indices(key_indices)

    # Read the original file
    with open(universe_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Update each index definition
    for index_name, stocks in results.items():
        var_name = index_name.replace(" ", "_").replace("-", "_").upper()

        # Create new stock list
        new_stocks = []
        for i in range(0, len(stocks), 5):
            batch = stocks[i:i+5]
            formatted = ', '.join(f'"{s}"' for s in batch)
            new_stocks.append(f"    {formatted},")

        new_definition = f"{var_name} = [\n" + "\n".join(new_stocks) + "\n]"

        # Replace the old definition with new one
        pattern = rf"{var_name}\s*=\s*\[[\s\S]*?\]"
        content = re.sub(pattern, new_definition, content)

    # Update the header comment with timestamp
    header_pattern = r'("""\s*.*?Last Updated:).*?(\n)'
    content = re.sub(
        header_pattern,
        rf'\1 {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}\2',
        content,
        flags=re.DOTALL
    )

    # Write back the updated content
    with open(universe_file, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"[SUCCESS] Updated {universe_file} with latest NSE data")
    print(f"Updated {len(results)} indices")

    return results


def refresh_universe_stocks():
    """Convenience function to refresh universe stocks"""
    return update_universe_file_inplace()


if __name__ == "__main__":
    # Test the fetcher
    fetcher = NSEFetcher()
    fetcher.generate_universe_file()
