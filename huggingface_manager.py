"""
Hugging Face Dataset Manager

Manages uploading and downloading OHLC stock data to/from Hugging Face Datasets.
Supports incremental updates - only fetches missing dates.
"""

import os
import pandas as pd
from datetime import date, datetime
from pathlib import Path
from typing import Optional, Callable
from dotenv import load_dotenv

load_dotenv()

# Get HF configuration from environment
HF_TOKEN = os.getenv("HF_TOKEN")
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO")


class HuggingFaceManager:
    """Manages stock data storage on Hugging Face Datasets."""
    
    def __init__(self, repo_id: Optional[str] = None, token: Optional[str] = None):
        """Initialize the manager.
        
        Args:
            repo_id: HuggingFace dataset repo ID (e.g., 'username/nse-dhan-ohlc')
            token: HuggingFace write access token
        """
        self.repo_id = repo_id or HF_DATASET_REPO
        self.token = token or HF_TOKEN
        
        if not self.repo_id:
            raise ValueError("HF_DATASET_REPO not configured. Add to .env file.")
        if not self.token:
            raise ValueError("HF_TOKEN not configured. Add to .env file.")
        
        # Import here to avoid slowing down app startup if not used
        from huggingface_hub import HfApi, hf_hub_download, upload_file
        self._api = HfApi(token=self.token)
        
    def _get_symbol_path(self, symbol: str) -> str:
        """Get the path within the HF repo for a symbol's data file."""
        return f"data/{symbol}.parquet"
    
    def download_symbol_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Download data for a symbol from Hugging Face.
        
        Args:
            symbol: Stock symbol (e.g., 'RELIANCE')
            
        Returns:
            DataFrame with OHLC data or None if not found
        """
        from huggingface_hub import hf_hub_download, HfFileSystemResolvedPath
        from huggingface_hub.utils import EntryNotFoundError
        
        try:
            file_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=self._get_symbol_path(symbol),
                repo_type="dataset",
                token=self.token
            )
            
            df = pd.read_parquet(file_path)
            
            # Ensure Date column is datetime
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
            
            return df
            
        except EntryNotFoundError:
            return None
        except Exception as e:
            print(f"Error downloading {symbol} from HF: {e}")
            return None
    
    def upload_symbol_data(self, symbol: str, df: pd.DataFrame) -> bool:
        """Upload data for a symbol to Hugging Face.
        
        Args:
            symbol: Stock symbol (e.g., 'RELIANCE')
            df: DataFrame with columns: Date, Open, High, Low, Close, Volume
            
        Returns:
            True if upload successful
        """
        from huggingface_hub import upload_file
        import tempfile
        
        try:
            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
                df.to_parquet(tmp.name, index=False)
                tmp_path = tmp.name
            
            # Upload to HF
            upload_file(
                path_or_fileobj=tmp_path,
                path_in_repo=self._get_symbol_path(symbol),
                repo_id=self.repo_id,
                repo_type="dataset",
                token=self.token
            )
            
            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)
            
            return True
            
        except Exception as e:
            print(f"Error uploading {symbol} to HF: {e}")
            return False
    
    def get_existing_date_range(self, symbol: str) -> tuple[Optional[date], Optional[date]]:
        """Get the date range of existing data for a symbol.
        
        Returns:
            (min_date, max_date) or (None, None) if no data exists
        """
        df = self.download_symbol_data(symbol)
        
        if df is None or df.empty:
            return None, None
        
        min_date = df['Date'].min().date()
        max_date = df['Date'].max().date()
        
        return min_date, max_date
    
    def get_missing_dates(
        self, 
        symbol: str, 
        from_date: date, 
        to_date: date
    ) -> list[tuple[date, date]]:
        """Get date ranges that need to be fetched for a symbol.
        
        Returns list of (start, end) tuples representing gaps in data.
        """
        existing_min, existing_max = self.get_existing_date_range(symbol)
        
        if existing_min is None:
            # No existing data, need everything
            return [(from_date, to_date)]
        
        gaps = []
        
        # Gap before existing data
        if from_date < existing_min:
            gaps.append((from_date, existing_min))
        
        # Gap after existing data
        if to_date > existing_max:
            gaps.append((existing_max, to_date))
        
        return gaps
    
    def sync_symbol(
        self,
        symbol: str,
        new_data: pd.DataFrame
    ) -> bool:
        """Sync data for a symbol - merge with existing and upload.
        
        Args:
            symbol: Stock symbol
            new_data: New data to add
            
        Returns:
            True if successful
        """
        if new_data is None or new_data.empty:
            return False
        
        # Get existing data
        existing = self.download_symbol_data(symbol)
        
        if existing is not None and not existing.empty:
            # Merge: existing + new, remove duplicates based on Date
            combined = pd.concat([existing, new_data], ignore_index=True)
            combined = combined.drop_duplicates(subset=['Date'], keep='last')
            combined = combined.sort_values('Date').reset_index(drop=True)
        else:
            combined = new_data.sort_values('Date').reset_index(drop=True)
        
        # Upload merged data
        return self.upload_symbol_data(symbol, combined)
    
    def sync_all_symbols(
        self,
        symbols: list[str],
        from_date: date,
        to_date: date,
        progress_callback: Optional[Callable[[int, int, str, str], None]] = None
    ) -> int:
        """Sync data for all symbols using BATCH UPLOAD (1 commit).
        
        This method:
        1. Fetches all data from Dhan API
        2. Saves to local temp folder
        3. Uploads entire folder in ONE commit (avoids rate limits)
        
        Args:
            symbols: List of stock symbols
            from_date: Start date for data
            to_date: End date for data
            progress_callback: Callback(current, total, symbol, status)
            
        Returns:
            Number of symbols successfully fetched
        """
        from dhan_data_fetcher import fetch_historical_data
        from huggingface_hub import upload_folder
        import tempfile
        import shutil
        import time
        
        # Create temp directory for batch collection
        temp_dir = Path(tempfile.mkdtemp(prefix="dhan_batch_"))
        data_dir = temp_dir / "data"
        data_dir.mkdir(exist_ok=True)
        
        success_count = 0
        
        try:
            # Phase 1: Fetch all data from Dhan and save locally
            if progress_callback:
                progress_callback(0, len(symbols), "Starting", "Phase 1: Fetching from Dhan API...")
            
            for i, symbol in enumerate(symbols):
                try:
                    if progress_callback:
                        progress_callback(i + 1, len(symbols), symbol, "Fetching from Dhan...")
                    
                    # Fetch data from Dhan
                    new_data = fetch_historical_data(symbol, from_date, to_date)
                    
                    if new_data is not None and not new_data.empty:
                        # Try to merge with existing HF data
                        try:
                            existing = self.download_symbol_data(symbol)
                            if existing is not None and not existing.empty:
                                combined = pd.concat([existing, new_data], ignore_index=True)
                                combined = combined.drop_duplicates(subset=['Date'], keep='last')
                                combined = combined.sort_values('Date').reset_index(drop=True)
                            else:
                                combined = new_data.sort_values('Date').reset_index(drop=True)
                        except:
                            combined = new_data.sort_values('Date').reset_index(drop=True)
                        
                        # Save to local temp folder
                        file_path = data_dir / f"{symbol}.parquet"
                        combined.to_parquet(file_path, index=False)
                        success_count += 1
                        
                        if progress_callback:
                            progress_callback(i + 1, len(symbols), symbol, f"Saved ✓ ({len(combined)} days)")
                    else:
                        if progress_callback:
                            progress_callback(i + 1, len(symbols), symbol, "No data from Dhan")
                    
                    # Small delay to avoid rate limiting
                    time.sleep(0.3)
                    
                except Exception as e:
                    print(f"Error fetching {symbol}: {e}")
                    if progress_callback:
                        progress_callback(i + 1, len(symbols), symbol, f"Error: {str(e)[:30]}")
            
            # Phase 2: Upload entire folder in ONE commit
            if success_count > 0:
                if progress_callback:
                    progress_callback(len(symbols), len(symbols), "Uploading", 
                                    f"Phase 2: Uploading {success_count} files in 1 commit...")
                
                try:
                    upload_folder(
                        folder_path=str(temp_dir),
                        repo_id=self.repo_id,
                        repo_type="dataset",
                        token=self.token,
                        commit_message=f"Batch upload: {success_count} symbols ({from_date} to {to_date})"
                    )
                    
                    if progress_callback:
                        progress_callback(len(symbols), len(symbols), "Complete", 
                                        f"✅ Uploaded {success_count} symbols in 1 commit!")
                except Exception as e:
                    print(f"Error uploading folder: {e}")
                    if progress_callback:
                        progress_callback(len(symbols), len(symbols), "Upload Error", str(e)[:50])
                    return 0
            
        finally:
            # Cleanup temp directory
            try:
                shutil.rmtree(temp_dir)
            except:
                pass
        
        return success_count
    
    def list_available_symbols(self) -> list[str]:
        """List all symbols available in the HF repository."""
        try:
            from huggingface_hub import list_repo_files
            
            files = list_repo_files(
                repo_id=self.repo_id,
                repo_type="dataset",
                token=self.token
            )
            
            # Extract symbol names from file paths
            symbols = []
            for f in files:
                if f.startswith("data/") and f.endswith(".parquet"):
                    symbol = f[5:-8]  # Remove "data/" and ".parquet"
                    symbols.append(symbol)
            
            return sorted(symbols)
            
        except Exception as e:
            print(f"Error listing HF files: {e}")
            return []


def is_hf_configured() -> bool:
    """Check if Hugging Face is properly configured."""
    return bool(HF_TOKEN) and bool(HF_DATASET_REPO)


# Test function
if __name__ == "__main__":
    if not is_hf_configured():
        print("HuggingFace not configured. Set HF_TOKEN and HF_DATASET_REPO in .env")
    else:
        print(f"HF Repository: {HF_DATASET_REPO}")
        
        hf = HuggingFaceManager()
        symbols = hf.list_available_symbols()
        print(f"Available symbols: {len(symbols)}")
        
        if symbols:
            print(f"First 10: {symbols[:10]}")

