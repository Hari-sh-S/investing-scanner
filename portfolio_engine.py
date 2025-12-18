import pandas as pd
import numpy as np
import yfinance as yf
from indicators import IndicatorLibrary
from scoring import ScoreParser
from pathlib import Path
from datetime import timedelta

class DataCache:
    """Efficient Parquet-based cache for stock data."""

    def __init__(self, cache_dir="data_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def _get_cache_path(self, ticker):
        """Generate cache file path."""
        filename = f"{ticker}.parquet"
        return self.cache_dir / filename

    def get(self, ticker):
        """Retrieve cached data if available."""
        cache_path = self._get_cache_path(ticker)

        if not cache_path.exists():
            return None

        try:
            df = pd.read_parquet(cache_path)
            # Fix MultiIndex columns from old cache format
            if isinstance(df.columns, pd.MultiIndex):
                print(f"[CACHE FIX] {ticker}: Converting MultiIndex columns")
                df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
            # Also check for tuple-like strings in column names
            elif len(df.columns) > 0 and isinstance(df.columns[0], tuple):
                print(f"[CACHE FIX] {ticker}: Converting tuple columns")
                df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
            return df
        except Exception as e:
            print(f"Cache read error for {ticker}: {e}")
            return None

    def set(self, ticker, data):
        """Store data in cache as Parquet."""
        cache_path = self._get_cache_path(ticker)
        try:
            data.to_parquet(cache_path, compression='snappy')
        except Exception as e:
            print(f"Cache save error for {ticker}: {e}")

    def exists(self, ticker):
        """Check if ticker data exists in cache."""
        return self._get_cache_path(ticker).exists()

    def get_cache_info(self):
        """Get cache statistics."""
        files = list(self.cache_dir.glob("*.parquet"))
        total_size = sum(f.stat().st_size for f in files)
        return {
            'total_files': len(files),
            'total_size_mb': total_size / (1024 * 1024),
            'tickers': [f.stem for f in files]
        }

    def clear(self):
        """Clear all cached data."""
        for file in self.cache_dir.glob("*.parquet"):
            file.unlink()

    def delete_ticker(self, ticker):
        """Delete cache for specific ticker."""
        cache_path = self._get_cache_path(ticker)
        if cache_path.exists():
            cache_path.unlink()


class PortfolioEngine:
    def __init__(self, universe, start_date, end_date, initial_capital=100000, use_cache=True):
        self.universe = universe
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.data = {}
        self.portfolio_value = []
        self.trades = []
        self.holdings_history = []
        self.parser = ScoreParser()
        self.cache = DataCache() if use_cache else None
        self.regime_index_data = None
        # EQUITY regime filter tracking
        self.equity_regime_analysis = None  # Stores theoretical vs actual comparison data
        self.regime_trigger_events = []  # List of {date, type: 'trigger'/'recovery', drawdown, peak}

    @staticmethod
    def _get_scalar(value):
        """Safely extract scalar from potential Series or DataFrame."""
        if isinstance(value, (pd.Series, pd.DataFrame)):
            return value.iloc[0] if len(value) > 0 else 0
        return value

    def download_and_cache_universe(self, universe_tickers, progress_callback=None, stop_flag=None):
        """Fast batch download with fallback to single ticker download."""
        import time

        # Filter already cached
        tickers_to_download = []
        for ticker in universe_tickers:
            if self.cache and self.cache.exists(ticker):
                continue
            tickers_to_download.append(ticker)

        if not tickers_to_download:
            return len(universe_tickers)

        success_count = 0
        start_time = time.time()
        
        # Convert to .NS format for yfinance
        tickers_ns = [t if t.endswith(('.NS', '.BO')) else f"{t}.NS" for t in tickers_to_download]
        ticker_map = {ns: orig for ns, orig in zip(tickers_ns, tickers_to_download)}
        
        # Download in chunks of 200 (optimal batch size)
        CHUNK_SIZE = 200
        chunks = [tickers_ns[i:i + CHUNK_SIZE] for i in range(0, len(tickers_ns), CHUNK_SIZE)]
        
        completed = 0
        for chunk_idx, chunk in enumerate(chunks):
            if stop_flag and stop_flag[0]:
                print(f"Stopped at chunk {chunk_idx}/{len(chunks)}")
                break
            
            # Batch download with threads
            batch_result = self._download_batch(chunk, ticker_map)
            
            # Count successes - no slow fallback
            for ticker_ns in chunk:
                if batch_result.get(ticker_ns, False):
                    success_count += 1
                completed += 1
            
            # Update progress after each chunk
            if progress_callback:
                elapsed = time.time() - start_time
                avg = elapsed / completed if completed > 0 else 0
                remaining_time = avg * (len(tickers_to_download) - completed)
                try:
                    progress_callback(completed, len(tickers_to_download), f"Batch {chunk_idx + 1}/{len(chunks)}", remaining_time)
                except TypeError:
                    progress_callback(completed, len(tickers_to_download), f"Batch {chunk_idx + 1}")

        if progress_callback:
            try:
                progress_callback(len(tickers_to_download), len(tickers_to_download), "Done", 0)
            except TypeError:
                progress_callback(len(tickers_to_download), len(tickers_to_download), "Done")

        elapsed = time.time() - start_time
        print(f"Downloaded {success_count}/{len(tickers_to_download)} stocks in {elapsed:.1f}s ({elapsed/max(len(tickers_to_download),1):.2f}s/stock)")
        return len(universe_tickers)
    
    def _download_batch(self, tickers, ticker_map):
        """Batch download multiple tickers at once using threads."""
        try:
            data = yf.download(
                tickers,
                period="max",
                interval="1d",
                group_by="ticker",
                threads=True,
                progress=False,
            )
        except Exception as e:
            print(f"Batch download failed: {e}")
            return {}
        
        saved = {}
        
        if isinstance(data.columns, pd.MultiIndex):
            for ticker_ns in tickers:
                try:
                    ticker = ticker_map[ticker_ns]
                    df = data[ticker_ns].dropna(how="all")
                    if not df.empty and len(df) >= 100:
                        df = df.reset_index()
                        # Save without indicators for speed - indicators calculated on fetch
                        expected_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                        df = df[[col for col in expected_cols if col in df.columns]]
                        if self.cache:
                            self.cache.set(ticker, df)
                        saved[ticker_ns] = True
                    else:
                        saved[ticker_ns] = False
                except Exception:
                    saved[ticker_ns] = False
        else:
            # Single ticker result (different format)
            if len(tickers) == 1 and not data.empty:
                ticker_ns = tickers[0]
                ticker = ticker_map[ticker_ns]
                df = data.dropna(how="all").reset_index()
                if len(df) >= 100:
                    expected_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                    df = df[[col for col in expected_cols if col in df.columns]]
                    if self.cache:
                        self.cache.set(ticker, df)
                    saved[ticker_ns] = True
        
        return saved
    
    def _download_single(self, ticker_ns, ticker, retries=2, backoff=1):
        """Download single ticker with minimal retry."""
        import time
        for attempt in range(1, retries + 1):
            try:
                df = yf.download(ticker_ns, period="max", interval="1d", progress=False)
                if not df.empty and len(df) >= 100:
                    df = df.reset_index()
                    expected_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                    df = df[[col for col in expected_cols if col in df.columns]]
                    if self.cache:
                        self.cache.set(ticker, df)
                    return True
            except Exception:
                pass
            time.sleep(backoff)
        return False
    
    def _process_and_cache_df(self, ticker, df):
        """Process dataframe and save to cache with indicators."""
        try:
            expected_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            df = df[[col for col in expected_cols if col in df.columns]]
            
            # Calculate indicators
            try:
                df_with_date_index = df.set_index('Date')
                df_with_date_index = IndicatorLibrary.add_momentum_volatility_metrics(df_with_date_index)
                df_with_date_index = IndicatorLibrary.add_regime_filters(df_with_date_index)
                df = df_with_date_index.reset_index()
            except:
                pass  # Use raw data if indicators fail
            
            if self.cache:
                self.cache.set(ticker, df)
        except Exception as e:
            print(f"Error caching {ticker}: {e}")

    def fetch_data(self, progress_callback=None):
        """Fetch data from cache. Indicators should already be pre-calculated during download."""
        print(f"Loading data for {len(self.universe)} stocks...")
        tickers_to_download = []

        def clean_dataframe(df, ticker_name="unknown"):
            """Remove duplicates, detect anomalies, and ensure clean data."""
            try:
                if df is None or df.empty:
                    return df
                # Remove duplicate indices (dates)
                if hasattr(df.index, 'duplicated') and df.index.duplicated().any():
                    df = df[~df.index.duplicated(keep='last')]
                # Sort by date
                df = df.sort_index()
                
                # DATA QUALITY: Detect extreme single-day price changes (>50%)
                # Just warn, don't fix
                if 'Close' in df.columns and len(df) > 3:
                    close = df['Close']
                    pct_change = close.pct_change().abs()
                    for i in range(1, len(pct_change) - 1):
                        if pct_change.iloc[i] > 0.5:  # >50% change
                            next_change = pct_change.iloc[i + 1] if i + 1 < len(pct_change) else 0
                            if next_change > 0.3:  # If it reverses next day
                                print(f"⚠️ DATA ANOMALY: {ticker_name} on {df.index[i].date()} - {pct_change.iloc[i]*100:.1f}% drop, then {next_change*100:.1f}% recovery")
                
                return df
            except Exception:
                return df  # Return unchanged if cleaning fails

        # First, try to load from cache
        for i, ticker in enumerate(self.universe):
            if progress_callback:
                progress_callback(i + 1, len(self.universe), ticker)

            if self.cache:
                cached_data = self.cache.get(ticker)
                if cached_data is not None:
                    try:
                        # Fix index - Date should be the index
                        if 'Date' in cached_data.columns:
                            cached_data['Date'] = pd.to_datetime(cached_data['Date'])
                            cached_data.set_index('Date', inplace=True)

                        # Ensure index is datetime
                        if not isinstance(cached_data.index, pd.DatetimeIndex):
                            cached_data.index = pd.to_datetime(cached_data.index)

                        # Clean data - remove duplicate indices and fix anomalies
                        cached_data = clean_dataframe(cached_data, ticker)

                        # Include 300 days BEFORE start_date for indicator lookback
                        # 6-month performance needs ~130 days, 1-year needs ~260 days
                        extended_start = pd.Timestamp(self.start_date) - pd.Timedelta(days=300)
                        mask = (cached_data.index >= extended_start) & \
                               (cached_data.index <= pd.Timestamp(self.end_date))
                        df_filtered = cached_data[mask].copy()
                        
                        # Flatten MultiIndex columns if present
                        if isinstance(df_filtered.columns, pd.MultiIndex):
                            df_filtered.columns = [col[0] if isinstance(col, tuple) else col for col in df_filtered.columns]

                        if not df_filtered.empty and len(df_filtered) >= 100:
                            self.data[ticker] = df_filtered
                            continue
                    except Exception as e:
                        print(f"Error loading {ticker}: {e}")

            # If not in cache or insufficient data, mark for download
            tickers_to_download.append(ticker)

        # Download missing tickers (with indicators calculated automatically)
        if tickers_to_download:
            print(f"Downloading {len(tickers_to_download)} missing stocks...")
            self.download_and_cache_universe(tickers_to_download, progress_callback)

            # Retry loading after download
            for ticker in tickers_to_download:
                if self.cache:
                    cached_data = self.cache.get(ticker)
                    if cached_data is not None:
                        try:
                            # Fix index - Date should be the index
                            if 'Date' in cached_data.columns:
                                cached_data['Date'] = pd.to_datetime(cached_data['Date'])
                                cached_data.set_index('Date', inplace=True)

                            # Ensure index is datetime
                            if not isinstance(cached_data.index, pd.DatetimeIndex):
                                cached_data.index = pd.to_datetime(cached_data.index)

                            # Clean data
                            cached_data = clean_dataframe(cached_data, ticker)

                            # Include 300 days BEFORE start_date for indicator lookback
                            extended_start = pd.Timestamp(self.start_date) - pd.Timedelta(days=300)
                            mask = (cached_data.index >= extended_start) & \
                                   (cached_data.index <= pd.Timestamp(self.end_date))
                            df_filtered = cached_data[mask].copy()
                            
                            # Flatten MultiIndex columns if present
                            if isinstance(df_filtered.columns, pd.MultiIndex):
                                df_filtered.columns = [col[0] if isinstance(col, tuple) else col for col in df_filtered.columns]

                            if not df_filtered.empty:
                                self.data[ticker] = df_filtered
                        except Exception as e:
                            print(f"Error loading {ticker} after download: {e}")

        print(f"Successfully loaded {len(self.data)} stocks")
        return len(self.data) > 0
    
    def calculate_indicators_for_formula(self, formula, regime_config=None):
        """Calculate only the indicators needed for the formula and regime filter."""
        # Determine which indicator types are needed
        needs_momentum = any(x in formula.upper() for x in ['PERFORMANCE', 'SHARPE', 'SORTINO', 'CALMAR', 'VOLATILITY', 'DRAWDOWN'])
        needs_regime = regime_config is not None and regime_config.get('type') != 'EQUITY'
        
        if not needs_momentum and not needs_regime:
            return  # No indicators needed
        
        print(f"Calculating indicators (momentum={needs_momentum}, regime={needs_regime})...")
        
        for ticker in self.data:
            try:
                df = self.data[ticker]
                
                # Flatten columns first if needed (fix for cloud cache issues)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
                elif len(df.columns) > 0 and isinstance(df.columns[0], tuple):
                    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
                
                # Only calculate if not already calculated
                if needs_momentum and '6 Month Performance' not in df.columns:
                    df = IndicatorLibrary.add_momentum_volatility_metrics(df)
                if needs_regime and 'EMA_200' not in df.columns:
                    df = IndicatorLibrary.add_regime_filters(df)
                
                # Flatten again after indicators (some libraries create MultiIndex)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
                
                self.data[ticker] = df
            except Exception as e:
                print(f"Error calculating indicators for {ticker}: {e}")

    def _get_rebalance_dates(self, all_dates, rebal_config):
        """Generate rebalance dates based on config. Ensures every period has a rebalance."""
        freq = rebal_config['frequency']
        all_dates_set = set(all_dates)
        
        if freq == 'Weekly':
            # Get day of week (0=Monday, 4=Friday)
            day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4}
            target_day = day_map[rebal_config['day']]
            
            rebalance_dates = [d for d in all_dates if d.weekday() == target_day]
        else:  # Monthly
            target_date = rebal_config['date']
            alt_option = rebal_config.get('alt_day', 'Next Day')
            
            rebalance_dates = []
            
            # Group dates by (year, month)
            month_groups = {}
            for date in all_dates:
                key = (date.year, date.month)
                if key not in month_groups:
                    month_groups[key] = []
                month_groups[key].append(date)
            
            # For each month, find the best rebalance date
            for (year, month), month_dates in month_groups.items():
                month_dates_sorted = sorted(month_dates)
                rebalance_date = None
                
                # First, try to find exact target date
                for d in month_dates_sorted:
                    if d.day == target_date:
                        rebalance_date = d
                        break
                
                # If not found, use alternative
                if rebalance_date is None:
                    if alt_option == 'Previous Day':
                        # Find the closest trading day BEFORE target date
                        for d in reversed(month_dates_sorted):
                            if d.day < target_date:
                                rebalance_date = d
                                break
                        # If no day before, take the first available day
                        if rebalance_date is None and month_dates_sorted:
                            rebalance_date = month_dates_sorted[0]
                    else:  # Next Day
                        # Find the closest trading day AFTER target date
                        for d in month_dates_sorted:
                            if d.day > target_date:
                                rebalance_date = d
                                break
                        # If no day after, take the last available day
                        if rebalance_date is None and month_dates_sorted:
                            rebalance_date = month_dates_sorted[-1]
                
                if rebalance_date:
                    rebalance_dates.append(rebalance_date)
        
        print(f"Generated {len(rebalance_dates)} rebalance dates from {len(all_dates)} trading days")
        return sorted(rebalance_dates)

    def _check_regime_filter(self, date, regime_config, current_equity=0, peak_equity=0):
        """Check if regime filter is triggered.
        
        For EQUITY type: checks drawdown from peak equity
        For other types: checks technical indicators on index
        
        Returns: (triggered: bool, action: str, drawdown_pct: float)
        """
        if not regime_config:
            return False, 'none', 0.0  # No filter active
        
        regime_type = regime_config['type']
        
        if regime_type == 'EQUITY':
            # Check drawdown from peak equity
            sl_pct = regime_config['value']
            if peak_equity > 0:
                drawdown_pct = ((peak_equity - current_equity) / peak_equity) * 100
            else:
                drawdown_pct = 0.0
            
            if drawdown_pct > sl_pct:
                return True, regime_config['action'], drawdown_pct
            return False, 'none', drawdown_pct
        
        # For EMA, MACD, SUPERTREND - need index data
        if self.regime_index_data is None or self.regime_index_data.empty:
            return False, 'none', 0.0
        
        # Use nearest available date if exact date not found (handles holidays)
        if date not in self.regime_index_data.index:
            nearest = self.regime_index_data.index.asof(date)
            if pd.isna(nearest):
                return False, 'none', 0.0
            row = self.regime_index_data.loc[nearest]
        else:
            row = self.regime_index_data.loc[date]
        
        # Helper to extract scalar from potential Series
        def get_scalar(val):
            if hasattr(val, 'iloc'):
                return float(val.iloc[0])
            return float(val) if val is not None else 0.0
        
        if regime_type == 'EMA':
            ema_period = regime_config['value']
            ema_col = f'EMA_{ema_period}'
            close_price = get_scalar(row.get('Close', 0))
            ema_value = get_scalar(row.get(ema_col, 0))
            
            triggered = ema_col in row.index and ema_value > 0 and close_price < ema_value
            print(f"REGIME CHECK [{date}]: Close={close_price:.2f}, {ema_col}={ema_value:.2f}, Triggered={triggered}")
            
            # Triggered when index closes BELOW EMA
            if triggered:
                return True, regime_config['action'], 0.0
        
        elif regime_type == 'MACD':
            macd_val = get_scalar(row.get('MACD', 0))
            signal_val = get_scalar(row.get('MACD_Signal', 0))
            triggered = macd_val < signal_val
            print(f"REGIME CHECK [{date}]: MACD={macd_val:.2f}, Signal={signal_val:.2f}, Triggered={triggered}")
            if triggered:
                return True, regime_config['action'], 0.0
        
        elif regime_type == 'SUPERTREND':
            # Use Supertrend_Direction column which has 'BUY' or 'SELL'
            st_direction = row.get('Supertrend_Direction', 'BUY')
            if hasattr(st_direction, 'iloc'):
                st_direction = st_direction.iloc[0]
            triggered = st_direction == 'SELL'
            print(f"REGIME CHECK [{date}]: SuperTrend Direction={st_direction}, Triggered={triggered}")
            if triggered:
                return True, regime_config['action'], 0.0
        
        return False, 'none', 0.0

    def run_rebalance_strategy(self, scoring_formula, num_stocks, exit_rank, 
                              rebal_config, regime_config=None, uncorrelated_config=None, reinvest_profits=True):
        """
        Advanced backtesting engine with all Sigma Scanner features.
        """
        if not self.data:
            print("No data available")
            return
        
        # Validate formula
        is_valid, msg = self.parser.validate_formula(scoring_formula)
        if not is_valid:
            print(f"Invalid formula: {msg}")
            return
        
        # Calculate indicators on-demand based on formula
        self.calculate_indicators_for_formula(scoring_formula, regime_config)
        
        # Load regime filter index data if needed
        if regime_config and regime_config['type'] != 'EQUITY':
            regime_index = regime_config['index']
            # Map universe names to Yahoo Finance tickers
            index_map = {
                # Broad Market Indices
                'NIFTY 50': '^NSEI',
                'NIFTY NEXT 50': '^NSMIDCP',
                'NIFTY 100': '^CNX100',
                'NIFTY 200': '^CNX200',
                'NIFTY 500': '^CRSLDX',
                'NIFTY MIDCAP 50': '^NSEMDCP50',
                'NIFTY MIDCAP 100': '^CNXMC',
                'NIFTY SMALLCAP 50': '^NIFTYSMCP50',
                'NIFTY SMALLCAP 100': '^CNXSC',
                'NIFTY LARGEMIDCAP 250': '^CNXLM250',
                'NIFTY MIDSMALLCAP 400': '^CNXMSC400',
                # Sectoral Indices
                'NIFTY BANK': '^NSEBANK',
                'NIFTY FINANCIAL SERVICES': '^CNXFINANCE',
                'NIFTY IT': '^CNXIT',
                'NIFTY PHARMA': '^CNXPHARMA',
                'NIFTY AUTO': '^CNXAUTO',
                'NIFTY FMCG': '^CNXFMCG',
                'NIFTY METAL': '^CNXMETAL',
                'NIFTY REALTY': '^CNXREALTY',
                'NIFTY ENERGY': '^CNXENERGY',
                'NIFTY CONSUMPTION': '^CNXCONSUM',
                'NIFTY MEDIA': '^CNXMEDIA',
                'NIFTY INFRASTRUCTURE': '^CNXINFRA',
                # Thematic
                'NIFTY PSU': '^CNXPSE',
                'NIFTY MNC': '^CNXMNC'
            }
            index_ticker = index_map.get(regime_index, '^NSEI')
            
            try:
                # Download EXTRA historical data (300 days before start_date) 
                # so EMA 200 can be properly calculated from day 1 of backtest
                extended_start = pd.Timestamp(self.start_date) - timedelta(days=400)  # ~300 trading days
                regime_data = yf.download(index_ticker, start=extended_start, end=self.end_date, progress=False)
                if not regime_data.empty:
                    print(f"Downloaded {len(regime_data)} days of regime index data (with 400-day pre-buffer for EMA)")
                    regime_data = IndicatorLibrary.add_regime_filters(regime_data)
                    self.regime_index_data = regime_data
                    # Debug: show first few EMA values
                    if 'EMA_200' in regime_data.columns:
                        print(f"EMA_200 range: {regime_data['EMA_200'].min():.2f} - {regime_data['EMA_200'].max():.2f}")
            except Exception as e:
                print(f"Could not load regime index data: {e}")
        
        # Get common date range
        all_dates = sorted(list(set().union(*[df.index for df in self.data.values()])))
        
        # Get rebalance dates
        rebalance_dates = self._get_rebalance_dates(all_dates, rebal_config)
        
        # Initialize portfolio
        cash = self.initial_capital
        holdings = {}  # {ticker: shares}
        portfolio_history = []
        regime_active = False
        regime_cash_reserve = 0
        last_known_prices = {}  # Track last known prices for holdings (for data gaps)
        
        # EQUITY regime filter tracking
        peak_equity = self.initial_capital  # Track highest equity reached
        equity_regime_active = False  # True when drawdown exceeds threshold, waiting for recovery
        theoretical_history = []  # Track what would happen without EQUITY filter
        theoretical_holdings = {}  # Separate holdings for theoretical curve
        theoretical_cash = self.initial_capital
        is_equity_regime = regime_config and regime_config['type'] == 'EQUITY'
        equity_sl_pct = regime_config['value'] if is_equity_regime else 0
        self.regime_trigger_events = []
        
        for date in all_dates:
            is_rebalance = date in rebalance_dates
            
            # Calculate current equity value FIRST (before any trades)
            current_holdings_value = 0.0
            for ticker, shares in holdings.items():
                if ticker in self.data:
                    if date in self.data[ticker].index:
                        cp = self._get_scalar(self.data[ticker].loc[date, 'Close'])
                        last_known_prices[ticker] = cp
                    elif ticker in last_known_prices:
                        cp = last_known_prices[ticker]
                    else:
                        continue
                    current_holdings_value += shares * cp
            current_equity = cash + current_holdings_value
            
            # Track theoretical equity (without EQUITY regime filter)
            if is_equity_regime:
                theoretical_holdings_value = 0.0
                for ticker, shares in theoretical_holdings.items():
                    if ticker in self.data:
                        if date in self.data[ticker].index:
                            cp = self._get_scalar(self.data[ticker].loc[date, 'Close'])
                        elif ticker in last_known_prices:
                            cp = last_known_prices[ticker]
                        else:
                            continue
                        theoretical_holdings_value += shares * cp
                theoretical_equity = theoretical_cash + theoretical_holdings_value
                theoretical_history.append({
                    'Date': date,
                    'Theoretical_Equity': theoretical_equity,
                    'Theoretical_Holdings': theoretical_holdings_value
                })
            
            # Update peak equity (only when not in equity regime active mode)
            if not equity_regime_active and current_equity > peak_equity:
                peak_equity = current_equity
            
            # EQUITY REGIME: Check for mid-day drawdown breach (on any day, not just rebalance)
            if is_equity_regime and not equity_regime_active and holdings:
                drawdown_pct = ((peak_equity - current_equity) / peak_equity) * 100 if peak_equity > 0 else 0
                
                if drawdown_pct > equity_sl_pct:
                    # TRIGGER: Drawdown exceeded threshold - SELL ALL immediately
                    print(f"🔴 EQUITY REGIME TRIGGERED [{date.date()}]: Drawdown={drawdown_pct:.2f}% > SL={equity_sl_pct}% (Peak={peak_equity:.0f}, Current={current_equity:.0f})")
                    
                    equity_regime_active = True
                    self.regime_trigger_events.append({
                        'date': date,
                        'type': 'trigger',
                        'drawdown': drawdown_pct,
                        'peak': peak_equity,
                        'current': current_equity
                    })
                    
                    # Sell all holdings immediately (mid-day sell)
                    for ticker, shares in list(holdings.items()):
                        if ticker in self.data and date in self.data[ticker].index:
                            sell_price = self._get_scalar(self.data[ticker].loc[date, 'Close'])
                            proceeds = shares * sell_price
                            cash += proceeds
                            
                            self.trades.append({
                                'Date': date,
                                'Ticker': ticker,
                                'Action': 'SELL',
                                'Shares': shares,
                                'Price': sell_price,
                                'Value': proceeds,
                                'Reason': 'EQUITY_REGIME_TRIGGER'
                            })
                    holdings = {}
                    regime_active = True
            
            if is_rebalance:
                # Sell all current holdings (regular rebalance sell)
                for ticker, shares in list(holdings.items()):
                    if ticker in self.data and date in self.data[ticker].index:
                        sell_price = self._get_scalar(self.data[ticker].loc[date, 'Close'])
                        proceeds = shares * sell_price
                        cash += proceeds
                        
                        self.trades.append({
                            'Date': date,
                            'Ticker': ticker,
                            'Action': 'SELL',
                            'Shares': shares,
                            'Price': sell_price,
                            'Value': proceeds
                        })
                
                holdings = {}
                
                # Recalculate current equity after sells
                current_equity = cash
                
                # Apply reinvest option
                if reinvest_profits:
                    # Use all available cash (capital + profits)
                    investable_capital = float(cash)
                else:
                    # Cap at initial capital only
                    investable_capital = min(float(cash), self.initial_capital)
                
                # EQUITY REGIME: Check for recovery on rebalance day
                if is_equity_regime and equity_regime_active:
                    # Recalculate drawdown from (frozen) peak
                    drawdown_pct = ((peak_equity - current_equity) / peak_equity) * 100 if peak_equity > 0 else 0
                    
                    if drawdown_pct <= equity_sl_pct:
                        # RECOVERY: Drawdown recovered - resume normal trading
                        print(f"🟢 EQUITY REGIME RECOVERED [{date.date()}]: Drawdown={drawdown_pct:.2f}% <= SL={equity_sl_pct}%")
                        equity_regime_active = False
                        regime_active = False
                        self.regime_trigger_events.append({
                            'date': date,
                            'type': 'recovery',
                            'drawdown': drawdown_pct,
                            'peak': peak_equity,
                            'current': current_equity
                        })
                        # Update peak to current (reset after recovery)
                        peak_equity = current_equity
                    else:
                        print(f"⏳ EQUITY REGIME STILL ACTIVE [{date.date()}]: Drawdown={drawdown_pct:.2f}% > SL={equity_sl_pct}%")
                
                # Check regime filter for non-EQUITY types, or use equity_regime_active for EQUITY type
                if is_equity_regime:
                    regime_triggered = equity_regime_active
                    regime_action = regime_config['action'] if equity_regime_active else 'none'
                    current_drawdown = ((peak_equity - current_equity) / peak_equity) * 100 if peak_equity > 0 else 0
                else:
                    regime_triggered, regime_action, current_drawdown = self._check_regime_filter(date, regime_config, current_equity, peak_equity)
                
                # Calculate allocations based on regime filter + uncorrelated interaction
                stocks_target = 0.0
                uncorrelated_target = 0.0

                
                if regime_triggered:
                    # For EQUITY regime: ALWAYS go 100% cash when triggered (drawdown protection)
                    # For other regimes: respect the action setting
                    if is_equity_regime:
                        # EQUITY regime: 100% cash until recovery - no stocks, no uncorrelated
                        stocks_target = 0.0
                        uncorrelated_target = 0.0  # Force 100% cash for max protection
                        regime_active = True
                        print(f"   EQUITY REGIME ACTIVE: 100% cash (no stocks, no uncorrelated) until recovery")
                    elif regime_action == 'Go Cash':
                        # 0% to stocks, uncorrelated gets its % from total, rest is cash
                        stocks_target = 0.0
                        if uncorrelated_config:
                            allocation_pct = uncorrelated_config['allocation_pct'] / 100.0
                            uncorrelated_target = investable_capital * allocation_pct
                        regime_active = True
                        
                    elif regime_action == 'Half Portfolio':
                        # ALWAYS 50% to stocks, uncorrelated from remaining 50%
                        stocks_target = investable_capital * 0.5
                        if uncorrelated_config:
                            allocation_pct = uncorrelated_config['allocation_pct'] / 100.0
                            # Uncorrelated from the OTHER 50% (cash reserve)
                            uncorrelated_target = (investable_capital * 0.5) * allocation_pct
                        regime_active = True
                else:
                    # No regime triggered - 100% to stocks, NO uncorrelated
                    regime_active = False
                    uncorrelated_target = 0.0
                    stocks_target = investable_capital
                
                # Debug: Log allocations on rebalance days
                if stocks_target == 0:
                    print(f"REBALANCE {date.date()}: REGIME={regime_triggered} ({regime_action}) - NO STOCKS, uncorrelated={uncorrelated_target:.0f}")
                else:
                    print(f"REBALANCE {date.date()}: REGIME={regime_triggered} - stocks={stocks_target:.0f}")
                
                # Execute uncorrelated asset purchase with calculated target
                if uncorrelated_target > 0 and uncorrelated_config:
                    uncorrelated_asset = uncorrelated_config['asset']
                    
                    # Download if needed
                    if uncorrelated_asset not in self.data:
                        try:
                            ticker_ns = uncorrelated_asset if uncorrelated_asset.endswith(('.NS', '.BO')) else f"{uncorrelated_asset}.NS"
                            unc_df = yf.download(ticker_ns, start=self.start_date, end=self.end_date, progress=False, auto_adjust=True)
                            if not unc_df.empty:
                                unc_df.reset_index(inplace=True)
                                unc_df['Date'] = pd.to_datetime(unc_df['Date'])
                                unc_df.set_index('Date', inplace=True)
                                # Reindex to all trading dates and forward-fill gaps (e.g. Oct 24 GOLDBEES)
                                # Use ffill then bfill to handle gaps at start and middle
                                unc_df = unc_df.reindex(all_dates).ffill().bfill()
                                self.data[uncorrelated_asset] = unc_df
                        except Exception as e:
                            print(f"Could not download {uncorrelated_asset}: {e}")
                    
                    # Buy uncorrelated asset
                    if uncorrelated_asset in self.data and date in self.data[uncorrelated_asset].index:
                        unc_price = self._get_scalar(self.data[uncorrelated_asset].loc[date, 'Close'])
                        unc_shares = int(uncorrelated_target / unc_price)
                        
                        if unc_shares > 0:
                            unc_cost = unc_shares * unc_price
                            cash -= unc_cost
                            holdings[uncorrelated_asset] = unc_shares
                            
                            self.trades.append({
                                'Date': date,
                                'Ticker': uncorrelated_asset,
                                'Action': 'BUY',
                                'Shares': unc_shares,
                                'Price': unc_price,
                                'Value': unc_cost,
                                'Score': 0,
                                'Rank': 'Uncorrelated'
                            })
                
                # stocks_target is now the amount available for stocks
                available_for_stocks = stocks_target
                
                # Calculate scores for all stocks - OPTIMIZED VECTORIZED VERSION
                # Exclude uncorrelated asset from stock scoring
                scores = {}
                uncorrelated_asset_ticker = uncorrelated_config['asset'] if uncorrelated_config else None
                
                # Collect all rows for this date (excluding uncorrelated asset)
                date_rows = {}
                for ticker, df in self.data.items():
                    # Skip uncorrelated asset - it's not a stock
                    if ticker == uncorrelated_asset_ticker:
                        continue
                    if date in df.index:
                        date_rows[ticker] = df.loc[date]
                
                # Score all stocks at once using vectorized calculation
                if date_rows:
                    # Create a DataFrame from all rows
                    all_rows_df = pd.DataFrame(date_rows).T
                    
                    # Calculate scores using vectorized method
                    try:
                        scores_series = self.parser.calculate_scores(all_rows_df, scoring_formula)
                        scores = scores_series.to_dict()
                        
                        # Filter out invalid scores
                        scores = {k: v for k, v in scores.items() if v > -999999}
                    except:
                        # Fallback to row-by-row if vectorized fails
                        for ticker, row in date_rows.items():
                            score = self.parser.parse_and_calculate(scoring_formula, row)
                            if score > -999999:
                                scores[ticker] = score
                
                # Rank stocks
                ranked_stocks = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                
                # Debug: Log scoring results
                if len(scores) == 0:
                    print(f"   [WARN] No stocks scored on {date.date()} - check indicator columns")
                elif len(ranked_stocks) < num_stocks:
                    print(f"   [WARN] Only {len(ranked_stocks)} stocks scored (need {num_stocks}) on {date.date()}")
                
                # Select top N stocks
                top_stocks = ranked_stocks[:num_stocks]
                
                # Buy stocks with available_for_stocks amount
                if top_stocks and available_for_stocks > 0:
                    position_value = available_for_stocks / len(top_stocks)
                    
                    for ticker, score in top_stocks:
                        buy_price = self._get_scalar(self.data[ticker].loc[date, 'Close'])
                        shares = int(position_value / buy_price)
                        
                        if shares > 0:
                            cost = shares * buy_price
                            cash -= cost
                            holdings[ticker] = shares
                            
                            self.trades.append({
                                'Date': date,
                                'Ticker': ticker,
                                'Action': 'BUY',
                                'Shares': shares,
                                'Price': buy_price,
                                'Value': cost,
                                'Score': score,
                                'Rank': ranked_stocks.index((ticker, score)) + 1
                            })
                # Update theoretical holdings (for EQUITY regime comparison)
                # Theoretical curve assumes NO regime filter - always trades normally
                if is_equity_regime and is_rebalance:
                    # First, SELL all theoretical holdings (same as actual rebalance does)
                    for ticker, shares in theoretical_holdings.items():
                        if ticker in self.data and date in self.data[ticker].index:
                            sell_price = self._get_scalar(self.data[ticker].loc[date, 'Close'])
                            theoretical_cash += shares * sell_price
                    
                    # Reset theoretical holdings
                    theoretical_holdings = {}
                    
                    # Buy top stocks with all theoretical cash (no regime filter reducing allocation)
                    if top_stocks and theoretical_cash > 0:
                        theo_position_value = theoretical_cash / max(1, len(top_stocks))
                        for ticker, score in top_stocks:
                            buy_price = self._get_scalar(self.data[ticker].loc[date, 'Close'])
                            theo_shares = int(theo_position_value / buy_price)
                            if theo_shares > 0:
                                theoretical_holdings[ticker] = theo_shares
                                theoretical_cash -= theo_shares * buy_price

            
            
            # Calculate portfolio value - use last known price if current data missing
            holdings_value = 0.0
            for ticker, shares in holdings.items():
                if ticker in self.data:
                    if date in self.data[ticker].index:
                        close_price = self._get_scalar(self.data[ticker].loc[date, 'Close'])
                        last_known_prices[ticker] = close_price  # Track last price
                    elif ticker in last_known_prices:
                        close_price = last_known_prices[ticker]  # Use last known
                    else:
                        continue  # No price available at all
                    holdings_value += shares * close_price
            
            total_value = cash + holdings_value
            
            # Calculate current drawdown for tracking
            current_dd = ((peak_equity - total_value) / peak_equity) * 100 if peak_equity > 0 else 0
            
            portfolio_history.append({
                'Date': date,
                'Cash': cash,
                'Holdings': holdings_value,
                'Portfolio Value': total_value,
                'Positions': len(holdings),
                'Regime_Active': regime_active,
                'Peak_Equity': peak_equity,
                'Drawdown_Pct': current_dd,
                'Equity_Regime_Active': equity_regime_active if is_equity_regime else False
            })
        
        # Store results
        self.portfolio_df = pd.DataFrame(portfolio_history).set_index('Date')
        self.trades_df = pd.DataFrame(self.trades)
        
        # Store EQUITY regime analysis data
        if is_equity_regime and theoretical_history:
            self.equity_regime_analysis = {
                'theoretical_curve': pd.DataFrame(theoretical_history).set_index('Date'),
                'trigger_events': self.regime_trigger_events,
                'sl_threshold': equity_sl_pct
            }
    
    def get_equity_regime_analysis(self):
        """Return EQUITY regime filter analysis data for visualization.
        
        Returns:
            dict with 'theoretical_curve', 'trigger_events', 'sl_threshold'
            or None if EQUITY filter was not used
        """
        return self.equity_regime_analysis
    
    def get_metrics(self):
        """Calculate comprehensive performance metrics."""
        if self.portfolio_df.empty:
            return None

        final_value = self.portfolio_df['Portfolio Value'].iloc[-1]
        total_return = final_value - self.initial_capital
        return_pct = (total_return / self.initial_capital) * 100

        # CAGR
        days = (self.portfolio_df.index[-1] - self.portfolio_df.index[0]).days
        years = days / 365.25
        cagr = ((final_value / self.initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0

        # Max Drawdown
        running_max = self.portfolio_df['Portfolio Value'].cummax()
        drawdown = (self.portfolio_df['Portfolio Value'] - running_max) / running_max * 100
        max_dd = abs(drawdown.min())

        # Volatility
        returns = self.portfolio_df['Portfolio Value'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) * 100

        # Sharpe Ratio
        rf_rate = 0.05
        sharpe = (cagr / 100 - rf_rate) / (volatility / 100) if volatility > 0 else 0

        # Win Rate and Trade Statistics
        wins = 0
        losses = 0
        win_amounts = []
        loss_amounts = []
        consecutive_wins = 0
        consecutive_losses = 0
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_streak = 0
        last_was_win = None
        total_trades = 0
        
        if not self.trades_df.empty and 'Action' in self.trades_df.columns:
            # Get BUY and SELL trades
            buy_trades = self.trades_df[self.trades_df['Action'] == 'BUY'].copy()
            sell_trades = self.trades_df[self.trades_df['Action'] == 'SELL'].copy()
            
            # For each SELL, find a matching BUY to calculate PnL
            # Group by Date to get rebalance-level PnL
            if not sell_trades.empty:
                for date in sell_trades['Date'].unique():
                    sells_on_date = sell_trades[sell_trades['Date'] == date]
                    total_sell = sells_on_date['Value'].sum()
                    
                    # Find corresponding previous BUY values (from holdings bought earlier)
                    # For simplicity, calculate rebalance-level PnL (sell_value - buy_value for same tickers)
                    for _, sell_row in sells_on_date.iterrows():
                        ticker = sell_row['Ticker']
                        sell_value = sell_row['Value']
                        
                        # Find previous BUY for this ticker (most recent before this sell)
                        prev_buys = buy_trades[(buy_trades['Ticker'] == ticker) & (buy_trades['Date'] < date)]
                        if not prev_buys.empty:
                            buy_row = prev_buys.iloc[-1]
                            buy_value = buy_row['Value']
                            pnl = sell_value - buy_value
                            
                            total_trades += 1
                            
                            if pnl > 0:
                                wins += 1
                                win_amounts.append(pnl)
                                if last_was_win == True:
                                    current_streak += 1
                                else:
                                    current_streak = 1
                                max_consecutive_wins = max(max_consecutive_wins, current_streak)
                                last_was_win = True
                            elif pnl < 0:
                                losses += 1
                                loss_amounts.append(abs(pnl))
                                if last_was_win == False:
                                    current_streak += 1
                                else:
                                    current_streak = 1
                                max_consecutive_losses = max(max_consecutive_losses, current_streak)
                                last_was_win = False
            
            win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        else:
            win_rate = 0
        
        # Expectancy = (Win% * Avg Win) - (Loss% * Avg Loss)
        avg_win = np.mean(win_amounts) if win_amounts else 0
        avg_loss = np.mean(loss_amounts) if loss_amounts else 0
        win_pct = wins / total_trades if total_trades > 0 else 0
        loss_pct = losses / total_trades if total_trades > 0 else 0
        expectancy = (win_pct * avg_win) - (loss_pct * avg_loss)
        
        # Drawdown Recovery Analysis
        running_max = self.portfolio_df['Portfolio Value'].cummax()
        is_in_drawdown = self.portfolio_df['Portfolio Value'] < running_max
        
        # Find drawdown periods and recovery
        recovery_days = 0
        recovery_trades = 0
        max_recovery_days = 0
        max_recovery_trades = 0
        
        if is_in_drawdown.any():
            # Find where drawdown starts and ends
            drawdown_start = None
            for i, (date, in_dd) in enumerate(is_in_drawdown.items()):
                if in_dd and drawdown_start is None:
                    drawdown_start = date
                elif not in_dd and drawdown_start is not None:
                    # Recovered from drawdown
                    days_in_dd = (date - drawdown_start).days
                    max_recovery_days = max(max_recovery_days, days_in_dd)
                    
                    # Count trades during this period
                    if not self.trades_df.empty and 'Date' in self.trades_df.columns:
                        trades_in_period = self.trades_df[
                            (self.trades_df['Date'] >= drawdown_start) & 
                            (self.trades_df['Date'] <= date)
                        ]
                        max_recovery_trades = max(max_recovery_trades, len(trades_in_period) // 2)
                    
                    drawdown_start = None
        
        # Zerodha Equity Delivery Charges Calculation
        # STT/CTT: 0.1% on buy & sell
        # Transaction charges: NSE 0.00297%
        # GST: 18% on (SEBI + transaction charges)
        # SEBI: ₹10/crore = 0.0001%
        # Stamp: 0.015% on buy side only
        
        total_turnover = 0
        total_buy_value = 0
        total_sell_value = 0
        
        if not self.trades_df.empty and 'Action' in self.trades_df.columns:
            buy_trades = self.trades_df[self.trades_df['Action'] == 'BUY']
            sell_trades = self.trades_df[self.trades_df['Action'] == 'SELL']
            total_buy_value = buy_trades['Value'].sum() if not buy_trades.empty else 0
            total_sell_value = sell_trades['Value'].sum() if not sell_trades.empty else 0
            total_turnover = total_buy_value + total_sell_value
        
        # Calculate charges
        stt_ctt = total_turnover * 0.001  # 0.1% on both sides
        transaction_charges = total_turnover * 0.0000297  # NSE 0.00297%
        sebi_charges = total_turnover * 0.000001  # ₹10/crore = 0.0001%
        stamp_charges = total_buy_value * 0.00015  # 0.015% on buy side
        gst = (transaction_charges + sebi_charges) * 0.18  # 18% GST
        
        total_charges = stt_ctt + transaction_charges + sebi_charges + stamp_charges + gst

        return {
            'Final Value': final_value,
            'Total Return': total_return,
            'Return %': return_pct,
            'CAGR %': cagr,
            'Max Drawdown %': max_dd,
            'Volatility %': volatility,
            'Sharpe Ratio': sharpe,
            'Win Rate %': win_rate,
            'Total Trades': total_trades,
            # New metrics
            'Max Consecutive Wins': max_consecutive_wins,
            'Max Consecutive Losses': max_consecutive_losses,
            'Days to Recover from DD': max_recovery_days,
            'Trades to Recover from DD': max_recovery_trades,
            'Expectancy': expectancy,
            'Avg Win': avg_win,
            'Avg Loss': avg_loss,
            # Zerodha charges
            'Total Turnover': total_turnover,
            'STT/CTT': stt_ctt,
            'Transaction Charges': transaction_charges,
            'SEBI Charges': sebi_charges,
            'Stamp Charges': stamp_charges,
            'GST': gst,
            'Total Charges': total_charges,
        }

    def get_monthly_returns(self):
        """Calculate monthly returns table similar to the format shown."""
        if self.portfolio_df.empty:
            return pd.DataFrame()

        # Get monthly portfolio values
        df = self.portfolio_df.copy()
        df['Year'] = df.index.year
        df['Month'] = df.index.month

        # Get last value of each month
        monthly_values = df.groupby(['Year', 'Month'])['Portfolio Value'].last()

        # Calculate monthly returns
        monthly_returns = monthly_values.pct_change() * 100

        # Pivot to year x month format
        monthly_df = monthly_returns.reset_index()
        monthly_df.columns = ['Year', 'Month', 'Return']

        # Create pivot table
        pivot = monthly_df.pivot(index='Year', columns='Month', values='Return')

        # Rename columns to month names
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        pivot.columns = [month_names[int(m)-1] for m in pivot.columns]

        # Calculate yearly total (compound returns)
        yearly_totals = []
        for year in pivot.index:
            year_data = df[df['Year'] == year]['Portfolio Value']
            if len(year_data) > 0:
                year_return = ((year_data.iloc[-1] / year_data.iloc[0]) - 1) * 100
                yearly_totals.append(year_return)
            else:
                yearly_totals.append(None)

        pivot['Total'] = yearly_totals

        # Reorder columns to have all 12 months + Total
        all_months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for month in all_months:
            if month not in pivot.columns:
                pivot[month] = None

        # Reorder columns
        pivot = pivot[all_months + ['Total']]

        # Format as percentages with proper display
        pivot = pivot.round(3)

        return pivot
