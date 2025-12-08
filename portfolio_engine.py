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
            return pd.read_parquet(cache_path)
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

    @staticmethod
    def _get_scalar(value):
        """Safely extract scalar from potential Series or DataFrame."""
        if isinstance(value, (pd.Series, pd.DataFrame)):
            return value.iloc[0] if len(value) > 0 else 0
        return value

    def download_and_cache_universe(self, universe_tickers, progress_callback=None, stop_flag=None):
        """Sequential download to avoid yfinance threading issues. Calculates indicators immediately."""
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
        last_update = start_time

        # Sequential download (yfinance has threading issues)
        for i, ticker in enumerate(tickers_to_download):
            # Check stop flag
            if stop_flag and stop_flag[0]:
                print(f"Stopped at {i}/{len(tickers_to_download)}")
                break

            try:
                ticker_ns = ticker if ticker.endswith(('.NS', '.BO')) else f"{ticker}.NS"
                df = yf.download(ticker_ns, period="max", interval="1d", progress=False, auto_adjust=True)

                if not df.empty and len(df) >= 100:
                    # Reset index to make Date a column
                    df.reset_index(inplace=True)

                    # Keep only OHLCV columns
                    expected_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                    df = df[[col for col in expected_cols if col in df.columns]]
                    
                    # OPTIMIZATION: Calculate indicators immediately during download
                    try:
                        # Set index for indicator calculation
                        df_with_date_index = df.set_index('Date')
                        
                        # Add momentum and volatility metrics
                        df_with_date_index = IndicatorLibrary.add_momentum_volatility_metrics(df_with_date_index)
                        
                        # Add regime filters
                        df_with_date_index = IndicatorLibrary.add_regime_filters(df_with_date_index)
                        
                        # Reset index back to column for caching
                        df = df_with_date_index.reset_index()
                    except Exception as e:
                        print(f"Indicator calculation failed for {ticker}: {e}")
                        # Continue with raw data if indicators fail

                    if self.cache:
                        self.cache.set(ticker, df)
                        success_count += 1

                # Update every 3 seconds
                current_time = time.time()
                if progress_callback and (current_time - last_update >= 3.0):
                    elapsed = current_time - start_time
                    avg = elapsed / (i + 1) if (i + 1) > 0 else 0
                    remaining_time = avg * (len(tickers_to_download) - (i + 1))
                    # Check if callback accepts 4 args (with remaining_time) or 3 args
                    try:
                        progress_callback(i + 1, len(tickers_to_download), ticker, remaining_time)
                    except TypeError:
                        progress_callback(i + 1, len(tickers_to_download), ticker)
                    last_update = current_time

            except Exception as e:
                print(f"Download error for {ticker}: {e}")
                continue

        # Final update
        if progress_callback:
            try:
                progress_callback(len(tickers_to_download), len(tickers_to_download), "Done", 0)
            except TypeError:
                progress_callback(len(tickers_to_download), len(tickers_to_download), "Done")

        print(f"Downloaded {success_count}/{len(tickers_to_download)} stocks (with indicators) in {time.time() - start_time:.1f}s")
        return len(universe_tickers)

    def fetch_data(self, progress_callback=None):
        """Fetch data from cache. Indicators should already be pre-calculated during download."""
        print(f"Loading data for {len(self.universe)} stocks...")
        tickers_to_download = []

        # First, try to load from cache
        for i, ticker in enumerate(self.universe):
            if progress_callback:
                progress_callback(i + 1, len(self.universe), ticker)

            if self.cache:
                cached_data = self.cache.get(ticker)
                if cached_data is not None:
                    # Fix index - Date should be the index
                    if 'Date' in cached_data.columns:
                        cached_data['Date'] = pd.to_datetime(cached_data['Date'])
                        cached_data.set_index('Date', inplace=True)

                    # Ensure index is datetime
                    if not isinstance(cached_data.index, pd.DatetimeIndex):
                        cached_data.index = pd.to_datetime(cached_data.index)

                    # Filter to date range
                    mask = (cached_data.index >= pd.Timestamp(self.start_date)) & \
                           (cached_data.index <= pd.Timestamp(self.end_date))
                    df_filtered = cached_data[mask].copy()

                    if not df_filtered.empty and len(df_filtered) >= 100:
                        self.data[ticker] = df_filtered
                        continue

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
                        # Fix index - Date should be the index
                        if 'Date' in cached_data.columns:
                            cached_data['Date'] = pd.to_datetime(cached_data['Date'])
                            cached_data.set_index('Date', inplace=True)

                        # Ensure index is datetime
                        if not isinstance(cached_data.index, pd.DatetimeIndex):
                            cached_data.index = pd.to_datetime(cached_data.index)

                        mask = (cached_data.index >= pd.Timestamp(self.start_date)) & \
                               (cached_data.index <= pd.Timestamp(self.end_date))
                        df_filtered = cached_data[mask].copy()

                        if not df_filtered.empty:
                            self.data[ticker] = df_filtered

        print(f"Successfully loaded {len(self.data)} stocks")
        return len(self.data) > 0

    def _get_rebalance_dates(self, all_dates, rebal_config):
        """Generate rebalance dates based on config."""
        freq = rebal_config['frequency']
        
        if freq == 'Weekly':
            # Get day of week (0=Monday, 4=Friday)
            day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4}
            target_day = day_map[rebal_config['day']]
            
            rebalance_dates = [d for d in all_dates if d.weekday() == target_day]
        else:  # Monthly
            target_date = rebal_config['date']
            alt_option = rebal_config['alt_day']
            
            rebalance_dates = []
            for date in all_dates:
                if date.day == target_date:
                    rebalance_dates.append(date)
                elif alt_option == 'Previous Day' and date.day == target_date - 1:
                    # Check if target_date doesn't exist in this month
                    next_day = date + timedelta(days=1)
                    if next_day.month != date.month or next_day not in all_dates:
                        rebalance_dates.append(date)
                elif alt_option == 'Next Day' and date.day == target_date + 1:
                    # Check if target_date doesn't exist
                    prev_day = date - timedelta(days=1)
                    if prev_day.day != target_date or prev_day not in all_dates:
                        rebalance_dates.append(date)
        
        return rebalance_dates

    def _check_regime_filter(self, date, regime_config, realized_pnl=0):
        """Check if regime filter is triggered."""
        if not regime_config:
            return False, 'none'  # No filter active
        
        regime_type = regime_config['type']
        
        if regime_type == 'EQUITY':
            # Check realized P&L
            sl_pct = regime_config['value']
            if realized_pnl < -sl_pct:
                return True, regime_config['action']
            return False, 'none'
        
        
        # For EMA, MACD, SUPERTREND - need index data
        if self.regime_index_data is None or self.regime_index_data.empty:
            return False, 'none'
        
        if date not in self.regime_index_data.index:
            return False, 'none'
        
        row = self.regime_index_data.loc[date]
        
        if regime_type == 'EMA':
            ema_period = regime_config['value']
            ema_col = f'EMA_{ema_period}'
            if ema_col in row and row['Close'] < row.get(ema_col, 0):
                return True, regime_config['action']
        
        elif regime_type == 'MACD':
            # Check MACD signal
            if row.get('MACD', 0) < row.get('MACD_Signal', 0):
                return True, regime_config['action']
        
        elif regime_type == 'SUPERTREND':
            # Check Supertrend
            if row.get('Supertrend', 'BUY') == 'SELL':
                return True, regime_config['action']
        
        return False, 'none'

    def run_rebalance_strategy(self, scoring_formula, num_stocks, exit_rank, 
                              rebal_config, regime_config=None, uncorrelated_config=None):
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
                'NIFTY MNC': '^CNXMN C'
            }
            index_ticker = index_map.get(regime_index, '^NSEI')
            
            try:
                regime_data = yf.download(index_ticker, start=self.start_date, end=self.end_date, progress=False)
                if not regime_data.empty:
                    regime_data = IndicatorLibrary.add_regime_filters(regime_data)
                    self.regime_index_data = regime_data
            except:
                print("Could not load regime index data")
        
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
        realized_pnl_running = 0
        
        for date in all_dates:
            is_rebalance = date in rebalance_dates
            
            # Check regime filter
            regime_triggered, regime_action = self._check_regime_filter(date, regime_config, realized_pnl_running)
            
            if is_rebalance:
                # Sell all current holdings
                for ticker, shares in holdings.items():
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
                
                # Calculate allocations based on regime filter + uncorrelated asset combination
                total_funds = float(cash)
                stocks_allocation = 0.0
                uncorrelated_allocation = 0.0
                cash_reserve = 0.0
                
                if regime_triggered:
                    if regime_action == 'Go Cash':
                        # 0% stocks, uncorrelated from total funds, rest to cash
                        stocks_allocation = 0.0
                        if uncorrelated_config:
                            allocation_pct = uncorrelated_config['allocation_pct'] / 100.0
                            uncorrelated_allocation = total_funds * allocation_pct
                            cash_reserve = total_funds - uncorrelated_allocation
                        else:
                            uncorrelated_allocation = 0.0
                            cash_reserve = total_funds
                        
                        regime_active = True
                        regime_cash_reserve = cash_reserve
                        
                    elif regime_action == 'Half Portfolio':
                        # 50% available, split between stocks and uncorrelated
                        available_funds = total_funds * 0.5
                        cash_reserve = total_funds * 0.5
                        
                        if uncorrelated_config:
                            # Uncorrelated gets allocation_pct of the AVAILABLE 50%
                            allocation_pct = uncorrelated_config['allocation_pct'] / 100.0
                            uncorrelated_allocation = available_funds * allocation_pct
                            stocks_allocation = available_funds - uncorrelated_allocation
                        else:
                            uncorrelated_allocation = 0.0
                            stocks_allocation = available_funds
                        
                        regime_active = True
                        regime_cash_reserve = cash_reserve
                else:
                    # No regime filter active - use all funds
                    regime_active = False
                    regime_cash_reserve = 0.0
                    
                    if uncorrelated_config:
                        allocation_pct = uncorrelated_config['allocation_pct'] / 100.0
                        uncorrelated_allocation = total_funds * allocation_pct
                        stocks_allocation = total_funds - uncorrelated_allocation
                    else:
                        uncorrelated_allocation = 0.0
                        stocks_allocation = total_funds
                
                
                # Execute uncorrelated asset purchase
                if uncorrelated_config and uncorrelated_allocation > 0:
                    uncorrelated_asset = uncorrelated_config['asset']
                    
                    # Download uncorrelated asset data if not in universe
                    if uncorrelated_asset not in self.data:
                        try:
                            ticker_ns = uncorrelated_asset if uncorrelated_asset.endswith(('.NS', '.BO')) else f"{uncorrelated_asset}.NS"
                            unc_df = yf.download(ticker_ns, start=self.start_date, end=self.end_date, progress=False, auto_adjust=True)
                            if not unc_df.empty:
                                unc_df.reset_index(inplace=True)
                                unc_df['Date'] = pd.to_datetime(unc_df['Date'])
                                unc_df.set_index('Date', inplace=True)
                                self.data[uncorrelated_asset] = unc_df
                        except Exception as e:
                            print(f"Could not download {uncorrelated_asset}: {e}")
                    
                    # Buy uncorrelated asset
                    if uncorrelated_asset in self.data and date in self.data[uncorrelated_asset].index:
                        unc_price = self._get_scalar(self.data[uncorrelated_asset].loc[date, 'Close'])
                        unc_shares = int(uncorrelated_allocation / unc_price)
                        
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
                
                # Calculate scores for all stocks - OPTIMIZED VECTORIZED VERSION
                scores = {}
                
                # Collect all rows for this date
                date_rows = {}
                for ticker, df in self.data.items():
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
                
                # Select top N stocks
                top_stocks = ranked_stocks[:num_stocks] if stocks_allocation > 0 else []
                
                # Buy top stocks (with calculated stocks allocation)
                if top_stocks and stocks_allocation > 0:
                    position_value = stocks_allocation / len(top_stocks)
                    
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
            
            
            # Calculate portfolio value
            holdings_value = 0.0
            for ticker, shares in holdings.items():
                if ticker in self.data and date in self.data[ticker].index:
                    close_price = self._get_scalar(self.data[ticker].loc[date, 'Close'])
                    holdings_value += shares * close_price
            
            total_value = cash + holdings_value
            portfolio_history.append({
                'Date': date,
                'Cash': cash,
                'Holdings': holdings_value,
                'Portfolio Value': total_value,
                'Positions': len(holdings),
                'Regime_Active': regime_active
            })
        
        # Store results
        self.portfolio_df = pd.DataFrame(portfolio_history).set_index('Date')
        self.trades_df = pd.DataFrame(self.trades)
    
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

        # Win Rate
        if not self.trades_df.empty:
            trades_grouped = self.trades_df.groupby(['Ticker', self.trades_df.index // 2]).apply(
                lambda x: x.iloc[-1]['Value'] - x.iloc[0]['Value'] if len(x) > 1 else 0
            ).dropna()
            wins = (trades_grouped > 0).sum()
            total_trades = len(trades_grouped)
            win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        else:
            win_rate = 0

        return {
            'Final Value': final_value,
            'Total Return': total_return,
            'Return %': return_pct,
            'CAGR %': cagr,
            'Max Drawdown %': max_dd,
            'Volatility %': volatility,
            'Sharpe Ratio': sharpe,
            'Win Rate %': win_rate,
            'Total Trades': len(self.trades_df) if not self.trades_df.empty else 0
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
