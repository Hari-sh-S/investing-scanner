import pandas as pd
import numpy as np
import yfinance as yf
from indicators import IndicatorLibrary

class BacktestEngine:
    def __init__(self, ticker, start_date, end_date, initial_capital=100000):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.data = pd.DataFrame()
        self.trades = []
        self.equity_curve = []

    def fetch_data(self):
        # Handle Indian tickers
        symbol = self.ticker if self.ticker.endswith(('.NS', '.BO')) else f"{self.ticker}.NS"
        try:
            self.data = yf.download(symbol, start=self.start_date, end=self.end_date, progress=False)
            if isinstance(self.data.columns, pd.MultiIndex):
                self.data.columns = self.data.columns.get_level_values(0)
            
            if self.data.empty:
                return False
            return True
        except Exception as e:
            print(f"Error fetching data: {e}")
            return False

    def add_indicators(self, indicators_config):
        """
        indicators_config: list of dicts, e.g.
        [{'type': 'SMA', 'params': {'window': 50}}, {'type': 'RSI', 'params': {'window': 14}}]
        """
        for ind in indicators_config:
            if ind['type'] == 'SMA':
                self.data = IndicatorLibrary.add_sma(self.data, **ind['params'])
            elif ind['type'] == 'EMA':
                self.data = IndicatorLibrary.add_ema(self.data, **ind['params'])
            elif ind['type'] == 'RSI':
                self.data = IndicatorLibrary.add_rsi(self.data, **ind['params'])
            elif ind['type'] == 'MACD':
                self.data = IndicatorLibrary.add_macd(self.data, **ind['params'])
            elif ind['type'] == 'Bollinger Bands':
                self.data = IndicatorLibrary.add_bollinger_bands(self.data, **ind['params'])
            elif ind['type'] == 'Supertrend':
                self.data = IndicatorLibrary.add_supertrend(self.data, **ind['params'])

    def run_strategy(self, entry_logic, exit_logic, stop_loss_pct=None, target_pct=None):
        """
        entry_logic: function(row) -> bool
        exit_logic: function(row) -> bool
        """
        position = 0 # 0: Flat, 1: Long, -1: Short (Short not fully implemented for equity delivery)
        entry_price = 0
        capital = self.initial_capital
        self.trades = []
        self.equity_curve = []

        # Pre-calculate logic if possible, but row-by-row is easier for complex conditions
        # For performance, vectorization is better, but for custom logic flexibility, iteration is okay for <10k rows.
        
        for index, row in self.data.iterrows():
            current_price = row['Close']
            date = index
            
            # Update Equity Curve (Mark to Market)
            if position == 1:
                current_equity = capital + (current_price - entry_price) * (capital // entry_price) # Approx
            else:
                current_equity = capital
            self.equity_curve.append({'Date': date, 'Equity': current_equity})

            # Check Exit Conditions (Stop Loss / Target)
            if position == 1:
                pct_change = (current_price - entry_price) / entry_price
                
                # Stop Loss
                if stop_loss_pct and pct_change <= -stop_loss_pct:
                    position = 0
                    capital = current_equity
                    self.trades.append({
                        'Entry Date': entry_date,
                        'Exit Date': date,
                        'Entry Price': entry_price,
                        'Exit Price': current_price,
                        'PnL': (current_price - entry_price) * qty,
                        'PnL %': pct_change * 100,
                        'Reason': 'Stop Loss'
                    })
                    continue

                # Target
                if target_pct and pct_change >= target_pct:
                    position = 0
                    capital = current_equity
                    self.trades.append({
                        'Entry Date': entry_date,
                        'Exit Date': date,
                        'Entry Price': entry_price,
                        'Exit Price': current_price,
                        'PnL': (current_price - entry_price) * qty,
                        'PnL %': pct_change * 100,
                        'Reason': 'Target'
                    })
                    continue
                
                # Strategy Exit
                if exit_logic(row):
                    position = 0
                    capital = current_equity
                    self.trades.append({
                        'Entry Date': entry_date,
                        'Exit Date': date,
                        'Entry Price': entry_price,
                        'Exit Price': current_price,
                        'PnL': (current_price - entry_price) * qty,
                        'PnL %': pct_change * 100,
                        'Reason': 'Strategy Exit'
                    })
                    continue

            # Check Entry Conditions
            if position == 0:
                if entry_logic(row):
                    position = 1
                    entry_price = current_price
                    entry_date = date
                    qty = capital // entry_price # Simple position sizing: All in
                    # Adjust capital for next iteration's MTM
                    # Real capital deduction happens on exit in this simple model, 
                    # but we track 'capital' as available cash. 
                    # Actually, let's track Total Portfolio Value.
                    
        self.equity_df = pd.DataFrame(self.equity_curve).set_index('Date')
        self.trades_df = pd.DataFrame(self.trades)

    def get_metrics(self):
        if self.trades_df.empty:
            return {
                "Total Return": 0,
                "Win Rate": 0,
                "Total Trades": 0,
                "Max Drawdown": 0
            }

        total_return = self.equity_df['Equity'].iloc[-1] - self.initial_capital
        total_return_pct = (total_return / self.initial_capital) * 100
        
        wins = self.trades_df[self.trades_df['PnL'] > 0]
        win_rate = (len(wins) / len(self.trades_df)) * 100
        
        # Max Drawdown
        running_max = self.equity_df['Equity'].cummax()
        drawdown = (self.equity_df['Equity'] - running_max) / running_max
        max_drawdown = drawdown.min() * 100

        return {
            "Total Return (INR)": round(total_return, 2),
            "Total Return (%)": round(total_return_pct, 2),
            "Win Rate (%)": round(win_rate, 2),
            "Total Trades": len(self.trades_df),
            "Max Drawdown (%)": round(max_drawdown, 2)
        }
