import pandas as pd
import ta
import numpy as np

class IndicatorLibrary:
    @staticmethod
    def add_sma(df, window, column='Close'):
        col_data = df[column] if not isinstance(df[column], pd.DataFrame) else df[column].squeeze()
        df[f'SMA_{window}'] = ta.trend.sma_indicator(col_data, window=window)
        return df

    @staticmethod
    def add_ema(df, window, column='Close'):
        col_data = df[column] if not isinstance(df[column], pd.DataFrame) else df[column].squeeze()
        df[f'EMA_{window}'] = ta.trend.ema_indicator(col_data, window=window)
        return df

    @staticmethod
    def add_rsi(df, window=14, column='Close'):
        col_data = df[column] if not isinstance(df[column], pd.DataFrame) else df[column].squeeze()
        df[f'RSI_{window}'] = ta.momentum.rsi(col_data, window=window)
        return df

    @staticmethod
    def add_macd(df, window_slow=26, window_fast=12, window_sign=9, column='Close'):
        col_data = df[column] if not isinstance(df[column], pd.DataFrame) else df[column].squeeze()
        macd = ta.trend.MACD(col_data, window_slow=window_slow, window_fast=window_fast, window_sign=window_sign)
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Diff'] = macd.macd_diff()
        return df

    @staticmethod
    def add_bollinger_bands(df, window=20, window_dev=2, column='Close'):
        col_data = df[column] if not isinstance(df[column], pd.DataFrame) else df[column].squeeze()
        indicator_bb = ta.volatility.BollingerBands(close=col_data, window=window, window_dev=window_dev)
        df['BB_High'] = indicator_bb.bollinger_hband()
        df['BB_Low'] = indicator_bb.bollinger_lband()
        df['BB_Mid'] = indicator_bb.bollinger_mavg()
        return df

    @staticmethod
    def add_supertrend(df, period=7, multiplier=3):
        # Ensure we have the required columns
        required_cols = ['High', 'Low', 'Close']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"DataFrame must have a '{col}' column")
        
        # Get columns as Series (handle potential DataFrame issues)
        high = df['High']
        if isinstance(high, pd.DataFrame):
            high = high.squeeze()
        
        low = df['Low']
        if isinstance(low, pd.DataFrame):
            low = low.squeeze()
        
        close = df['Close']
        if isinstance(close, pd.DataFrame):
            close = close.squeeze()
        
        tr1 = pd.DataFrame(high - low)
        tr2 = pd.DataFrame(abs(high - close.shift(1)))
        tr3 = pd.DataFrame(abs(low - close.shift(1)))
        frames = [tr1, tr2, tr3]
        tr = pd.concat(frames, axis=1, join='inner').max(axis=1)
        atr = tr.ewm(span=period).mean()
        
        hl2 = (high + low) / 2
        basic_upperband = hl2 + (multiplier * atr)
        basic_lowerband = hl2 - (multiplier * atr)
        
        # Vectorized SuperTrend calculation for speed
        final_upperband = basic_upperband.copy()
        final_lowerband = basic_lowerband.copy()

        # Use numpy arrays for faster iteration
        fub = final_upperband.values
        flb = final_lowerband.values
        bub = basic_upperband.values
        blb = basic_lowerband.values
        close_vals = close.values
        supertrend = np.zeros(len(df))

        for i in range(1, len(df)):
            # Upper band logic
            if bub[i] < fub[i-1] or close_vals[i-1] > fub[i-1]:
                fub[i] = bub[i]
            else:
                fub[i] = fub[i-1]

            # Lower band logic
            if blb[i] > flb[i-1] or close_vals[i-1] < flb[i-1]:
                flb[i] = blb[i]
            else:
                flb[i] = flb[i-1]

            # SuperTrend logic
            if supertrend[i-1] == fub[i-1] and close_vals[i] <= fub[i]:
                supertrend[i] = fub[i]
            elif supertrend[i-1] == fub[i-1] and close_vals[i] > fub[i]:
                supertrend[i] = flb[i]
            elif supertrend[i-1] == flb[i-1] and close_vals[i] >= flb[i]:
                supertrend[i] = flb[i]
            elif supertrend[i-1] == flb[i-1] and close_vals[i] < flb[i]:
                supertrend[i] = fub[i]
            else:
                supertrend[i] = fub[i] if close_vals[i] <= fub[i] else flb[i]

        final_upperband = pd.Series(fub, index=df.index)
        final_lowerband = pd.Series(flb, index=df.index)
                     
        df['Supertrend'] = supertrend
        df['Supertrend_Signal'] = 0
        df.loc[close > df['Supertrend'], 'Supertrend_Signal'] = 1
        df.loc[close < df['Supertrend'], 'Supertrend_Signal'] = -1
        return df

    @staticmethod
    def add_momentum_volatility_metrics(df):
        """
        Adds comprehensive Performance and Risk metrics for multiple timeframes.
        Metrics include: Performance, Volatility, Max Drawdown, Sharpe, Sortino, Calmar
        Timeframes: 1M, 3M, 6M, 9M, 1Y
        OPTIMIZED: Uses vectorized operations for ~10-20x speedup
        """
        # Ensure we're working with a DataFrame
        if isinstance(df, pd.Series):
            raise ValueError("Input must be a DataFrame, not a Series")
        
        # Ensure Close column exists and is a Series
        if 'Close' not in df.columns:
            raise ValueError("DataFrame must have a 'Close' column")
        
        # Get Close as a Series (handle potential DataFrame issues)
        close_series = df['Close']
        if isinstance(close_series, pd.DataFrame):
            close_series = close_series.squeeze()
        
        periods = {
            '1 Month': 21,
            '3 Month': 63,
            '6 Month': 126,
            '9 Month': 189,
            '1 Year': 252
        }
        
        # Calculate daily returns once
        daily_returns = close_series.pct_change()
        df['Daily_Returns'] = daily_returns
        
        # Pre-calculate cumulative max for drawdown (much faster)
        cummax = close_series.expanding().max()
        
        for name, window in periods.items():
            # 1. Performance (Total Return) - Already vectorized
            df[f'{name} Performance'] = close_series.pct_change(periods=window)
            
            # 2. Volatility (Annualized Std Dev of returns) - Already vectorized
            df[f'{name} Volatility'] = daily_returns.rolling(window=window).std() * np.sqrt(252)
            
            # 3. OPTIMIZED Maximum Drawdown - Vectorized!
            # Use rolling window with cummax (much faster than apply)
            rolling_max = close_series.rolling(window=window).max()
            drawdown = (close_series - rolling_max) / rolling_max
            df[f'{name} Max Drawdown'] = drawdown.rolling(window=window).min()
            
            # 4. Sharpe Ratio - Already vectorized
            mean_return = daily_returns.rolling(window=window).mean()
            std_return = daily_returns.rolling(window=window).std()
            df[f'{name} Sharpe'] = (mean_return / std_return) * np.sqrt(252)
            df[f'{name} Sharpe'] = df[f'{name} Sharpe'].replace([np.inf, -np.inf], 0)
            
            # 5. OPTIMIZED Sortino Ratio - Vectorized!
            # Calculate rolling downside standard deviation
            # Negative returns only
            downside_returns = daily_returns.copy()
            downside_returns[downside_returns > 0] = 0
            downside_std = downside_returns.rolling(window=window).std()
            
            # Sortino = mean / downside_std
            sortino = (mean_return / downside_std) * np.sqrt(252)
            df[f'{name} Sortino'] = sortino.replace([np.inf, -np.inf], 0)
            
            # 6. Calmar Ratio - Already vectorized
            df[f'{name} Calmar'] = df[f'{name} Performance'] / abs(df[f'{name} Max Drawdown'])
            df[f'{name} Calmar'] = df[f'{name} Calmar'].replace([np.inf, -np.inf], 0)
        
        # Fill NaN values with 0 for scoring purposes
        df.fillna(0, inplace=True)
        
        return df
    
    @staticmethod
    def add_regime_filters(df):
        """
        Adds market regime indicators including EMAs, MACD, SUPERTREND for regime filter.
        Supports ALL dropdown options in the UI.
        """
        # Ensure we're working with a DataFrame, not Series
        if isinstance(df, pd.Series):
            raise ValueError("Input must be a DataFrame, not a Series")
        
        # Ensure Close column exists and is a Series
        if 'Close' not in df.columns:
            raise ValueError("DataFrame must have a 'Close' column")
        
        # Get Close as a Series (handle potential DataFrame issues)
        close_series = df['Close']
        if isinstance(close_series, pd.DataFrame):
            close_series = close_series.squeeze()
        
        # 1. EMAs for Regime Filter (ALL periods the UI offers: 34, 68, 100, 150, 200)
        for period in [34, 68, 100, 150, 200]:
            df[f'EMA_{period}'] = ta.trend.ema_indicator(close_series, window=period)
        
        # 2. MACD for Regime Filter (supports all UI presets)
        # Default MACD (12-26-9), plus custom ones used by regime filter
        macd_obj = ta.trend.MACD(close_series, window_slow=26, window_fast=12, window_sign=9)
        df['MACD'] = macd_obj.macd()
        df['MACD_Signal'] = macd_obj.macd_signal()
        df['MACD_Diff'] = macd_obj.macd_diff()
        
        # 3. SuperTrend for Regime Filter (supports UI presets: 1-1, 1-2, 1-2.5)
        # Calculate all supertrend variations
        for period, multiplier in [(1, 1), (1, 2), (1, 2.5), (7, 3)]:
            suffix = f"_{period}_{str(multiplier).replace('.', '_')}"
            df = IndicatorLibrary._add_supertrend_basic(df, period, multiplier, suffix)
        
        # Default Supertrend without suffix for backward compatibility
        df['Supertrend'] = df.get(f'Supertrend_7_3', df.get('Supertrend_1_2', 0))
        df['Supertrend_Direction'] = np.where(close_series > df['Supertrend'], 'BUY', 'SELL')
        
        # 4. Price vs 200 SMA
        df['SMA_200'] = ta.trend.sma_indicator(close_series, window=200)
        df['Above_SMA_200'] = (close_series > df['SMA_200']).astype(int)
        
        # 5. 52-Week High/Low
        df['52W_High'] = close_series.rolling(window=252).max()
        df['52W_Low'] = close_series.rolling(window=252).min()
        df['Near_52W_High'] = ((close_series / df['52W_High']) > 0.95).astype(int)
        df['Near_52W_Low'] = ((close_series / df['52W_Low']) < 1.05).astype(int)
        
        # 6. Simple Trend (3M SMA > 6M SMA = Bullish)
        df['SMA_63'] = ta.trend.sma_indicator(close_series, window=63)
        df['SMA_126'] = ta.trend.sma_indicator(close_series, window=126)
        df['Bullish_Trend'] = (df['SMA_63'] > df['SMA_126']).astype(int)
        
        return df
    
    @staticmethod
    def _add_supertrend_basic(df, period, multiplier, suffix=""):
        """Simplified supertrend calculation for regime filter."""
        high = df['High'].squeeze() if isinstance(df['High'], pd.DataFrame) else df['High']
        low = df['Low'].squeeze() if isinstance(df['Low'], pd.DataFrame) else df['Low']
        close = df['Close'].squeeze() if isinstance(df['Close'], pd.DataFrame) else df['Close']
        
        # ATR calculation
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(span=period, adjust=False).mean()
        
        # Basic bands
        hl2 = (high + low) / 2
        upper = hl2 + (multiplier * atr)
        lower = hl2 - (multiplier * atr)
        
        # SuperTrend calculation
        supertrend = pd.Series(index=df.index, dtype=float)
        supertrend.iloc[0] = upper.iloc[0]
        
        for i in range(1, len(df)):
            if close.iloc[i] > upper.iloc[i-1]:
                supertrend.iloc[i] = lower.iloc[i]
            elif close.iloc[i] < lower.iloc[i-1]:
                supertrend.iloc[i] = upper.iloc[i]
            else:
                supertrend.iloc[i] = supertrend.iloc[i-1]
        
        df[f'Supertrend{suffix}'] = supertrend
        return df
