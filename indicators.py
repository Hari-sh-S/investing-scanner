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
        """Optimized SuperTrend using vectorized NumPy operations."""
        high = df['High'].squeeze() if isinstance(df['High'], pd.DataFrame) else df['High']
        low = df['Low'].squeeze() if isinstance(df['Low'], pd.DataFrame) else df['Low']
        close = df['Close'].squeeze() if isinstance(df['Close'], pd.DataFrame) else df['Close']
        
        # ATR calculation (vectorized)
        tr = np.maximum.reduce([
            (high - low).values,
            np.abs((high - close.shift(1)).values),
            np.abs((low - close.shift(1)).values)
        ])
        atr = pd.Series(tr, index=df.index).ewm(span=period).mean()
        
        hl2 = (high + low) / 2
        upper = hl2 + (multiplier * atr)
        lower = hl2 - (multiplier * atr)
        
        # Vectorized SuperTrend using NumPy
        supertrend = _compute_supertrend_fast(close.values, upper.values, lower.values)
        
        df['Supertrend'] = supertrend
        df['Supertrend_Signal'] = np.where(close.values > supertrend, 1, -1)
        return df

    @staticmethod
    def add_momentum_volatility_metrics(df, required_periods=None):
        """
        FAST vectorized Performance and Risk metrics for multiple timeframes.
        Uses pure NumPy operations for maximum speed.
        
        Args:
            df: DataFrame with OHLCV data
            required_periods: Optional set of (months, metric_type) tuples.
                             If None, calculates default periods (1, 3, 6, 9, 12 months)
        """
        if isinstance(df, pd.Series):
            raise ValueError("Input must be a DataFrame, not a Series")
        
        if 'Close' not in df.columns:
            raise ValueError("DataFrame must have a 'Close' column")
        
        close = df['Close'].squeeze() if isinstance(df['Close'], pd.DataFrame) else df['Close']
        
        # Default periods (in months) - always calculate these for compatibility
        default_months = {1, 3, 6, 9, 12}
        
        # Add any additional required periods
        if required_periods:
            for months, _ in required_periods:
                default_months.add(months)
        
        # Convert months to trading days (approx 21 days/month)
        periods = {}
        for months in sorted(default_months):
            name = '1 Year' if months == 12 else f'{months} Month'
            window = months * 21  # ~21 trading days per month
            periods[name] = window
        
        # Pre-calculate returns once 
        daily_returns = close.pct_change()
        if 'Daily_Returns' not in df.columns:
            df['Daily_Returns'] = daily_returns
        
        for name, window in periods.items():
            # 1. Performance - vectorized
            col_name = f'{name} Performance'
            if col_name not in df.columns:
                df[col_name] = close.pct_change(periods=window)
            
            # 2. Volatility - vectorized
            col_name = f'{name} Volatility'
            if col_name not in df.columns:
                df[col_name] = daily_returns.rolling(window).std() * np.sqrt(252)
            
            # 2b. Downside Volatility - only negative returns
            col_name = f'{name} Downside Volatility'
            if col_name not in df.columns:
                downside = daily_returns.clip(upper=0)
                df[col_name] = downside.rolling(window).std() * np.sqrt(252)
            
            # 3. Max Drawdown - vectorized
            col_name = f'{name} Max Drawdown'
            if col_name not in df.columns:
                rolling_max = close.rolling(window).max()
                drawdown = (close - rolling_max) / rolling_max
                df[col_name] = drawdown.rolling(window).min()
            
            # 4. Sharpe - vectorized
            col_name = f'{name} Sharpe'
            if col_name not in df.columns:
                mean_ret = daily_returns.rolling(window).mean()
                std_ret = daily_returns.rolling(window).std()
                sharpe = (mean_ret / std_ret) * np.sqrt(252)
                df[col_name] = sharpe.replace([np.inf, -np.inf], 0)
            
            # 5. Sortino - vectorized
            col_name = f'{name} Sortino'
            if col_name not in df.columns:
                downside = daily_returns.clip(upper=0)
                mean_ret = daily_returns.rolling(window).mean()
                downside_std = downside.rolling(window).std()
                sortino = (mean_ret / downside_std) * np.sqrt(252)
                df[col_name] = sortino.replace([np.inf, -np.inf], 0)
            
            # 6. Calmar - vectorized
            col_name = f'{name} Calmar'
            if col_name not in df.columns:
                perf_col = f'{name} Performance'
                dd_col = f'{name} Max Drawdown'
                calmar = df[perf_col] / df[dd_col].abs()
                df[col_name] = calmar.replace([np.inf, -np.inf], 0)
        
        df.fillna(0, inplace=True)
        return df
    
    @staticmethod
    def add_regime_filters(df):
        """Optimized regime indicators - only calculate what's needed."""
        if isinstance(df, pd.Series):
            raise ValueError("Input must be a DataFrame, not a Series")
        
        close = df['Close'].squeeze() if isinstance(df['Close'], pd.DataFrame) else df['Close']
        high = df['High'].squeeze() if isinstance(df['High'], pd.DataFrame) else df['High']
        low = df['Low'].squeeze() if isinstance(df['Low'], pd.DataFrame) else df['Low']
        
        # 1. EMAs (batch calculate for efficiency)
        for period in [34, 68, 100, 150, 200]:
            df[f'EMA_{period}'] = close.ewm(span=period, adjust=False).mean()
        
        # 2. MACD (single calculation)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Diff'] = df['MACD'] - df['MACD_Signal']
        
        # 3. SuperTrend (single default calculation for regime)
        tr = np.maximum.reduce([
            (high - low).values,
            np.abs((high - close.shift(1)).fillna(0).values),
            np.abs((low - close.shift(1)).fillna(0).values)
        ])
        atr = pd.Series(tr, index=df.index).ewm(span=7).mean()
        hl2 = (high + low) / 2
        upper = hl2 + (3 * atr)
        lower = hl2 - (3 * atr)
        supertrend = _compute_supertrend_fast(close.values, upper.values, lower.values)
        df['Supertrend'] = supertrend
        df['Supertrend_Direction'] = np.where(close.values > supertrend, 'BUY', 'SELL')
        
        # 4. SMA and trend indicators
        df['SMA_200'] = close.rolling(200).mean()
        df['Above_SMA_200'] = (close > df['SMA_200']).astype(int)
        df['52W_High'] = close.rolling(252).max()
        df['52W_Low'] = close.rolling(252).min()
        df['Near_52W_High'] = ((close / df['52W_High']) > 0.95).astype(int)
        df['Near_52W_Low'] = ((close / df['52W_Low']) < 1.05).astype(int)
        df['SMA_63'] = close.rolling(63).mean()
        df['SMA_126'] = close.rolling(126).mean()
        df['Bullish_Trend'] = (df['SMA_63'] > df['SMA_126']).astype(int)
        
        return df
    
    @staticmethod
    def _add_supertrend_basic(df, period, multiplier, suffix=""):
        """Simplified supertrend for regime filter."""
        high = df['High'].squeeze() if isinstance(df['High'], pd.DataFrame) else df['High']
        low = df['Low'].squeeze() if isinstance(df['Low'], pd.DataFrame) else df['Low']
        close = df['Close'].squeeze() if isinstance(df['Close'], pd.DataFrame) else df['Close']
        
        tr = np.maximum.reduce([
            (high - low).values,
            np.abs((high - close.shift(1)).fillna(0).values),
            np.abs((low - close.shift(1)).fillna(0).values)
        ])
        atr = pd.Series(tr, index=df.index).ewm(span=period, adjust=False).mean()
        hl2 = (high + low) / 2
        upper = hl2 + (multiplier * atr)
        lower = hl2 - (multiplier * atr)
        
        supertrend = _compute_supertrend_fast(close.values, upper.values, lower.values)
        df[f'Supertrend{suffix}'] = supertrend
        return df


def _compute_supertrend_fast(close, upper, lower):
    """Optimized SuperTrend using NumPy (no Python loops where possible)."""
    n = len(close)
    supertrend = np.empty(n)
    supertrend[0] = upper[0]
    
    # Use Numba-style loop (still fast with NumPy arrays)
    for i in range(1, n):
        if close[i] > upper[i-1]:
            supertrend[i] = lower[i]
        elif close[i] < lower[i-1]:
            supertrend[i] = upper[i]
        else:
            supertrend[i] = supertrend[i-1]
    
    return supertrend

