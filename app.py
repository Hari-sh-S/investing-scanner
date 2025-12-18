import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from engine import BacktestEngine
from portfolio_engine import PortfolioEngine
from scoring import ScoreParser
from nifty_universe import (get_all_universe_names, get_universe, 
                            get_broad_market_universes, get_sectoral_universes,
                            get_cap_based_universes, get_thematic_universes)
from report_generator import create_excel_with_charts, create_pdf_report, prepare_complete_log_data
import datetime
import io
import time
import json
from pathlib import Path

# Page config - MUST be first Streamlit command
st.set_page_config(
    page_title="Investing Scanner",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Show loading indicator immediately for wake-up
with st.spinner("🔄 App is waking up... Please wait..."):
    pass  # Spinner shows during import time

# Initialize session state for backtest logs
BACKTEST_LOG_FILE = Path("backtest_logs.json")

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_backtest_logs_cached():
    """Load backtest logs from file with caching."""
    if BACKTEST_LOG_FILE.exists():
        try:
            with open(BACKTEST_LOG_FILE, 'r') as f:
                logs_data = json.load(f)
                return logs_data
        except Exception as e:
            print(f"Error loading logs: {e}")
            return []
    return []

def load_backtest_logs():
    """Load backtest logs from file."""
    return load_backtest_logs_cached()

def save_backtest_logs(logs):
    """Save backtest logs to file with complete data (no truncation)."""
    try:
        serializable_logs = []
        for log in logs:
            serializable_log = {
                'timestamp': log['timestamp'],
                'name': log['name'],
                'config': log['config'],
                'metrics': log['metrics'],
                # Store complete data without truncation
                'portfolio_values': log.get('portfolio_values', []),
                'trades': log.get('trades', []),
                'monthly_returns': log.get('monthly_returns', {})
            }
            serializable_logs.append(serializable_log)
        
        with open(BACKTEST_LOG_FILE, 'w') as f:
            json.dump(serializable_logs, f, indent=2, default=str)
        # Clear cache after saving
        load_backtest_logs_cached.clear()
    except Exception as e:
        print(f"Error saving logs: {e}")

@st.cache_data(ttl=86400)  # Cache universe names for 24 hours
def get_cached_universe_names():
    """Cache universe names to speed up app loading."""
    try:
        return sorted(get_all_universe_names())
    except Exception:
        return ["NIFTY 50", "NIFTY 100", "NIFTY 200"]  # Fallback

# Initialize session state with error handling
try:
    if 'backtest_logs' not in st.session_state:
        st.session_state.backtest_logs = load_backtest_logs()
    if 'backtest_engines' not in st.session_state:
        st.session_state.backtest_engines = {}
    if 'app_ready' not in st.session_state:
        st.session_state.app_ready = True
except Exception as e:
    st.error(f"Error initializing app: {e}. Please refresh the page.")
    st.session_state.backtest_logs = []
    st.session_state.backtest_engines = {}
    st.session_state.app_ready = True

# CSS
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 0rem;
    }
    section[data-testid="stSidebar"] {
        display: none;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        background-color: white;
        border-radius: 6px;
        padding: 0 20px;
        border: 1px solid #ddd;
        color: #1a1a1a !important;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #28a745;
        color: white !important;
        font-weight: 600;
    }
    .progress-text {
        font-size: 16px;
        color: #ffffff;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .stock-name {
        color: #00ff88;
        font-weight: 700;
        font-size: 18px;
        text-shadow: 0 0 10px rgba(0, 255, 136, 0.5);
    }
    .time-remaining {
        font-size: 14px;
        color: #aaaaaa;
        margin-left: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Header
col_title, col_actions = st.columns([3, 1])
with col_title:
    st.title("Investing Scanner")
    st.caption("Advanced Backtesting for Indian Stock Market")

st.markdown("---")

# Main Tabs
main_tabs = st.tabs(["Backtest", "Backtest Logs", "Data Download"])

# ==================== TAB 1: BACKTEST ====================
with main_tabs[0]:
    col_config, col_scoring, col_metrics = st.columns([1.2, 2, 1])
    
    with col_config:
        st.subheader("Configuration")
        
        st.markdown("**Universe**")
        
        # Get all available universes
        all_universes = sorted(get_all_universe_names()) + ["Custom"]
        
        selected_universe = st.selectbox(
            "Select", 
            all_universes,
            label_visibility="collapsed"
        )
        
        if selected_universe == "Custom":
            custom_input = st.text_input("Stocks (comma-separated)", "RELIANCE, TCS, INFY")
            universe = [s.strip() for s in custom_input.split(',')]
        else:
            universe = get_universe(selected_universe)
            st.caption(f"{len(universe)} stocks")
        
        st.markdown("**Capital & Size**")
        initial_capital = st.number_input("Starting Capital (₹)", 10000, 100000000, 100000, 10000)
        num_stocks = st.number_input("No. of Stocks in Portfolio*", 1, 50, 5)
        exit_rank = st.number_input("Exit Rank*", num_stocks, 200, num_stocks * 2, 
                                    help="Stocks will exit if they fall below this rank. Should be > Portfolio Size")
        
        reinvest_profits = st.checkbox("Reinvest Profits", value=True,
                                       help="If enabled, reinvest starting capital + profits. If disabled, only reinvest initial capital amount.")
        
        st.markdown("**Time Period**")
        start_date = st.date_input("Start Date", datetime.date(2020, 1, 1))
        end_date = st.date_input("End Date", datetime.date.today())
        
        st.markdown("**Rebalancing**")
        rebal_freq_options = ["Weekly", "Monthly"]
        rebalance_label = st.selectbox("Frequency", rebal_freq_options, index=1)
        
        if rebalance_label == "Weekly":
            rebal_day = st.selectbox("Rebalance Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
            rebalance_date = None
        else:  # Monthly
            rebalance_date = st.number_input("Rebalance Date (1-30)", 1, 30, 1,
                                            help="Day of month to rebalance portfolio")
            rebal_day = None
        
        alt_day_option = st.selectbox("Alternative Rebalance Day", 
                                     ["Previous Day", "Next Day"],
                                     index=1,
                                     help="If rebalance day is holiday, use this option")
        
        st.markdown("**Regime Filter**")
        use_regime_filter = st.checkbox("Enable Regime Filter", value=False)
        
        regime_config = None
        if use_regime_filter:
            regime_type = st.selectbox("Regime Filter Type", 
                                      ["EMA", "MACD", "SUPERTREND", "EQUITY", "EQUITY_MA"],
                                      help="EQUITY_MA: Uses moving average of your equity curve (meta-strategy)")
            
            # Initialize defaults
            recovery_dd = None
            ma_period = None
            
            if regime_type == "EMA":
                ema_period = st.selectbox("EMA Period", [34, 68, 100, 150, 200])
                regime_value = ema_period
            elif regime_type == "MACD":
                macd_preset = st.selectbox("MACD Settings", 
                                          ["35-70-12", "50-100-15", "75-150-12"])
                regime_value = macd_preset
            elif regime_type == "SUPERTREND":
                st_preset = st.selectbox("SuperTrend (Period-Multiplier)", 
                                        ["1-1", "1-2", "1-2.5"])
                regime_value = st_preset
            elif regime_type == "EQUITY":
                realized_sl = st.number_input("Drawdown SL % (Trigger)", 1, 50, 10,
                                              help="Sell all holdings when drawdown from peak equity exceeds this %")
                recovery_dd = st.number_input("Recovery DD % (Re-entry)", 0, 49, 5,
                                              help="Only re-enter market when drawdown recovers below this %. Should be less than Trigger % to avoid whipsaw.")
                regime_value = realized_sl
            else:  # EQUITY_MA
                ma_period = st.selectbox("Equity Curve MA Period (days)", 
                                        [20, 30, 50, 100, 200],
                                        index=2,  # Default to 50
                                        help="Reduce exposure when portfolio equity falls below this MA")
                regime_value = ma_period
            
            # Regime action - now available for ALL types including EQUITY
            regime_action = st.selectbox("Regime Filter Action",
                                        ["Go Cash", "Half Portfolio"],
                                        help="Action to take when regime filter triggers")
            
            # Index selection for regime filter - only for non-EQUITY and non-EQUITY_MA types
            if regime_type not in ["EQUITY", "EQUITY_MA"]:
                regime_index = st.selectbox("Regime Filter Index", sorted(get_all_universe_names()))
            else:
                regime_index = None
            
            regime_config = {
                'type': regime_type,
                'value': regime_value,
                'action': regime_action,
                'index': regime_index,
                'recovery_dd': recovery_dd,  # Recovery threshold for EQUITY regime
                'ma_period': ma_period if regime_type == "EQUITY_MA" else None  # MA period for EQUITY_MA
            }
            
            # Uncorrelated Asset - ONLY when regime filter is enabled
            st.markdown("**Uncorrelated Asset**")
            use_uncorrelated = st.checkbox("Invest in Uncorrelated Asset", value=False,
                                          help="Allocate to uncorrelated asset when regime filter triggers")
            
            uncorrelated_config = None
            if use_uncorrelated:
                asset_type = st.text_input("Asset Type", "GOLDBEES",
                                          help="Enter ticker symbol (e.g., GOLDBEES for Gold)")
                allocation_pct = st.number_input("Allocation %", 1, 100, 20,
                                                help="% of portfolio value to allocate when regime triggers")
                
                uncorrelated_config = {
                    'asset': asset_type,
                    'allocation_pct': allocation_pct
                }
        else:
            uncorrelated_config = None
    
    with col_scoring:
        st.subheader("Scoring Console")
        
        parser = ScoreParser()
        examples = parser.get_example_formulas()
        
        template = st.selectbox("Template", ["Custom"] + list(examples.keys()))
        default = examples.get(template, "6 Month Performance")
        
        formula = st.text_area("Scoring Formula", default, height=120)
        
        valid, msg = parser.validate_formula(formula)
        if valid:
            st.success("✅ " + msg)
        else:
            st.error("❌ " + msg)
        
        st.markdown("---")
        run_btn = st.button("🚀 Run Backtest", type="primary")
    
    with col_metrics:
        st.subheader("Metrics")
        st.markdown("**Performance**")
        for m in ["1 Month Performance", "3 Month Performance", "6 Month Performance"]:
            st.caption(m)
        
        st.markdown("**Volatility**")
        for m in ["1 Month Volatility", "3 Month Volatility", "6 Month Volatility"]:
            st.caption(m)
        
        st.markdown("**Risk-Adjusted**")
        for m in ["6 Month Sharpe", "6 Month Sortino", "6 Month Calmar"]:
            st.caption(m)
    
    # Results Section
    if run_btn:
        if not valid:
            st.error("Fix formula first")
        else:
            # Initialize tracking variables
            start_time = time.time()
            processed_count = [0]  # Use list to avoid nonlocal scope issues
            total_count = len(universe)
            
            def progress_callback(current, total, ticker):
                processed_count[0] = current

                # Calculate time stats
                elapsed = time.time() - start_time
                elapsed_mins = int(elapsed // 60)
                elapsed_secs = int(elapsed % 60)

                if processed_count[0] > 0:
                    avg_time_per_stock = elapsed / processed_count[0]
                    remaining_stocks = total - processed_count[0]
                    time_remaining_sec = avg_time_per_stock * remaining_stocks

                    remaining_mins = int(time_remaining_sec // 60)
                    remaining_secs = int(time_remaining_sec % 60)
                    time_str = f"{remaining_mins:02d}:{remaining_secs:02d}"
                else:
                    time_str = "Calculating..."

                # Update progress bar
                progress = min(processed_count[0] / total, 1.0)
                prog_bar.progress(progress)

                # Update status text with all details
                pct = (processed_count[0] / total * 100) if total > 0 else 0
                status_container.markdown(f"""
                <div style="padding: 10px; background: rgba(0,0,0,0.1); border-radius: 5px;">
                    <div style="font-size: 16px; font-weight: bold;">📊 {ticker}</div>
                    <div style="margin-top: 5px;">
                        Progress: {processed_count[0]}/{total} ({pct:.1f}%)
                    </div>
                    <div style="margin-top: 5px;">
                        ⏱️ Remaining: {time_str} | ⏰ Elapsed: {elapsed_mins:02d}:{elapsed_secs:02d}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with st.spinner("Initializing backtest..."):
                prog_bar = st.progress(0)
                status_container = st.empty()
                
                engine = PortfolioEngine(universe, start_date, end_date, initial_capital)
                if engine.fetch_data(progress_callback=progress_callback):
                    prog_bar.empty()
                    status_container.empty()
                    
                    with st.spinner("Running strategy simulation..."):
                        # Build rebalance config
                        rebal_config = {
                            'frequency': rebalance_label,
                            'date': rebalance_date,
                            'day': rebal_day,
                            'alt_day': alt_day_option
                        }
                        
                        engine.run_rebalance_strategy(
                            formula, 
                            num_stocks,
                            exit_rank,
                            rebal_config,
                            regime_config,
                            uncorrelated_config,
                            reinvest_profits
                        )
                        metrics = engine.get_metrics()
                        
                        # Store in session_state so results persist across reruns (for benchmark comparison)
                        st.session_state['backtest_engine'] = engine
                        st.session_state['backtest_metrics'] = metrics
                        st.session_state['backtest_start_date'] = start_date
                        st.session_state['backtest_end_date'] = end_date
                    
                    if metrics:
                        # Prepare complete log data (no truncation)
                        complete_log_data = prepare_complete_log_data(
                            {
                                'name': f"Backtest_{datetime.datetime.now().strftime('%m%d_%H%M')}",
                                'initial_capital': initial_capital,
                                'universe_name': selected_universe,
                                'num_stocks': num_stocks,
                                'exit_rank': exit_rank,
                                'rebalance_freq': rebalance_label,
                                'start_date': start_date.strftime('%Y-%m-%d'),
                                'end_date': end_date.strftime('%Y-%m-%d'),
                                'regime_config': regime_config if regime_config else {},
                                'uncorrelated_config': uncorrelated_config if uncorrelated_config else {},
                                'formula': formula
                            },
                            metrics,
                            engine
                        )
                        
                        # Save to logs with complete data
                        backtest_log = {
                            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'name': f"Backtest_{datetime.datetime.now().strftime('%m%d_%H%M')}",
                            'config': complete_log_data['config'],
                            'metrics': metrics,
                            'portfolio_values': complete_log_data['portfolio_values'],
                            'trades': complete_log_data['trades'],
                            'monthly_returns': complete_log_data['monthly_returns']
                        }
                        st.session_state.backtest_logs.append(backtest_log)
                        save_backtest_logs(st.session_state.backtest_logs)  # Save to file
                        
                        # Store current backtest data in session_state for persistence
                        st.session_state['current_backtest'] = {
                            'engine': engine,
                            'metrics': metrics,
                            'backtest_log': backtest_log,
                            'start_date': start_date,
                            'end_date': end_date
                        }
                        st.session_state['current_backtest_active'] = True
                        
                        st.markdown("---")
                        
                        # Action buttons - Excel and PDF downloads
                        col_h, col_excel, col_pdf = st.columns([3, 1, 1])
                        with col_h:
                            st.subheader("Backtest Results")
                        with col_excel:
                            excel_data = create_excel_with_charts(backtest_log['config'], metrics, engine)
                            st.download_button(
                                label="📥 Excel",
                                data=excel_data,
                                file_name=f"backtest_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        with col_pdf:
                            pdf_data = create_pdf_report(backtest_log['config'], metrics, engine)
                            st.download_button(
                                label="📄 PDF",
                                data=pdf_data,
                                file_name=f"backtest_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                mime="application/pdf"
                            )
                        
                        # Result tabs - add Equity Regime Testing tab if EQUITY filter is used
                        equity_analysis = None
                        if hasattr(engine, 'get_equity_regime_analysis'):
                            equity_analysis = engine.get_equity_regime_analysis()
                        
                        # Build tab list dynamically
                        tab_names = ["Performance Metrics", "Charts", "Monthly Breakup", "Monthly Report", "Trade History"]
                        if equity_analysis:
                            tab_names.append("Equity Regime Testing")
                        
                        result_tabs = st.tabs(tab_names)
                        
                        with result_tabs[0]:
                            st.markdown("### Key Performance Indicators")
                            
                            kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
                            
                            kpi_col1.metric("Final Value", f"₹{metrics['Final Value']:,.0f}")
                            kpi_col1.metric("Total Return", f"₹{metrics['Total Return']:,.0f}")
                            kpi_col1.metric("Return %", f"{metrics['Return %']:.2f}%")
                            
                            kpi_col2.metric("CAGR %", f"{metrics['CAGR %']:.2f}%")
                            kpi_col2.metric("Max Drawdown %", f"{metrics['Max Drawdown %']:.2f}%")
                            kpi_col2.metric("Volatility %", f"{metrics.get('Volatility %', 0):.2f}%")
                            
                            kpi_col3.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")
                            kpi_col3.metric("Win Rate %", f"{metrics['Win Rate %']:.2f}%")
                            kpi_col3.metric("Total Trades", metrics['Total Trades'])
                            
                            kpi_col4.metric("Avg Trade/Year", f"{metrics['Total Trades'] / max(1, (end_date - start_date).days / 365.25):.1f}")
                            kpi_col4.metric("Expectancy", f"₹{metrics.get('Expectancy', 0):,.0f}")
                            
                            # Additional Metrics Row
                            st.markdown("---")
                            st.markdown("**📊 Advanced Metrics**")
                            adv_col1, adv_col2, adv_col3, adv_col4 = st.columns(4)
                            
                            adv_col1.metric("Max Consecutive Wins", metrics.get('Max Consecutive Wins', 0))
                            adv_col1.metric("Max Consecutive Losses", metrics.get('Max Consecutive Losses', 0))
                            
                            adv_col2.metric("Avg Win", f"₹{metrics.get('Avg Win', 0):,.0f}")
                            adv_col2.metric("Avg Loss", f"₹{metrics.get('Avg Loss', 0):,.0f}")
                            
                            adv_col3.metric("Days to Recover from DD", metrics.get('Days to Recover from DD', 0))
                            adv_col3.metric("Trades to Recover from DD", metrics.get('Trades to Recover from DD', 0))
                            
                            adv_col4.metric("Total Turnover", f"₹{metrics.get('Total Turnover', 0):,.0f}")
                            adv_col4.metric("Total Charges (Zerodha)", f"₹{metrics.get('Total Charges', 0):,.0f}")
                            
                            # Charges Breakdown Expander
                            with st.expander("📋 Zerodha Charges Breakdown"):
                                charges_col1, charges_col2 = st.columns(2)
                                charges_col1.write(f"**STT/CTT (0.1%):** ₹{metrics.get('STT/CTT', 0):,.2f}")
                                charges_col1.write(f"**Transaction Charges:** ₹{metrics.get('Transaction Charges', 0):,.2f}")
                                charges_col1.write(f"**SEBI Charges:** ₹{metrics.get('SEBI Charges', 0):,.2f}")
                                charges_col2.write(f"**Stamp Charges (0.015%):** ₹{metrics.get('Stamp Charges', 0):,.2f}")
                                charges_col2.write(f"**GST (18%):** ₹{metrics.get('GST', 0):,.2f}")
                                charges_col2.write(f"**Total Charges:** ₹{metrics.get('Total Charges', 0):,.2f}")
                        with result_tabs[1]:
                            st.markdown("### Performance Charts")
                            
                            # Equity Curve
                            fig_equity = go.Figure()
                            fig_equity.add_trace(go.Scatter(
                                x=engine.portfolio_df.index,
                                y=engine.portfolio_df['Portfolio Value'],
                                fill='tozeroy',
                                line_color='#28a745',
                                name='Portfolio Value'
                            ))
                            fig_equity.update_layout(
                                title="Equity Curve",
                                xaxis_title="Date",
                                yaxis_title="Portfolio Value (₹)",
                                height=400,
                                margin=dict(l=0,r=0,t=40,b=0),
                                showlegend=False,
                                template='plotly_white'
                            )
                            st.plotly_chart(fig_equity, use_container_width=True)
                            
                            # Drawdown Chart
                            running_max = engine.portfolio_df['Portfolio Value'].cummax()
                            dd = (engine.portfolio_df['Portfolio Value'] - running_max) / running_max * 100
                            
                            fig_dd = go.Figure()
                            fig_dd.add_trace(go.Scatter(
                                x=dd.index,
                                y=dd,
                                fill='tozeroy',
                                line_color='#dc3545',
                                name='Drawdown'
                            ))
                            fig_dd.update_layout(
                                title="Drawdown Analysis",
                                xaxis_title="Date",
                                yaxis_title="Drawdown %",
                                height=350,
                                margin=dict(l=0,r=0,t=40,b=0),
                                showlegend=False,
                                template='plotly_white'
                            )
                            st.plotly_chart(fig_dd, use_container_width=True)

                        with result_tabs[2]:
                            st.markdown("### Monthly Returns Breakup")
                            if not engine.portfolio_df.empty:
                                monthly_returns = engine.get_monthly_returns()
                                if not monthly_returns.empty:
                                    # Format the dataframe for display
                                    display_monthly = monthly_returns.copy()

                                    # Format percentages with color coding
                                    def color_negative_red(val):
                                        if pd.isna(val):
                                            return ''
                                        color = '#28a745' if val > 0 else '#dc3545' if val < 0 else '#6c757d'
                                        return f'color: {color}; font-weight: 600'

                                    # Apply styling
                                    styled_df = display_monthly.style.applymap(color_negative_red)

                                    st.dataframe(styled_df, use_container_width=True, height=400)

                                    # Summary statistics
                                    st.markdown("---")
                                    col1, col2, col3, col4 = st.columns(4)

                                    all_returns = monthly_returns.iloc[:, :-1].values.flatten()
                                    all_returns = all_returns[~pd.isna(all_returns)]

                                    if len(all_returns) > 0:
                                        col1.metric("Positive Months", f"{(all_returns > 0).sum()} ({(all_returns > 0).sum()/len(all_returns)*100:.1f}%)")
                                        col2.metric("Negative Months", f"{(all_returns < 0).sum()} ({(all_returns < 0).sum()/len(all_returns)*100:.1f}%)")
                                        col3.metric("Best Month", f"{all_returns.max():.2f}%")
                                        col4.metric("Worst Month", f"{all_returns.min():.2f}%")
                            else:
                                st.info("No data available for monthly breakdown")

                        with result_tabs[3]:
                            st.markdown("### Monthly Portfolio Report")
                            if not engine.portfolio_df.empty:
                                monthly_df = engine.portfolio_df.copy()
                                monthly_df['Year'] = monthly_df.index.year
                                monthly_df['Month'] = monthly_df.index.month
                                monthly_df['Day'] = monthly_df.index.day
                                
                                display_df = monthly_df[['Year', 'Month', 'Day', 'Portfolio Value', 'Cash', 'Positions']]
                                st.dataframe(display_df, use_container_width=True, height=400)

                        with result_tabs[4]:
                            st.markdown("### Trade History")
                            if not engine.trades_df.empty:
                                # Create consolidated trade view matching BUY with SELL
                                trades_df = engine.trades_df.copy()
                                buy_trades = trades_df[trades_df['Action'] == 'BUY'].copy()
                                sell_trades = trades_df[trades_df['Action'] == 'SELL'].copy()
                                
                                consolidated_trades = []
                                
                                for _, sell in sell_trades.iterrows():
                                    ticker = sell['Ticker']
                                    sell_date = sell['Date']
                                    
                                    # Find the most recent BUY for this ticker before this SELL
                                    prev_buys = buy_trades[
                                        (buy_trades['Ticker'] == ticker) & 
                                        (buy_trades['Date'] < sell_date)
                                    ]
                                    
                                    if not prev_buys.empty:
                                        buy = prev_buys.iloc[-1]
                                        buy_price = float(buy['Price'])
                                        sell_price = float(sell['Price'])
                                        shares = int(buy['Shares'])
                                        roi = ((sell_price - buy_price) / buy_price) * 100
                                        
                                        consolidated_trades.append({
                                            'Stock': ticker.replace('.NS', ''),
                                            'Buy Date': pd.to_datetime(buy['Date']).strftime('%Y-%m-%d'),
                                            'Buy Price': round(buy_price, 2),
                                            'Exit Date': pd.to_datetime(sell_date).strftime('%Y-%m-%d'),
                                            'Exit Price': round(sell_price, 2),
                                            'Shares': shares,
                                            'ROI %': round(roi, 2)
                                        })
                                
                                if consolidated_trades:
                                    trade_display = pd.DataFrame(consolidated_trades)
                                    
                                    # Color ROI column
                                    def color_roi(val):
                                        if val > 0:
                                            return 'color: #28a745; font-weight: bold'
                                        elif val < 0:
                                            return 'color: #dc3545; font-weight: bold'
                                        return ''
                                    
                                    styled_trades = trade_display.style.applymap(
                                        color_roi, subset=['ROI %']
                                    )
                                    st.dataframe(styled_trades, use_container_width=True, height=400)
                                    
                                    # Summary stats
                                    st.markdown("---")
                                    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                                    stat_col1.metric("Total Trades", len(consolidated_trades))
                                    profitable = len([t for t in consolidated_trades if t['ROI %'] > 0])
                                    stat_col2.metric("Profitable", f"{profitable} ({profitable/len(consolidated_trades)*100:.1f}%)")
                                    avg_roi = sum(t['ROI %'] for t in consolidated_trades) / len(consolidated_trades)
                                    stat_col3.metric("Avg ROI", f"{avg_roi:.2f}%")
                                    best_trade = max(consolidated_trades, key=lambda x: x['ROI %'])
                                    stat_col4.metric("Best Trade", f"{best_trade['Stock']} ({best_trade['ROI %']:.1f}%)")
                                else:
                                    st.info("No completed trades to display")
                            else:
                                st.info("No trades executed")
                        # Equity Regime Testing Tab (only shown if EQUITY regime filter was used)
                        if equity_analysis:
                            with result_tabs[5]:  # Last tab
                                st.markdown("### 📊 Equity Regime Testing")
                                st.warning("⚠️ **DISCLAIMER**: This section is for testing purposes only. The theoretical curve shows what would have happened WITHOUT the EQUITY regime filter.")
                                
                                st.markdown(f"**Stop-Loss Threshold:** {equity_analysis['sl_threshold']}%")
                                
                                # Trigger Events Table
                                trigger_events = equity_analysis.get('trigger_events', [])
                                if trigger_events:
                                    st.markdown("### Regime Trigger Events")
                                    events_data = []
                                    for event in trigger_events:
                                        events_data.append({
                                            'Date': event['date'].strftime('%Y-%m-%d'),
                                            'Event': '🔴 TRIGGERED' if event['type'] == 'trigger' else '🟢 RECOVERED',
                                            'Drawdown %': f"{event['drawdown']:.2f}%",
                                            'Peak Equity': f"₹{event['peak']:,.0f}",
                                            'Current Equity': f"₹{event['current']:,.0f}"
                                        })
                                    st.dataframe(pd.DataFrame(events_data), use_container_width=True)
                                else:
                                    st.info("No regime triggers during this backtest period. The drawdown never exceeded your SL threshold.")
                                
                                # Theoretical vs Actual Equity Curve
                                theoretical_curve = equity_analysis.get('theoretical_curve')
                                if theoretical_curve is not None and not theoretical_curve.empty:
                                    st.markdown("### Theoretical vs Actual Equity Curve")
                                    st.caption("Shows what would have happened WITHOUT the EQUITY regime filter (no mid-drawdown exits)")
                                    
                                    fig_compare = go.Figure()
                                    fig_compare.add_trace(go.Scatter(
                                        x=engine.portfolio_df.index,
                                        y=engine.portfolio_df['Portfolio Value'],
                                        name='Actual (With EQUITY Filter)',
                                        line=dict(color='#28a745', width=2)
                                    ))
                                    fig_compare.add_trace(go.Scatter(
                                        x=theoretical_curve.index,
                                        y=theoretical_curve['Theoretical_Equity'],
                                        name='Theoretical (Without Filter)',
                                        line=dict(color='#007bff', width=2, dash='dot')
                                    ))
                                    fig_compare.update_layout(
                                        title="Actual vs Theoretical Equity Curve",
                                        xaxis_title="Date",
                                        yaxis_title="Portfolio Value (₹)",
                                        height=450,
                                        template='plotly_dark'
                                    )
                                    st.plotly_chart(fig_compare, use_container_width=True)
                                    
                                    # Summary metrics comparison
                                    actual_final = engine.portfolio_df['Portfolio Value'].iloc[-1]
                                    theoretical_final = theoretical_curve['Theoretical_Equity'].iloc[-1]
                                    actual_return = ((actual_final / engine.initial_capital) - 1) * 100
                                    theoretical_return = ((theoretical_final / engine.initial_capital) - 1) * 100
                                    
                                    comp_col1, comp_col2, comp_col3 = st.columns(3)
                                    comp_col1.metric("Actual Final Value", f"₹{actual_final:,.0f}")
                                    comp_col2.metric("Theoretical Final Value", f"₹{theoretical_final:,.0f}")
                                    
                                    diff = actual_return - theoretical_return
                                    if diff > 0:
                                        comp_col3.metric("Filter Benefit", f"+{diff:.2f}%", delta=f"+{diff:.2f}%")
                                    else:
                                        comp_col3.metric("Filter Impact", f"{diff:.2f}%", delta=f"{diff:.2f}%")
                                
                                st.markdown("---")
                                
                                # Peak Equity and Drawdown Chart
                                st.markdown("### Peak Equity & Drawdown Tracking")
                                if 'Peak_Equity' in engine.portfolio_df.columns:
                                    fig_peak = go.Figure()
                                    fig_peak.add_trace(go.Scatter(
                                        x=engine.portfolio_df.index,
                                        y=engine.portfolio_df['Portfolio Value'],
                                        name='Portfolio Value',
                                        line=dict(color='#28a745', width=2)
                                    ))
                                    fig_peak.add_trace(go.Scatter(
                                        x=engine.portfolio_df.index,
                                        y=engine.portfolio_df['Peak_Equity'],
                                        name='Peak Equity',
                                        line=dict(color='#ffc107', width=2, dash='dot')
                                    ))
                                    # Add threshold line from peak
                                    threshold_line = engine.portfolio_df['Peak_Equity'] * (1 - equity_analysis['sl_threshold'] / 100)
                                    fig_peak.add_trace(go.Scatter(
                                        x=engine.portfolio_df.index,
                                        y=threshold_line,
                                        name=f"SL Threshold ({equity_analysis['sl_threshold']}%)",
                                        line=dict(color='#dc3545', width=1, dash='dash')
                                    ))
                                    
                                    # Add trigger event markers
                                    for event in trigger_events:
                                        color = 'red' if event['type'] == 'trigger' else 'green'
                                        symbol = 'triangle-down' if event['type'] == 'trigger' else 'triangle-up'
                                        fig_peak.add_trace(go.Scatter(
                                            x=[event['date']],
                                            y=[event['current']],
                                            mode='markers',
                                            marker=dict(size=12, color=color, symbol=symbol),
                                            name=f"{'Trigger' if event['type'] == 'trigger' else 'Recovery'} ({event['date'].strftime('%Y-%m-%d')})",
                                            showlegend=True
                                        ))
                                    
                                    fig_peak.update_layout(
                                        title="Portfolio Value vs Peak Equity with SL Threshold",
                                        xaxis_title="Date",
                                        yaxis_title="Value (₹)",
                                        height=450,
                                        template='plotly_dark'
                                    )
                                    st.plotly_chart(fig_peak, use_container_width=True)
                                
                                # Drawdown percentage chart
                                st.markdown("### Drawdown from Peak")
                                if 'Drawdown_Pct' in engine.portfolio_df.columns:
                                    fig_dd = go.Figure()
                                    fig_dd.add_trace(go.Scatter(
                                        x=engine.portfolio_df.index,
                                        y=-engine.portfolio_df['Drawdown_Pct'],  # Negative to show as positive area below
                                        fill='tozeroy',
                                        line=dict(color='#dc3545', width=1),
                                        name='Drawdown %'
                                    ))
                                    # Add threshold line
                                    fig_dd.add_hline(
                                        y=-equity_analysis['sl_threshold'], 
                                        line_dash="dash", 
                                        line_color="yellow",
                                        annotation_text=f"SL Threshold ({equity_analysis['sl_threshold']}%)"
                                    )
                                    fig_dd.update_layout(
                                        title="Drawdown % from Peak Equity",
                                        xaxis_title="Date",
                                        yaxis_title="Drawdown %",
                                        height=350,
                                        template='plotly_dark'
                                    )
                                    st.plotly_chart(fig_dd, use_container_width=True)
                                    
                                    # Answer user's question in the UI
                                    st.markdown("---")
                                    st.info(f"**Note:** With EQUITY regime filter enabled at {equity_analysis['sl_threshold']}% SL, the maximum drawdown should be approximately capped at this threshold. When the drawdown breaches the SL, all positions are sold to prevent further losses.")
                    else:
                        st.warning("No trades generated")
                else:
                    st.error("Data fetch failed")
    
    # STANDALONE BENCHMARK COMPARISON - Persists across reruns using session_state
    if st.session_state.get('current_backtest_active') and 'current_backtest' in st.session_state:
        st.markdown("---")
        st.subheader("📊 Benchmark Comparison")
        
        stored_data = st.session_state['current_backtest']
        stored_engine = stored_data['engine']
        bt_start = stored_data['start_date']
        bt_end = stored_data['end_date']
        
        # Yahoo Finance index mappings - extensive list of available NSE indices
        yahoo_index_map = {
            # Major Indices
            "NIFTY 50": "^NSEI",
            "NIFTY NEXT 50": "^NSMIDCP",
            "NIFTY 100": "^CNX100",
            "NIFTY 200": "^CNX200",
            "NIFTY 500": "^CRSLDX",
            "NIFTY BANK": "^NSEBANK",
            "NIFTY FIN SERVICE": "^CNXFIN",
            "NIFTY IT": "^CNXIT",
            # Midcap & Smallcap
            "NIFTY MIDCAP 50": "^NIFTYMIDCAP50",
            "NIFTY MIDCAP 100": "^CNXMDCP",
            "NIFTY SMLCAP 50": "^NSMALLCAP50",
            "NIFTY SMLCAP 100": "^CNXSC",
            "NIFTY SMLCAP 250": "^NSMALLCAP250",
            # Sectoral
            "NIFTY AUTO": "^CNXAUTO",
            "NIFTY PHARMA": "^CNXPHARMA",
            "NIFTY PSE": "^CNXPSE",
            "NIFTY REALTY": "^CNXREALTY",
            "NIFTY INFRA": "^CNXINFRA",
            "NIFTY ENERGY": "^CNXENERGY",
            "NIFTY FMCG": "^CNXFMCG",
            "NIFTY METAL": "^CNXMETAL",
            "NIFTY COMMODITIES": "^CNXCMDT",
            "NIFTY CONSUMPTION": "^CNXCONSUMD",
            "NIFTY CPSE": "^CNXCPSE",
            "NIFTY MEDIA": "^CNXMEDIA",
            "NIFTY PRIVATE BANK": "^NIFTYPVTBANK",
            "NIFTY PSU BANK": "^CNXPSUBANK",
            # Thematic
            "NIFTY MNC": "^CNXMNC",
            "NIFTY SERV SECTOR": "^CNXSERVICE",
            "NIFTY GROWSECT 15": "^NIFTYGROWSECT15",
            "NIFTY100 QUALITY 30": "^NIFTYQUALLV30",
            "NIFTY50 VALUE 20": "^NIFTY50VALUE20",
            "NIFTY DIVIDEND OPPS 50": "^CNXDIVIDEND",
            # Strategy
            "NIFTY ALPHA 50": "^NIFTYALPHA50",
            "NIFTY HIGH BETA 50": "^NIFTYHIGHBETA50",
            "NIFTY LOW VOLATILITY 50": "^NIFTYLOWVOL50",
        }
        
        benchmark_options = list(yahoo_index_map.keys())
        
        # Get stored selection or default
        stored_benchmark = st.session_state.get('benchmark_selection', 'NIFTY 50')
        try:
            default_idx = benchmark_options.index(stored_benchmark)
        except ValueError:
            default_idx = 0
        
        selected_benchmark = st.selectbox(
            "Select Benchmark Index", 
            benchmark_options,
            index=default_idx,
            key="standalone_benchmark_selector"
        )
        st.session_state['benchmark_selection'] = selected_benchmark
        
        try:
            import yfinance as yf
            benchmark_ticker = yahoo_index_map.get(selected_benchmark, "^NSEI")
            benchmark_data = yf.download(benchmark_ticker, start=bt_start, end=bt_end, progress=False)
            
            if not benchmark_data.empty:
                portfolio_values = stored_engine.portfolio_df['Portfolio Value']
                portfolio_norm = (portfolio_values / portfolio_values.iloc[0] - 1) * 100
                
                benchmark_close = benchmark_data['Close']
                if isinstance(benchmark_close, pd.DataFrame):
                    benchmark_close = benchmark_close.iloc[:, 0]
                benchmark_norm = (benchmark_close / benchmark_close.iloc[0] - 1) * 100
                
                # Calculate drawdowns
                portfolio_cummax = portfolio_values.cummax()
                portfolio_dd = ((portfolio_values - portfolio_cummax) / portfolio_cummax) * 100
                benchmark_cummax = benchmark_close.cummax()
                benchmark_dd = ((benchmark_close - benchmark_cummax) / benchmark_cummax) * 100
                
                # PnL Comparison Chart
                fig_pnl = go.Figure()
                fig_pnl.add_trace(go.Scatter(x=portfolio_norm.index, y=portfolio_norm, name="Portfolio", line=dict(color="#28a745", width=2)))
                fig_pnl.add_trace(go.Scatter(x=benchmark_norm.index, y=benchmark_norm, name=selected_benchmark, line=dict(color="#007bff", width=2)))
                fig_pnl.update_layout(title=f"Cumulative Returns: Portfolio vs {selected_benchmark}", xaxis_title="Date", yaxis_title="Return (%)", height=400, template="plotly_dark")
                st.plotly_chart(fig_pnl, use_container_width=True)
                
                # Drawdown Comparison
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(x=portfolio_dd.index, y=portfolio_dd, name="Portfolio DD", line=dict(color="#28a745", width=2), fill='tozeroy', fillcolor='rgba(40, 167, 69, 0.2)'))
                fig_dd.add_trace(go.Scatter(x=benchmark_dd.index, y=benchmark_dd, name=f"{selected_benchmark} DD", line=dict(color="#007bff", width=2), fill='tozeroy', fillcolor='rgba(0, 123, 255, 0.2)'))
                fig_dd.update_layout(title=f"Drawdown Comparison", xaxis_title="Date", yaxis_title="Drawdown (%)", height=400, template="plotly_dark")
                st.plotly_chart(fig_dd, use_container_width=True)
                
                # Summary
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Portfolio Return", f"{portfolio_norm.iloc[-1]:.1f}%")
                col2.metric(f"{selected_benchmark} Return", f"{benchmark_norm.iloc[-1]:.1f}%")
                col3.metric("Portfolio Max DD", f"{portfolio_dd.min():.1f}%")
                col4.metric(f"{selected_benchmark} Max DD", f"{benchmark_dd.min():.1f}%")
                
                alpha = portfolio_norm.iloc[-1] - benchmark_norm.iloc[-1]
                if alpha > 0:
                    st.success(f"🎯 **Alpha Generated: +{alpha:.1f}%**")
                else:
                    st.warning(f"📉 **Alpha: {alpha:.1f}%**")
            else:
                st.warning(f"Could not fetch data for {selected_benchmark}")
        except Exception as e:
            st.error(f"Error loading benchmark: {e}")

# ==================== TAB 2: BACKTEST LOGS ====================
with main_tabs[1]:
    st.subheader("Backtest History")
    
    if not st.session_state.backtest_logs:
        st.info("No backtest logs yet. Run a backtest to see results here.")
    else:
        st.markdown(f"**Total Backtests:** {len(st.session_state.backtest_logs)}")
        
        # Display logs in reverse chronological order
        for idx, log in enumerate(reversed(st.session_state.backtest_logs)):
            with st.expander(f"📊 {log['name']} - {log['timestamp']}"):
                st.markdown(f"**Universe:** {log['config']['universe_name']}")
                st.markdown(f"**Period:** {log['config']['start_date']} to {log['config']['end_date']}")
                st.markdown(f"**Formula:** `{log['config']['formula']}`")
                
                # Key metrics - display vertically to avoid nested columns
                metrics = log['metrics']
                st.markdown(f"**Final Value:** ₹{metrics['Final Value']:,.0f} | **CAGR:** {metrics['CAGR %']:.2f}% | **Sharpe:** {metrics['Sharpe Ratio']:.2f} | **Win Rate:** {metrics['Win Rate %']:.1f}%")
                
                # Show additional log info if available
                if log.get('trades'):
                    st.caption(f"📈 {len(log['trades'])} trades recorded")
                if log.get('portfolio_values'):
                    st.caption(f"📊 {len(log['portfolio_values'])} daily values stored")
                
                # Download buttons - Excel and PDF
                dl_col1, dl_col2 = st.columns(2)
                with dl_col1:
                    excel_data = create_excel_with_charts(log['config'], metrics)
                    st.download_button(
                        label="📥 Download Excel",
                        data=excel_data,
                        file_name=f"{log['name']}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key=f"excel_{idx}"
                    )
                with dl_col2:
                    pdf_data = create_pdf_report(log['config'], metrics)
                    st.download_button(
                        label="📄 Download PDF",
                        data=pdf_data,
                        file_name=f"{log['name']}.pdf",
                        mime="application/pdf",
                        key=f"pdf_{idx}"
                    )
        
        # Clear all logs button
        if st.button("🗑️ Clear All Logs"):
            st.session_state.backtest_logs = []
            st.session_state.backtest_engines = {}
            save_backtest_logs([])  # Save empty list to file
            st.experimental_rerun()

# ==================== TAB 3: DATA DOWNLOAD ====================
with main_tabs[2]:
    st.subheader("📥 Data Download")
    st.markdown("Download historical data for all universes. This is a one-time setup - data will be cached for fast backtests.")

    from portfolio_engine import DataCache
    cache = DataCache()
    cache_info = cache.get_cache_info()

    # Show cache status
    col1, col2, col3 = st.columns([1, 1, 1])
    col1.metric("Cached Stocks", cache_info['total_files'])
    col2.metric("Storage Used", f"{cache_info['total_size_mb']:.2f} MB")
    
    with col3:
        st.write("")  # Spacer
        if st.button("🗑️ Clear All Cache", type="secondary", key="clear_cache"):
            cache.clear()
            st.success("✅ Cache cleared! Please refresh the page.")
            st.rerun()

    st.markdown("---")
    
    # Refresh Universes from NSE
    st.markdown("### 🔄 Refresh Universe Constituents")
    st.info("Fetch live index constituents from NSE India. This updates the stock lists for all universes.")
    
    refresh_col1, refresh_col2 = st.columns([1, 3])
    
    with refresh_col1:
        if st.button("🔄 Refresh Universes", type="secondary", key="refresh_universes"):
            try:
                from nse_fetcher import refresh_universes, load_from_cache
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def progress_callback(pct, msg):
                    progress_bar.progress(pct)
                    status_text.text(msg)
                
                success, message = refresh_universes(progress_callback)
                
                progress_bar.progress(1.0)
                status_text.text("Complete!")
                
                if success:
                    cached, timestamp = load_from_cache()
                    st.success(f"✅ {message}. Cached at: {timestamp}")
                    
                    # Show summary - only for our specified indexes
                    from nifty_universe import INDEX_NAMES
                    with st.expander("Universe Summary"):
                        for name in INDEX_NAMES:
                            if name in cached:
                                st.write(f"**{name}**: {len(cached[name])} stocks")
                            else:
                                st.write(f"**{name}**: Not in cache")
                else:
                    # Show cached data info even if refresh failed
                    cached, timestamp = load_from_cache()
                    if cached:
                        st.warning(f"⚠️ Live refresh failed (NSE blocks cloud servers). Using cached data from: {timestamp} ({len(cached)} universes)")
                    else:
                        st.error(f"❌ {message}")
            except Exception as e:
                # Check if we have cached data to fall back to
                try:
                    from nse_fetcher import load_from_cache
                    cached, timestamp = load_from_cache()
                    if cached:
                        st.warning(f"⚠️ NSE blocks cloud requests. Using pre-loaded cache: {timestamp} ({len(cached)} universes)")
                    else:
                        st.error(f"❌ Error refreshing: {e}. Run locally: python nse_fetcher.py")
                except:
                    st.error(f"❌ Error: {e}")
    
    with refresh_col2:
        # Show current cache status
        try:
            from nifty_universe import INDEX_NAMES
            from nse_fetcher import load_from_cache
            cached, timestamp = load_from_cache()
            st.write(f"📊 Active indexes: **{len(INDEX_NAMES)}**")
            if cached and timestamp:
                st.write(f"📅 Cache updated: **{timestamp}**")
        except:
            st.write("⚠️ Universe data not initialized.")
    
    st.markdown("---")

    # Download All Data Button
    st.markdown("### 🔽 Download All Universe Data")
    st.info("This will download and cache data for ALL stocks across ALL universes. Takes ~10-15 minutes.")
    
    # Clear cache option
    col_clear, col_download = st.columns(2)
    with col_clear:
        if st.button("🗑️ Clear Cache First", key="clear_data_cache_btn"):
            from portfolio_engine import DataCache
            cache = DataCache()
            cache.clear()
            st.success("✅ Cache cleared! Now click 'Download All Data' to get fresh data.")
            st.rerun()
    
    with col_download:
        download_clicked = st.button("📥 Download All Data", type="primary", key="download_all_data_btn")

    if download_clicked:
        # Get all unique tickers from all universes
        all_tickers = set()
        all_universe_names = get_all_universe_names()

        for universe_name in all_universe_names:
            universe = get_universe(universe_name)
            all_tickers.update(universe)

        all_tickers = sorted(list(all_tickers))

        st.markdown(f"### Downloading {len(all_tickers)} unique stocks...")

        # Progress display
        progress_bar = st.progress(0)
        status_text = st.empty()

        start_time = time.time()

        def download_progress(current, total, ticker, remaining_seconds):
            pct = (current / total) if total > 0 else 0
            progress_bar.progress(min(pct, 1.0))

            mins = int(remaining_seconds // 60)
            secs = int(remaining_seconds % 60)
            elapsed = time.time() - start_time
            elapsed_mins = int(elapsed // 60)
            elapsed_secs = int(elapsed % 60)

            status_text.markdown(f"""
            <div style="padding: 10px; background: rgba(0,255,136,0.1); border-radius: 5px;">
                <div style="font-size: 16px; font-weight: bold;">📊 {ticker}</div>
                <div>Progress: {current}/{total} ({pct*100:.1f}%)</div>
                <div>⏱️ Remaining: {mins:02d}:{secs:02d} | ⏰ Elapsed: {elapsed_mins:02d}:{elapsed_secs:02d}</div>
            </div>
            """, unsafe_allow_html=True)

        temp_engine = PortfolioEngine(all_tickers, datetime.date(2020, 1, 1), datetime.date.today())
        success_count = temp_engine.download_and_cache_universe(all_tickers, download_progress, None)

        progress_bar.empty()
        status_text.empty()

        total_time = time.time() - start_time
        st.success(f"✅ Downloaded {success_count}/{len(all_tickers)} stocks in {int(total_time)}s!")
        st.balloons()


