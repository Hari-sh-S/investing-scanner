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
import datetime
import io
import time
import json
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Investing Scanner",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state for backtest logs
BACKTEST_LOG_FILE = Path("backtest_logs.json")

def load_backtest_logs():
    """Load backtest logs from file."""
    if BACKTEST_LOG_FILE.exists():
        try:
            with open(BACKTEST_LOG_FILE, 'r') as f:
                logs_data = json.load(f)
                # Convert back to proper format (engine objects can't be serialized, so we skip them)
                return logs_data
        except Exception as e:
            print(f"Error loading logs: {e}")
            return []
    return []

def save_backtest_logs(logs):
    """Save backtest logs to file."""
    try:
        # Serialize logs (skip engine objects)
        serializable_logs = []
        for log in logs:
            serializable_log = {
                'timestamp': log['timestamp'],
                'name': log['name'],
                'config': log['config'],
                'metrics': log['metrics']
                # Note: 'engine' object is not serializable, will be recreated when needed
            }
            serializable_logs.append(serializable_log)
        
        with open(BACKTEST_LOG_FILE, 'w') as f:
            json.dump(serializable_logs, f, indent=2)
    except Exception as e:
        print(f"Error saving logs: {e}")

# Load logs on startup
if 'backtest_logs' not in st.session_state:
    st.session_state.backtest_logs = load_backtest_logs()
if 'backtest_engines' not in st.session_state:
    st.session_state.backtest_engines = {}  # Store engines separately

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

# Helper function to create Excel download
def create_excel_download(config, metrics, engine=None):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Inputs Sheet
        inputs_data = {
            'Parameter': ['Strategy Name', 'Starting Capital', 'Universe',
                        'No Of Stocks in Portfolio', 'Exit Rank',
                        'Rebalance Frequency', 'Start Date', 'End Date',
                        'Regime Filter', 'Strategy'],
            'Value': [
                config.get('name', 'Backtest'),
                config['initial_capital'],
                config['universe_name'],
                config['num_stocks'],
                config.get('exit_rank', config['num_stocks']),
                config['rebalance_freq'],
                config['start_date'],
                config['end_date'],
                str(config.get('regime_config', {})),
                config['formula']
            ]
        }
        pd.DataFrame(inputs_data).to_excel(writer, sheet_name='Inputs', index=False)

        # Performance Metrics Sheet
        perf_data = {
            'Metric': ['Start Date', 'End Date', 'Invested Capital', 'Current Capital',
                     'Win Rate(%)', 'Max. DD(%)', 'CAGR(%)',
                     'Sharpe Ratio', 'Total Trades'],
            'Score': [
                config['start_date'],
                config['end_date'],
                config['initial_capital'],
                metrics['Final Value'],
                metrics['Win Rate %'],
                metrics['Max Drawdown %'],
                metrics['CAGR %'],
                metrics['Sharpe Ratio'],
                metrics['Total Trades']
            ]
        }
        pd.DataFrame(perf_data).to_excel(writer, sheet_name='Performance Metrics', index=False)

        # Monthly Returns Table
        if engine and hasattr(engine, 'portfolio_df'):
            if not engine.portfolio_df.empty:
                monthly_returns = engine.get_monthly_returns()
                if not monthly_returns.empty:
                    monthly_returns.to_excel(writer, sheet_name='Monthly Returns')

        # Daily Portfolio Report and Trade History - only if engine is available
        if engine and hasattr(engine, 'portfolio_df'):
            if not engine.portfolio_df.empty:
                daily_data = engine.portfolio_df.copy()
                daily_data['Year'] = daily_data.index.year
                daily_data['Month'] = daily_data.index.month
                daily_data['Day'] = daily_data.index.day
                daily_data[['Year', 'Month', 'Day', 'Portfolio Value', 'Cash', 'Positions']].to_excel(
                    writer, sheet_name='Daily Report', index=False
                )

            # Trade History
            if hasattr(engine, 'trades_df') and not engine.trades_df.empty:
                engine.trades_df.to_excel(writer, sheet_name='Trade History', index=False)

    output.seek(0)
    return output

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
            rebalance_date = st.number_input("Rebalance Date (1-30)", 1, 30, 15,
                                            help="Day of month to rebalance portfolio")
            rebal_day = None
        
        alt_day_option = st.selectbox("Alternative Rebalance Day", 
                                     ["Previous Day", "Next Day"],
                                     help="If rebalance day is holiday, use this option")
        
        st.markdown("**Regime Filter**")
        use_regime_filter = st.checkbox("Enable Regime Filter", value=False)
        
        regime_config = None
        if use_regime_filter:
            regime_type = st.selectbox("Regime Filter Type", 
                                      ["EMA", "MACD", "SUPERTREND", "EQUITY"])
            
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
            else:  # EQUITY
                realized_sl = st.number_input("Realized PnL SL %", 1, 50, 10)
                regime_value = realized_sl
            
            # Regime action (only for non-EQUITY)
            if regime_type != "EQUITY":
                regime_action = st.selectbox("Regime Filter Action",
                                            ["Half Portfolio", "Go Cash"])

                # Index selection for regime filter - use all available universes
                regime_index = st.selectbox("Regime Filter Index", sorted(get_all_universe_names()))
            else:
                regime_action = "Half Portfolio"  # Fixed for EQUITY
                regime_index = None
            
            regime_config = {
                'type': regime_type,
                'value': regime_value,
                'action': regime_action,
                'index': regime_index
            }
        
        st.markdown("**Uncorrelated Asset**")
        use_uncorrelated = st.checkbox("Invest in Uncorrelated Asset", value=False)
        
        uncorrelated_config = None
        if use_uncorrelated:
            asset_type = st.text_input("Asset Type", "GOLDBEES",
                                      help="Enter ticker symbol (e.g., GOLDBEES for Gold)")
            allocation_pct = st.number_input("Allocation %", 1, 100, 20,
                                            help="% of portfolio value to allocate")
            
            uncorrelated_config = {
                'asset': asset_type,
                'allocation_pct': allocation_pct
            }
    
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
                    
                    if metrics:
                        # Save to logs
                        backtest_log = {
                            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'name': f"Backtest_{datetime.datetime.now().strftime('%m%d_%H%M')}",
                            'config': {
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
                            'metrics': metrics,
                        }
                        st.session_state.backtest_logs.append(backtest_log)
                        save_backtest_logs(st.session_state.backtest_logs)  # Save to file
                        
                        st.markdown("---")
                        
                        # Action buttons
                        col_h, col_download = st.columns([4, 1])
                        with col_h:
                            st.subheader("Backtest Results")
                        with col_download:
                            excel_data = create_excel_download(backtest_log['config'], metrics, engine)
                            st.download_button(
                                label="📥 Excel",
                                data=excel_data,
                                file_name=f"backtest_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        
                        # Result tabs
                        result_tabs = st.tabs(["Performance Metrics", "Charts", "Monthly Breakup", "Monthly Report", "Trade History"])
                        
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
                                trade_display = engine.trades_df.copy()
                                trade_display['Date'] = pd.to_datetime(trade_display['Date']).dt.strftime('%Y-%m-%d')
                                st.dataframe(trade_display, use_container_width=True, height=400)
                            else:
                                st.info("No trades executed")
                    else:
                        st.warning("No trades generated")
                else:
                    st.error("Data fetch failed")

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
                
                # Download button
                excel_data = create_excel_download(log['config'], metrics)
                st.download_button(
                    label="📥 Download Excel",
                    data=excel_data,
                    file_name=f"{log['name']}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"download_{idx}"
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

    # Download All Data Button
    st.markdown("### 🔽 Download All Universe Data")
    st.info("This will download and cache data for ALL stocks across ALL universes. Takes ~10-15 minutes.")

    if st.button("📥 Download All Data", type="primary", key="download_all"):
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

    st.markdown("---")

    # Refresh Stock Lists from NSE
    st.markdown("### 🔄 Refresh Stock Lists from NSE")
    st.info("Update universe stock lists with latest constituents from NSE India.")

    if st.button("🔄 Refresh from NSE", type="secondary"):
        with st.spinner("Fetching live data from NSE India..."):
            try:
                from nse_fetcher import update_universe_file_inplace

                results = update_universe_file_inplace("nifty_universe.py")

                st.success(f"✅ Updated {len(results)} indices from NSE!")

                # Show summary
                for idx_name, stocks in results.items():
                    st.caption(f"✓ {idx_name}: {len(stocks)} stocks")

                st.warning("⚠️ Restart the app to use updated stock lists")
                st.balloons()

            except Exception as e:
                st.error(f"❌ Error: {e}")

