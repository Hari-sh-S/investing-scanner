---
title: Investing Scanner
emoji: 📊
colorFrom: red
colorTo: blue
sdk: streamlit
sdk_version: "1.41.1"
app_file: app.py
pinned: true
---

# 📊 Investing Scanner

An advanced backtesting platform for the Indian Stock Market, built with Streamlit. Analyze portfolios, test strategies, and optimize your trading approach using real NSE data.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

## 🚀 Features

### Core Functionality
- **Advanced Backtesting Engine** - Run comprehensive portfolio backtests with historical NSE data
- **Multiple Universe Support** - Test across Nifty50, Nifty500, sectoral indices, and custom stock lists
- **Flexible Rebalancing** - Weekly or monthly rebalancing with customizable dates
- **Regime Filters** - EMA, MACD, SuperTrend, and Equity-based filters to adapt to market conditions
- **Uncorrelated Assets** - Allocate to Gold, Bonds, or other assets for diversification
- **Exit Rank Strategy** - Dynamic position sizing based on stock rankings

### Performance Analytics
- CAGR, Sharpe Ratio, Max Drawdown, Win Rate
- Monthly returns heatmap
- Equity curve and drawdown charts
- Detailed trade history and portfolio reports

### Data Management
- **Parquet-based caching** - Lightning-fast data retrieval
- **NSE Integration** - Live stock list updates from NSE India
- **Persistent backtest logs** - Save and compare multiple strategies

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/investing-scanner.git
cd investing-scanner
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

## 🌐 Live Demo

Visit the live application: [Investing Scanner on Streamlit Cloud](https://YOUR_APP_URL.streamlit.app)

## 🎯 Quick Start

### Running Your First Backtest

1. **Select a Universe** - Choose from Nifty50, Nifty500, or any sectoral index
2. **Configure Portfolio** - Set initial capital, number of stocks, and rebalancing frequency
3. **Define Strategy** - Use the scoring console to create your ranking formula
   - Example: `6 Month Performance` ranks stocks by 6-month returns
4. **Run Backtest** - Click "🚀 Run Backtest" and analyze results

### Example Strategies

**Momentum Strategy**
```
6 Month Performance
```

**Quality + Momentum**
```
(6 Month Performance + 6 Month Sharpe) / 2
```

**Multi-Factor**
```
(3 Month Performance * 0.3) + (6 Month Performance * 0.4) + (6 Month Sharpe * 0.3)
```

## 📁 Project Structure

```
investing-scanner/
├── app.py                    # Main Streamlit application
├── engine.py                 # Core backtesting logic
├── portfolio_engine.py       # Portfolio management and rebalancing
├── scoring.py               # Strategy scoring and validation
├── indicators.py            # Technical indicators calculation
├── nse_fetcher.py          # NSE data fetching utilities
├── nifty_universe.py       # Stock universe definitions
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🛠️ Technologies Used

- **[Streamlit](https://streamlit.io/)** - Web application framework
- **[yfinance](https://github.com/ranaroussi/yfinance)** - Yahoo Finance data fetcher
- **[Pandas](https://pandas.pydata.org/)** - Data manipulation
- **[Plotly](https://plotly.com/)** - Interactive charts
- **[TA-Lib](https://github.com/mrjbq7/ta-lib)** - Technical analysis indicators
- **PyArrow** - Parquet file support

## 📊 Supported Universes

### Broad Market
- Nifty 50, Nifty 100, Nifty 200, Nifty 500
- Nifty Midcap 50/100/150
- Nifty Smallcap 50/100/250

### Sectoral
- Banking, Auto, Pharma, IT, FMCG, Metal
- Energy, Realty, Media, Infra, PSU Bank
- And 15+ more sectors

### Thematic
- Dividend Opportunities, Growth Sectors, Consumption
- Digital India, EV, Healthcare, And more

## 🔧 Configuration

### Regime Filters
Configure market regime detection to adjust portfolio allocation:
- **EMA** - Exit when price < EMA (34/68/100/150/200)
- **MACD** - Exit on bearish MACD crossover
- **SuperTrend** - Exit when SuperTrend turns bearish
- **Equity** - Reduce allocation on portfolio drawdown

### Rebalancing Options
- **Weekly** - Choose day of week (Monday-Friday)
- **Monthly** - Choose date (1-30)
- **Alternative Day** - Handle holidays (Previous/Next day)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is for educational purposes. Please ensure compliance with data provider terms of service.

## 🙏 Acknowledgments

- Built for the Indian stock market community
- Data sourced from Yahoo Finance and NSE India

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

⭐ **Star this repository** if you find it useful!
