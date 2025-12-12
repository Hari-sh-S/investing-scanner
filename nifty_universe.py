# NIFTY Universe Data - 55 NSE Indexes
# Last Updated: 2025-12-13
# Source: NSE official indices - using exact API names

# Base stock lists - will be dynamically fetched from NSE cache
# These are placeholders that get populated from nse_universe_cache.json

# All 55 indexes using exact NSE API names
INDEX_NAMES = [
    "NIFTY 50",
    "NIFTY NEXT 50",
    "NIFTY BANK",
    "NIFTY FIN SERVICE",      # NSE API name for Financial Services
    "NIFTY MID SELECT",       # NSE API name for Midcap Select
    "NIFTY 100",
    "NIFTY 200",
    "NIFTY 500",
    "NIFTY MIDCAP 50",
    "NIFTY MIDCAP 100",
    "NIFTY MIDCAP 150",
    "NIFTY SMLCAP 100",       # NSE API name for Smallcap 100
    "NIFTY SMLCAP 50",        # NSE API name for Smallcap 50
    "NIFTY SMLCAP 250",       # NSE API name for Smallcap 250
    "NIFTY MIDSML 400",       # NSE API name for Midsmallcap 400
    "NIFTY500 MULTICAP",      # NSE API name for Multicap 50:25:25
    "NIFTY LARGEMID250",      # NSE API name for Largemidcap 250
    "NIFTY TOTAL MKT",        # NSE API name for Total Market
    "NIFTY MICROCAP250",      # NSE API name for Microcap 250
    "NIFTY500 LMS EQL",       # NSE API name for LMS Equal-Cap
    "NIFTY FPI 150",          # NSE API name for India FPI 150
    "NIFTY ALPHA 50",
    "NIFTY50 EQL WGT",        # NSE API name for Nifty50 Equal Weight
    "NIFTY100 EQL WGT",       # NSE API name for Nifty100 Equal Weight
    "NIFTY100 LOWVOL30",      # NSE API name for Low Volatility 30
    "NIFTY200 QUALTY30",      # NSE API name for Quality 30
    "NIFTY ALPHALOWVOL",      # NSE API name for Alpha Low-Vol 30
    "NIFTY200MOMENTM30",      # NSE API name for Momentum 30
    "NIFTY M150 QLTY50",      # NSE API name for Midcap150 Quality 50
    "NIFTY200 ALPHA 30",
    "NIFTYM150MOMNTM50",      # NSE API name for Midcap150 Momentum 50
    "NIFTY500MOMENTM50",      # NSE API name for Nifty500 Momentum 50
    "NIFTYMS400 MQ 100",      # NSE API name for Midsmall400 MQ 100
    "NIFTYSML250MQ 100",      # NSE API name for Smallcap250 MQ 100
    "NIFTY TOP 10 EW",        # NSE API name for Top 10 EW
    "NIFTY AQL 30",           # NSE API name for Alpha Quality LV 30
    "NIFTY AQLV 30",          # NSE API name for Alpha Quality Value LV 30
    "NIFTY HIGHBETA 50",      # NSE API name for High Beta 50
    "NIFTY LOW VOL 50",       # NSE API name for Low Volatility 50
    "NIFTY QLTY LV 30",       # NSE API name for Quality Low-Vol 30
    "NIFTY SML250 Q50",       # NSE API name for Smallcap250 Quality 50
    "NIFTY TOP 15 EW",        # NSE API name for Top 15 EW
    "NIFTY100 ALPHA 30",
    "NIFTY200 VALUE 30",
    "NIFTY500 EW",            # NSE API name for Nifty500 Equal Weight
    "NIFTY MULTI MQ 50",      # NSE API name for Multicap MQ 50
    "NIFTY500 VALUE 50",
    "NIFTY TOP 20 EW",        # NSE API name for Top 20 EW
    "NIFTY500 QLTY50",        # NSE API name for Nifty500 Quality 50
    "NIFTY500 LOWVOL50",      # NSE API name for Nifty500 Low Vol 50
    "NIFTY500 MQVLV50",       # NSE API name for Multifactor MQVLV
    "NIFTY500 FLEXICAP",      # NSE API name for Flexicap Quality 30
    "NIFTY TMMQ 50",          # NSE API name for Total Market MQ 50
]

# Fallback stock lists for key indexes (used when NSE cache is unavailable)
NIFTY_50 = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", 
    "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
    "LT", "AXISBANK", "ASIANPAINT", "MARUTI", "HCLTECH",
    "WIPRO", "ULTRACEMCO", "TITAN", "BAJFINANCE", "SUNPHARMA",
    "NESTLEIND", "TECHM", "ONGC", "NTPC", "POWERGRID",
    "M&M", "TATAMOTORS", "TATASTEEL", "ADANIPORTS", "COALINDIA",
    "BAJAJFINSV", "INDUSINDBK", "HINDALCO", "JSWSTEEL", "GRASIM",
    "DIVISLAB", "DRREDDY", "CIPLA", "APOLLOHOSP", "EICHERMOT",
    "HEROMOTOCO", "BRITANNIA", "TATACONSUM", "BAJAJ-AUTO", "SHRIRAMFIN",
    "ADANIENT", "SBILIFE", "HDFCLIFE", "BPCL", "IOC"
]

NIFTY_NEXT_50 = [
    "ADANIGREEN", "ADANIPOWER", "ADANITRANS", "AMBUJACEM", "ATGL",
    "BERGEPAINT", "BEL", "BOSCHLTD", "CANBK", "CHOLAFIN",
    "COLPAL", "DABUR", "DLF", "DMART", "GAIL",
    "GODREJCP", "HAL", "HAVELLS", "ICICIPRULI", "INDIGO",
    "JINDALSTEL", "LTIM", "MARICO", "MOTHERSON", "MPHASIS",
    "NAUKRI", "NMDC", "PAYTM", "PIDILITIND", "PNB",
    "RECLTD", "SAIL", "SBICARD", "SIEMENS", "TATAPOWER",
    "TATATECH", "TORNTPHARM", "TRENT", "TVSMOTOR", "UBL",
    "VEDL", "VOLTAS", "ZOMATO", "ZYDUSLIFE", "ICICIGI",
    "PGHH", "PETRONET", "PFC", "INDHOTEL", "BAJAJHLDNG"
]

NIFTY_BANK = [
    "HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK",
    "INDUSINDBK", "BANDHANBNK", "FEDERALBNK", "PNB", "IDFCFIRSTB",
    "AUBANK", "BANKBARODA"
]

NIFTY_100 = list(dict.fromkeys(NIFTY_50 + NIFTY_NEXT_50))

# Universe dictionary - all 60 indexes
UNIVERSES = {name: [] for name in INDEX_NAMES}

# Populate key fallback lists
UNIVERSES["NIFTY 50"] = NIFTY_50
UNIVERSES["NIFTY NEXT 50"] = NIFTY_NEXT_50
UNIVERSES["NIFTY BANK"] = NIFTY_BANK
UNIVERSES["NIFTY 100"] = NIFTY_100
UNIVERSES["NIFTY FINANCIAL SERVICES"] = list(dict.fromkeys(NIFTY_BANK + ["BAJFINANCE", "BAJAJFINSV", "SBILIFE", "HDFCLIFE", "ICICIGI", "SHRIRAMFIN"]))
UNIVERSES["NIFTY MIDCAP SELECT"] = NIFTY_NEXT_50[:25]
UNIVERSES["NIFTY 200"] = NIFTY_100
UNIVERSES["NIFTY 500"] = NIFTY_100
UNIVERSES["NIFTY MIDCAP 50"] = NIFTY_NEXT_50[:50]
UNIVERSES["NIFTY MIDCAP 100"] = NIFTY_NEXT_50
UNIVERSES["NIFTY MIDCAP 150"] = NIFTY_NEXT_50
UNIVERSES["NIFTY SMALLCAP 100"] = NIFTY_NEXT_50
UNIVERSES["NIFTY SMALLCAP 50"] = NIFTY_NEXT_50[:50]
UNIVERSES["NIFTY SMALLCAP 250"] = NIFTY_100
UNIVERSES["NIFTY MIDSMALLCAP 400"] = NIFTY_100
UNIVERSES["NIFTY500 MULTICAP 50:25:25"] = NIFTY_100
UNIVERSES["NIFTY LARGEMIDCAP 250"] = NIFTY_100
UNIVERSES["NIFTY TOTAL MARKET"] = NIFTY_100
UNIVERSES["NIFTY MICROCAP 250"] = NIFTY_NEXT_50
UNIVERSES["NIFTY500 LARGEMIDSMALL EQUAL-CAP WEIGHTED"] = NIFTY_100
UNIVERSES["NIFTY INDIA FPI 150"] = NIFTY_100
UNIVERSES["NIFTY50 DIVIDEND POINTS"] = NIFTY_50[:20]
UNIVERSES["NIFTY ALPHA 50"] = NIFTY_50
UNIVERSES["NIFTY50 EQUAL WEIGHT"] = NIFTY_50
UNIVERSES["NIFTY100 EQUAL WEIGHT"] = NIFTY_100
UNIVERSES["NIFTY100 LOW VOLATILITY 30"] = NIFTY_50[:30]
UNIVERSES["NIFTY200 QUALITY 30"] = NIFTY_50[:30]
UNIVERSES["NIFTY ALPHA LOW-VOLATILITY 30"] = NIFTY_50[:30]
UNIVERSES["NIFTY200 MOMENTUM 30"] = NIFTY_50[:30]
UNIVERSES["NIFTY MIDCAP150 QUALITY 50"] = NIFTY_NEXT_50[:50]
UNIVERSES["NIFTY200 ALPHA 30"] = NIFTY_50[:30]
UNIVERSES["NIFTY MIDCAP150 MOMENTUM 50"] = NIFTY_NEXT_50[:50]
UNIVERSES["NIFTY500 MOMENTUM 50"] = NIFTY_50
UNIVERSES["NIFTY MIDSMALLCAP400 MOMENTUM QUALITY 100"] = NIFTY_100
UNIVERSES["NIFTY SMALLCAP250 MOMENTUM QUALITY 100"] = NIFTY_100
UNIVERSES["NIFTY TOP 10 EQUAL WEIGHT"] = NIFTY_50[:10]
UNIVERSES["NIFTY ALPHA QUALITY LOW-VOLATILITY 30"] = NIFTY_50[:30]
UNIVERSES["NIFTY ALPHA QUALITY VALUE LOW-VOLATILITY 30"] = NIFTY_50[:30]
UNIVERSES["NIFTY HIGH BETA 50"] = NIFTY_50
UNIVERSES["NIFTY LOW VOLATILITY 50"] = NIFTY_50
UNIVERSES["NIFTY QUALITY LOW-VOLATILITY 30"] = NIFTY_50[:30]
UNIVERSES["NIFTY SMALLCAP250 QUALITY 50"] = NIFTY_NEXT_50[:50]
UNIVERSES["NIFTY TOP 15 EQUAL WEIGHT"] = NIFTY_50[:15]
UNIVERSES["NIFTY100 ALPHA 30"] = NIFTY_50[:30]
UNIVERSES["NIFTY200 VALUE 30"] = NIFTY_50[:30]
UNIVERSES["NIFTY500 EQUAL WEIGHT"] = NIFTY_100
UNIVERSES["NIFTY500 MULTICAP MOMENTUM QUALITY 50"] = NIFTY_50
UNIVERSES["NIFTY500 VALUE 50"] = NIFTY_50
UNIVERSES["NIFTY TOP 20 EQUAL WEIGHT"] = NIFTY_50[:20]
UNIVERSES["NIFTY500 QUALITY 50"] = NIFTY_50
UNIVERSES["NIFTY500 LOW VOLATILITY 50"] = NIFTY_50
UNIVERSES["NIFTY500 MULTIFACTOR MQVLV 50"] = NIFTY_50
UNIVERSES["NIFTY50 USD"] = NIFTY_50
UNIVERSES["NIFTY500 FLEXICAP QUALITY 30"] = NIFTY_50[:30]
UNIVERSES["NIFTY TOTAL MARKET MOMENTUM QUALITY 50"] = NIFTY_50


def _load_nse_cache():
    """Load NSE cache if available."""
    try:
        from pathlib import Path
        import json
        cache_file = Path("nse_universe_cache.json")
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                data = json.load(f)
                return data.get('universes', {})
    except:
        pass
    return {}


def get_universe(name):
    """Get stock list for a given universe name. Uses NSE cache if available."""
    # First try NSE cache
    cached = _load_nse_cache()
    if name in cached and cached[name]:
        return cached[name]
    
    # Fallback to hardcoded
    return UNIVERSES.get(name, [])


def get_all_universe_names():
    """Get all available universe names in the specified order."""
    return INDEX_NAMES.copy()


def get_broad_market_universes():
    """Get broad market indices."""
    return [
        "NIFTY 50", "NIFTY NEXT 50", "NIFTY BANK", "NIFTY FINANCIAL SERVICES",
        "NIFTY MIDCAP SELECT", "NIFTY 100", "NIFTY 200", "NIFTY 500"
    ]


def get_sectoral_universes():
    """Get sectoral/thematic indices (now just a subset of key indexes)."""
    return [
        "NIFTY BANK", "NIFTY FINANCIAL SERVICES", "NIFTY MIDCAP SELECT"
    ]


def get_cap_based_universes():
    """Get cap-based indices."""
    return [
        "NIFTY MIDCAP 50", "NIFTY MIDCAP 100", "NIFTY MIDCAP 150",
        "NIFTY SMALLCAP 50", "NIFTY SMALLCAP 100", "NIFTY SMALLCAP 250",
        "NIFTY MIDSMALLCAP 400", "NIFTY MICROCAP 250"
    ]


def get_thematic_universes():
    """Get strategy/factor indices."""
    return [
        "NIFTY ALPHA 50", "NIFTY HIGH BETA 50", "NIFTY LOW VOLATILITY 50",
        "NIFTY500 MOMENTUM 50", "NIFTY500 QUALITY 50", "NIFTY500 VALUE 50"
    ]
