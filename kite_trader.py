"""
Kite Connect Integration for executing real trades on Zerodha.
Handles authentication flow and order placement with Inverse Volatility position sizing.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== TOKEN PERSISTENCE ====================

def _get_hf_credentials():
    """Get HF credentials from Streamlit secrets for token storage."""
    try:
        token = st.secrets.get("HF_TOKEN")
        repo = st.secrets.get("HF_DATASET_REPO")
        if token and repo:
            return token, repo
    except Exception:
        pass
    return None, None


def save_kite_token_to_hf(access_token: str, user_id: str, user_name: str) -> bool:
    """
    Save Kite access token to HuggingFace for persistence across sessions.
    
    Args:
        access_token: The Kite access token
        user_id: The user's Kite user ID
        user_name: The user's name
        
    Returns:
        True if saved successfully
    """
    from datetime import datetime
    import json
    import tempfile
    from pathlib import Path
    
    hf_token, repo = _get_hf_credentials()
    if not hf_token or not repo:
        logger.warning("HuggingFace not configured, token will not persist")
        return False
    
    try:
        from huggingface_hub import upload_file
        
        token_data = {
            'access_token': access_token,
            'user_id': user_id,
            'user_name': user_name,
            'saved_at': datetime.now().isoformat()
        }
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(token_data, f, indent=2)
            tmp_path = f.name
        
        # Upload to HF
        upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo="kite_token.json",
            repo_id=repo,
            repo_type="dataset",
            token=hf_token,
            commit_message="Save Kite access token"
        )
        
        # Cleanup
        Path(tmp_path).unlink(missing_ok=True)
        
        logger.info(f"Kite token saved to HuggingFace for user: {user_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving Kite token to HF: {e}")
        return False


def load_kite_token_from_hf() -> Optional[Dict]:
    """
    Load saved Kite token from HuggingFace.
    
    Returns:
        Dict with token data or None if not found
    """
    import json
    
    hf_token, repo = _get_hf_credentials()
    if not hf_token or not repo:
        return None
    
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError
        
        file_path = hf_hub_download(
            repo_id=repo,
            filename="kite_token.json",
            repo_type="dataset",
            token=hf_token
        )
        
        with open(file_path, 'r') as f:
            token_data = json.load(f)
        
        logger.info(f"Kite token loaded from HuggingFace for user: {token_data.get('user_id', 'unknown')}")
        return token_data
        
    except Exception as e:
        # EntryNotFoundError is expected if no token saved yet
        if "EntryNotFoundError" not in str(type(e)):
            logger.debug(f"No saved Kite token found: {e}")
        return None


def validate_kite_token(access_token: str) -> bool:
    """
    Validate if a Kite access token is still valid by making an API call.
    
    Args:
        access_token: The token to validate
        
    Returns:
        True if token is valid, False otherwise
    """
    try:
        from kiteconnect import KiteConnect
        
        api_key, _ = get_kite_credentials()
        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)
        
        # Try to fetch profile - this will fail if token is invalid
        profile = kite.profile()
        logger.info(f"Kite token validated for user: {profile.get('user_id', 'unknown')}")
        return True
        
    except Exception as e:
        logger.info(f"Kite token validation failed: {e}")
        return False


def clear_kite_token_from_hf() -> bool:
    """
    Delete saved Kite token from HuggingFace.
    
    Returns:
        True if deleted successfully
    """
    hf_token, repo = _get_hf_credentials()
    if not hf_token or not repo:
        return False
    
    try:
        from huggingface_hub import HfApi
        
        api = HfApi(token=hf_token)
        api.delete_file(
            path_in_repo="kite_token.json",
            repo_id=repo,
            repo_type="dataset",
            commit_message="Clear Kite access token"
        )
        
        logger.info("Kite token cleared from HuggingFace")
        return True
        
    except Exception as e:
        logger.debug(f"Error clearing Kite token from HF: {e}")
        return False


def restore_kite_session() -> bool:
    """
    Attempt to restore Kite session from saved HuggingFace token.
    Validates the token before restoring.
    
    Returns:
        True if session restored successfully, False otherwise
    """
    # Skip if already authenticated
    if st.session_state.get('kite_access_token'):
        return True
    
    # Try to load saved token
    token_data = load_kite_token_from_hf()
    if not token_data:
        return False
    
    access_token = token_data.get('access_token')
    if not access_token:
        return False
    
    # Validate the token
    if validate_kite_token(access_token):
        # Restore session state
        st.session_state.kite_access_token = access_token
        st.session_state.kite_user_id = token_data.get('user_id', '')
        st.session_state.kite_user_name = token_data.get('user_name', '')
        logger.info(f"Kite session restored for user: {st.session_state.kite_user_id}")
        return True
    else:
        # Token is invalid/expired, clear it
        clear_kite_token_from_hf()
        return False

# ==================== END TOKEN PERSISTENCE ====================


def is_kite_configured() -> bool:
    """Check if Kite API credentials are configured in Streamlit secrets."""
    try:
        api_key = st.secrets.get("KITE_API_KEY")
        api_secret = st.secrets.get("KITE_API_SECRET")
        return bool(api_key and api_secret)
    except Exception:
        return False


def get_kite_credentials() -> Tuple[str, str]:
    """Get Kite API credentials from Streamlit secrets."""
    api_key = st.secrets["KITE_API_KEY"]
    api_secret = st.secrets["KITE_API_SECRET"]
    return api_key, api_secret


def get_kite_client():
    """Get authenticated Kite client from session state."""
    try:
        from kiteconnect import KiteConnect
        
        if 'kite_access_token' not in st.session_state or not st.session_state.kite_access_token:
            return None
            
        api_key, _ = get_kite_credentials()
        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(st.session_state.kite_access_token)
        return kite
    except Exception as e:
        logger.error(f"Error getting Kite client: {e}")
        return None


def get_login_url() -> str:
    """Generate Kite login URL."""
    try:
        from kiteconnect import KiteConnect
        api_key, _ = get_kite_credentials()
        kite = KiteConnect(api_key=api_key)
        return kite.login_url()
    except Exception as e:
        logger.error(f"Error generating login URL: {e}")
        return ""


def handle_kite_callback(request_token: str) -> bool:
    """Exchange request_token for access_token after OAuth callback."""
    try:
        from kiteconnect import KiteConnect
        
        api_key, api_secret = get_kite_credentials()
        kite = KiteConnect(api_key=api_key)
        
        # Generate session with request token
        data = kite.generate_session(request_token, api_secret=api_secret)
        access_token = data["access_token"]
        user_id = data.get("user_id", "")
        user_name = data.get("user_name", "")
        
        # Store in session state
        st.session_state.kite_access_token = access_token
        st.session_state.kite_user_id = user_id
        st.session_state.kite_user_name = user_name
        
        # Save token to HuggingFace for persistence
        save_kite_token_to_hf(access_token, user_id, user_name)
        
        logger.info(f"Kite session generated for user: {user_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error handling Kite callback: {e}")
        st.session_state.kite_access_token = None
        return False


def calculate_inverse_volatility_weights(
    tickers: List[str], 
    data: Dict[str, pd.DataFrame],
    lookback_days: int = 60
) -> Dict[str, float]:
    """
    Calculate inverse volatility weights for position sizing.
    
    Higher weight for stocks with lower volatility.
    """
    volatilities = {}
    
    for ticker in tickers:
        if ticker not in data or data[ticker].empty:
            continue
            
        df = data[ticker]
        if len(df) < lookback_days:
            continue
            
        # Calculate daily returns volatility (annualized)
        if 'Close' in df.columns:
            close = df['Close'].tail(lookback_days)
        else:
            continue
            
        returns = close.pct_change().dropna()
        if len(returns) < 10:
            continue
            
        vol = returns.std() * np.sqrt(252)  # Annualized volatility
        if vol > 0:
            volatilities[ticker] = vol
    
    if not volatilities:
        # Equal weight fallback if no volatility data
        return {ticker: 1.0 / len(tickers) for ticker in tickers}
    
    # Inverse volatility weighting
    inverse_vols = {ticker: 1.0 / vol for ticker, vol in volatilities.items()}
    total_inverse = sum(inverse_vols.values())
    
    weights = {ticker: inv_vol / total_inverse for ticker, inv_vol in inverse_vols.items()}
    
    return weights


def calculate_order_quantities(
    open_positions: List[Dict],
    capital: float,
    data: Dict[str, pd.DataFrame],
    max_position_pct: float = 25.0
) -> List[Dict]:
    """
    Calculate order quantities using the same proportional weights from the backtest.
    
    The backtest already applied inverse volatility sizing when determining shares.
    We use those share values as weights and scale to the new capital.
    
    Returns list of orders with quantities.
    ALL stocks from open_positions will be included.
    """
    if not open_positions:
        return []
    
    # Calculate total value from backtest positions (shares * current price)
    # This gives us the proportional weights from the backtest's inverse vol sizing
    total_backtest_value = 0
    position_values = []
    
    for pos in open_positions:
        shares = pos.get('Shares', 1)
        current_price = pos.get('Current Price', pos.get('Buy Price', 0))
        value = shares * current_price
        position_values.append({
            'stock': pos['Stock'],
            'shares': shares,
            'price': current_price,
            'value': value
        })
        total_backtest_value += value
    
    if total_backtest_value <= 0:
        logger.warning("Total backtest value is 0, using equal weights")
        equal_weight = 1.0 / len(open_positions)
        for pv in position_values:
            pv['weight'] = equal_weight
    else:
        # Calculate weights based on backtest position values
        for pv in position_values:
            pv['weight'] = pv['value'] / total_backtest_value
    
    # Apply max position cap
    max_weight = max_position_pct / 100.0
    for pv in position_values:
        pv['weight'] = min(pv['weight'], max_weight)
    
    # Normalize after cap
    total_weight = sum(pv['weight'] for pv in position_values)
    if total_weight > 0:
        for pv in position_values:
            pv['weight'] = pv['weight'] / total_weight
    
    orders = []
    total_used = 0
    
    # First pass: calculate initial quantities
    for pv in position_values:
        ticker_clean = pv['stock']
        weight = pv['weight']
        position_capital = capital * weight
        current_price = pv['price']
        
        if current_price <= 0:
            logger.warning(f"Skipping {ticker_clean}: no valid price")
            orders.append({
                'tradingsymbol': ticker_clean,
                'exchange': 'NSE',
                'quantity': 0,
                'price': 0,
                'weight_pct': round(weight * 100, 2),
                'allocated_capital': round(position_capital, 2),
                'estimated_value': 0,
                'note': 'No price data'
            })
            continue
        
        # Calculate quantity (whole shares only)
        quantity = int(position_capital / current_price)
        used = quantity * current_price
        total_used += used
        
        orders.append({
            'tradingsymbol': ticker_clean,
            'exchange': 'NSE',
            'quantity': quantity,
            'price': current_price,
            'weight_pct': round(weight * 100, 2),
            'allocated_capital': round(position_capital, 2),
            'estimated_value': round(used, 2),
            'note': '' if quantity > 0 else f'Need ₹{current_price:,.0f}+'
        })
    
    # Second pass: use remaining capital to buy stocks with 0 quantity
    remaining_capital = capital - total_used
    
    # Sort zero-quantity orders by price (ascending) to prioritize cheaper stocks first
    zero_qty_indices = [i for i, o in enumerate(orders) if o['quantity'] == 0 and o['price'] > 0]
    zero_qty_indices.sort(key=lambda i: orders[i]['price'])
    
    for idx in zero_qty_indices:
        order = orders[idx]
        price = order['price']
        
        # How many can we buy with remaining capital?
        can_buy = int(remaining_capital / price)
        
        if can_buy > 0:
            order['quantity'] = can_buy
            cost = can_buy * price
            order['estimated_value'] = round(cost, 2)
            order['note'] = f'+{can_buy} from unused capital'
            remaining_capital -= cost
            total_used += cost
    
    # Update notes for any still-zero stocks
    for order in orders:
        if order['quantity'] == 0 and order['price'] > 0:
            order['note'] = f'Need ₹{order["price"]:,.0f}+ (only ₹{remaining_capital:,.0f} unused)'
    
    return orders


def execute_orders_on_kite(orders: List[Dict], dry_run: bool = False) -> Dict:
    """
    Execute orders on Zerodha Kite.
    
    Args:
        orders: List of order dicts with tradingsymbol, quantity, etc.
        dry_run: If True, just validates without placing orders
        
    Returns:
        Dict with success status and order details
    """
    results = {
        'success': False,
        'orders_placed': [],
        'orders_failed': [],
        'message': ''
    }
    
    kite = get_kite_client()
    if not kite:
        results['message'] = "Kite not authenticated. Please login first."
        return results
    
    try:
        for order in orders:
            try:
                if dry_run:
                    # Just validate
                    results['orders_placed'].append({
                        'tradingsymbol': order['tradingsymbol'],
                        'quantity': order['quantity'],
                        'status': 'DRY_RUN',
                        'message': 'Order validated (not placed)'
                    })
                else:
                    # Place actual order
                    order_id = kite.place_order(
                        variety=kite.VARIETY_REGULAR,
                        tradingsymbol=order['tradingsymbol'],
                        exchange=kite.EXCHANGE_NSE,
                        transaction_type=kite.TRANSACTION_TYPE_BUY,
                        quantity=order['quantity'],
                        order_type=kite.ORDER_TYPE_MARKET,
                        product=kite.PRODUCT_CNC,  # Delivery
                        validity=kite.VALIDITY_DAY
                    )
                    
                    results['orders_placed'].append({
                        'tradingsymbol': order['tradingsymbol'],
                        'quantity': order['quantity'],
                        'order_id': order_id,
                        'status': 'PLACED',
                        'message': f'Order placed successfully: {order_id}'
                    })
                    
            except Exception as e:
                results['orders_failed'].append({
                    'tradingsymbol': order['tradingsymbol'],
                    'quantity': order['quantity'],
                    'status': 'FAILED',
                    'message': str(e)
                })
        
        if results['orders_placed'] and not results['orders_failed']:
            results['success'] = True
            results['message'] = f"Successfully placed {len(results['orders_placed'])} orders"
        elif results['orders_placed'] and results['orders_failed']:
            results['success'] = True
            results['message'] = f"Placed {len(results['orders_placed'])} orders, {len(results['orders_failed'])} failed"
        else:
            results['message'] = f"All {len(results['orders_failed'])} orders failed"
            
    except Exception as e:
        results['message'] = f"Error executing orders: {str(e)}"
        logger.error(f"Order execution error: {e}")
    
    return results


def get_kite_holdings():
    """Get current holdings from Kite."""
    kite = get_kite_client()
    if not kite:
        return None
    
    try:
        return kite.holdings()
    except Exception as e:
        logger.error(f"Error fetching holdings: {e}")
        return None


def get_kite_positions():
    """Get current positions from Kite."""
    kite = get_kite_client()
    if not kite:
        return None
    
    try:
        return kite.positions()
    except Exception as e:
        logger.error(f"Error fetching positions: {e}")
        return None


def get_kite_margins():
    """Get available margins from Kite."""
    kite = get_kite_client()
    if not kite:
        return None
    
    try:
        return kite.margins()
    except Exception as e:
        logger.error(f"Error fetching margins: {e}")
        return None
