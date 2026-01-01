"""
Dhan API Configuration
Uses official dhanhq SDK for authentication
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Dhan API Credentials
DHAN_CLIENT_ID = os.getenv("DHAN_CLIENT_ID")
DHAN_ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN")


def validate_credentials():
    """Check if credentials are configured"""
    if not DHAN_CLIENT_ID or DHAN_CLIENT_ID == "your_client_id_here":
        raise ValueError("DHAN_CLIENT_ID not configured. Please update .env file.")
    if not DHAN_ACCESS_TOKEN or DHAN_ACCESS_TOKEN == "your_access_token_here":
        raise ValueError("DHAN_ACCESS_TOKEN not configured. Please update .env file.")
    return True


def get_dhan_client():
    """Get authenticated Dhan client using official SDK.
    
    Handles different SDK versions:
    - Older versions: use DhanContext + dhanhq(context)
    - Newer versions: use dhanhq(client_id, access_token) directly
    """
    validate_credentials()
    
    # Try newer SDK first (2 args: client_id, access_token)
    try:
        from dhanhq import dhanhq
        # Check if it accepts 2 args by looking at signature
        import inspect
        sig = inspect.signature(dhanhq.__init__)
        params = list(sig.parameters.keys())
        
        if 'dhan_context' in params:
            # Old version - needs DhanContext
            from dhanhq import DhanContext
            context = DhanContext(DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN)
            return dhanhq(context)
        else:
            # New version - direct init
            return dhanhq(DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN)
    except Exception as e:
        # Fallback: try DhanContext approach
        try:
            from dhanhq import DhanContext, dhanhq
            context = DhanContext(DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN)
            return dhanhq(context)
        except Exception:
            # Last fallback: try direct 2-arg init
            from dhanhq import dhanhq
            return dhanhq(DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN)

