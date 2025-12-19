"""
Symbol Mapping and Normalization

Handles symbol renames, mergers, delistings, and format normalization.
"""

from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Optional


class SymbolStatus(Enum):
    ACTIVE = "active"
    RENAMED = "renamed"
    MERGED = "merged"
    DELISTED = "delisted"
    SUSPENDED = "suspended"


@dataclass
class SymbolEvent:
    """A symbol change event (rename, merger, delisting)."""
    symbol: str
    event_date: str  # ISO format
    event_type: SymbolStatus
    new_symbol: Optional[str] = None  # For renames/mergers
    notes: str = ""


# Known symbol aliases (old -> new)
# Updated as of Dec 2024
SYMBOL_ALIASES = {
    # Renames
    'MCDOWELL-N': 'UBBL',
    'UNITECH': 'UNITECH',  # Delisted but keep for historical
    'RCOM': 'RCOM',  # Delisted
    'RPOWER': 'RPOWER',  # Delisted
    'RELCAPITAL': 'RELCAPITAL',  # Delisted
    'DHFL': 'DHFL',  # Delisted
    'YESBANK': 'YESBANK',  # Still trading after restructure
    'IDEA': 'VHL',  # Vodafone Idea
    'BHARTIINFRA': 'INDUSTOWER',
    'JUBLFOOD': 'JUBLFOOD',  # Jubilant FoodWorks
    'MOTHERSUMI': 'MOTHERSON',  # After merger
    'DIVISLAB': 'DIVISLAB',
    'HDFC': 'HDFCBANK',  # Merged 2023
    'SRTRANSFIN': 'SHRIRAMFIN',  # Renamed
    'PNB': 'PNB',
    'BANKBARODA': 'BANKBARODA',
    
    # Format normalization
    'RELIANCE.NS': 'RELIANCE',
    'TCS.NS': 'TCS',
    'INFY.NS': 'INFY',
}

# Known delisting events
DELISTING_EVENTS = [
    SymbolEvent('RCOM', '2022-12-01', SymbolStatus.DELISTED, notes='Insolvency'),
    SymbolEvent('DHFL', '2021-02-01', SymbolStatus.DELISTED, notes='Insolvency'),
    SymbolEvent('RELCAPITAL', '2022-06-01', SymbolStatus.DELISTED, notes='Insolvency'),
    SymbolEvent('UNITECH', '2020-01-01', SymbolStatus.SUSPENDED, notes='Suspended'),
    SymbolEvent('HDFC', '2023-07-01', SymbolStatus.MERGED, new_symbol='HDFCBANK', notes='Merged with HDFC Bank'),
]


def normalize_symbol(symbol: str) -> str:
    """
    Normalize a symbol to standard NSE format.
    
    - Remove .NS, .BO suffixes
    - Strip whitespace
    - Convert to uppercase
    - Apply known aliases
    """
    if not symbol:
        return symbol
    
    # Clean up
    symbol = symbol.strip().upper()
    
    # Remove exchange suffixes
    for suffix in ['.NS', '.BO', '.NSE', '.BSE']:
        if symbol.endswith(suffix):
            symbol = symbol[:-len(suffix)]
    
    # Apply alias if exists
    if symbol in SYMBOL_ALIASES:
        symbol = SYMBOL_ALIASES[symbol]
    
    return symbol


def get_symbol_status(symbol: str, as_of_date: date) -> tuple[SymbolStatus, Optional[str]]:
    """
    Check if a symbol was active, renamed, or delisted at a given date.
    
    Returns: (status, successor_symbol if applicable)
    """
    symbol = normalize_symbol(symbol)
    
    for event in DELISTING_EVENTS:
        if event.symbol == symbol:
            event_date = date.fromisoformat(event.event_date)
            if as_of_date >= event_date:
                return event.event_type, event.new_symbol
    
    return SymbolStatus.ACTIVE, None


def get_current_symbol(historical_symbol: str, as_of_date: date) -> str:
    """
    Get the current trading symbol for a historical symbol.
    
    Follows the chain of renames/mergers to find the final symbol.
    """
    symbol = normalize_symbol(historical_symbol)
    status, new_symbol = get_symbol_status(symbol, as_of_date)
    
    if status == SymbolStatus.MERGED and new_symbol:
        return new_symbol
    elif status == SymbolStatus.RENAMED and new_symbol:
        return new_symbol
    
    return symbol


def is_symbol_tradeable(symbol: str, as_of_date: date) -> bool:
    """Check if a symbol was tradeable on a given date."""
    status, _ = get_symbol_status(symbol, as_of_date)
    return status in [SymbolStatus.ACTIVE, SymbolStatus.RENAMED]
