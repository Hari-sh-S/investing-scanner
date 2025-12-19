"""
Point-in-Time Universe API

The main interface for survivorship-bias-free universe lookups.
"""

import pandas as pd
from datetime import date
from typing import Optional

from .store import load_all_snapshots, get_available_indices as _get_available_indices, DEFAULT_DATA_DIR
from .symbol_mapping import normalize_symbol, is_symbol_tradeable


class MissingConstituentDataError(Exception):
    """Raised when no constituent data is available for the requested date."""
    pass


class FutureDateError(Exception):
    """Raised when a future date is requested (lookahead bias prevention)."""
    pass


def get_index_universe(
    index_name: str,
    as_of_date: pd.Timestamp,
    check_tradeable: bool = True,
    raise_on_missing: bool = True
) -> list[str]:
    """
    Returns index constituents valid at as_of_date.
    
    This is the PRIMARY API for backtest integration.
    
    Args:
        index_name: Index identifier (e.g., 'NIFTY500', 'NIFTY50')
        as_of_date: The date for which to get constituents
        check_tradeable: If True, filter out delisted/suspended stocks
        raise_on_missing: If True, raise error when no data; else return empty list
    
    Returns:
        List of stock symbols valid at the given date
    
    Raises:
        FutureDateError: If as_of_date is in the future
        MissingConstituentDataError: If no constituent data exists for/before as_of_date
    
    Example:
        >>> universe = get_index_universe('NIFTY500', pd.Timestamp('2022-06-15'))
        >>> len(universe)
        500
    """
    # Prevent lookahead bias
    today = date.today()
    query_date = as_of_date.date() if hasattr(as_of_date, 'date') else as_of_date
    
    if query_date > today:
        raise FutureDateError(
            f"Cannot request future constituents ({query_date}). "
            f"Today is {today}. This would introduce lookahead bias."
        )
    
    # Load all snapshots for this index
    snapshots = load_all_snapshots(index_name.upper())
    
    if not snapshots:
        if raise_on_missing:
            raise MissingConstituentDataError(
                f"No constituent data found for index '{index_name}'. "
                f"Available indices: {get_available_indices()}"
            )
        return []
    
    # Find the latest snapshot with effective_date <= as_of_date
    valid_snapshot = None
    for snapshot in snapshots:
        snapshot_date = snapshot.effective_date_parsed
        if snapshot_date <= query_date:
            valid_snapshot = snapshot
        else:
            break  # Snapshots are sorted, so we can stop
    
    if valid_snapshot is None:
        if raise_on_missing:
            earliest = snapshots[0].effective_date
            raise MissingConstituentDataError(
                f"No constituent data available for {index_name} before {query_date}. "
                f"Earliest available: {earliest}"
            )
        return []
    
    # Normalize and filter symbols
    symbols = [normalize_symbol(s) for s in valid_snapshot.symbols]
    
    if check_tradeable:
        symbols = [s for s in symbols if is_symbol_tradeable(s, query_date)]
    
    return symbols


def get_available_indices() -> list[str]:
    """
    List all indices with available historical data.
    
    Returns:
        List of index names (e.g., ['NIFTY50', 'NIFTY500'])
    """
    return _get_available_indices()


def get_coverage_range(index_name: str) -> tuple[Optional[date], Optional[date]]:
    """
    Get the date range covered by available data for an index.
    
    Returns:
        (earliest_date, latest_date) or (None, None) if no data
    """
    snapshots = load_all_snapshots(index_name.upper())
    
    if not snapshots:
        return None, None
    
    earliest = snapshots[0].effective_date_parsed
    latest = snapshots[-1].effective_date_parsed
    
    return earliest, latest


def get_universe_with_fallback(
    index_name: str,
    as_of_date: pd.Timestamp,
    fallback_universe: list[str]
) -> tuple[list[str], bool]:
    """
    Get historical universe with fallback to provided universe if missing.
    
    Returns:
        (symbols, used_historical) - tuple of symbols and whether historical data was used
    """
    try:
        symbols = get_index_universe(index_name, as_of_date)
        return symbols, True
    except MissingConstituentDataError:
        return fallback_universe, False
