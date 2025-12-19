"""
Historical Index Constituents Module

Provides survivorship-bias-free universe lookups by returning
index constituents valid at each point in time.

Usage:
    from historical_constituents import get_index_universe
    
    universe = get_index_universe('NIFTY500', pd.Timestamp('2022-06-15'))
"""

from .api import get_index_universe, get_available_indices, get_coverage_range
from .store import load_snapshot, list_snapshots
from .validation import validate_universe

__all__ = [
    'get_index_universe',
    'get_available_indices', 
    'get_coverage_range',
    'load_snapshot',
    'list_snapshots',
    'validate_universe'
]
