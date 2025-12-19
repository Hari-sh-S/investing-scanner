"""
Validation and Audit Utilities

Sanity checks for constituent data and audit logging for backtests.
"""

import logging
from dataclasses import dataclass
from datetime import date
from typing import Optional

from .store import ConstituentSnapshot, load_all_snapshots


# Configure logging
logger = logging.getLogger('historical_constituents')


# Expected symbol counts per index
EXPECTED_COUNTS = {
    'NIFTY50': (50, 50),      # (min, max)
    'NIFTY100': (100, 100),
    'NIFTY200': (200, 200),
    'NIFTY500': (490, 510),   # Some variance allowed
    'NIFTYMIDCAP150': (145, 155),
    'NIFTYSMALLCAP250': (245, 255),
    'BSE500': (490, 510),
}


@dataclass
class ValidationResult:
    """Result of validation checks."""
    is_valid: bool
    warnings: list[str]
    errors: list[str]
    
    def __bool__(self):
        return self.is_valid


@dataclass  
class UniverseChange:
    """Changes between two snapshots."""
    from_quarter: str
    to_quarter: str
    additions: list[str]
    removals: list[str]
    turnover_pct: float


def validate_snapshot(snapshot: ConstituentSnapshot) -> ValidationResult:
    """
    Validate a single constituent snapshot.
    
    Checks:
    - Symbol count in expected range
    - No duplicate symbols
    - Valid symbol format
    """
    warnings = []
    errors = []
    
    index = snapshot.index.upper()
    count = len(snapshot.symbols)
    
    # Check symbol count
    if index in EXPECTED_COUNTS:
        min_count, max_count = EXPECTED_COUNTS[index]
        if count < min_count:
            errors.append(f"Too few symbols: {count} < {min_count}")
        elif count > max_count:
            warnings.append(f"More symbols than expected: {count} > {max_count}")
    
    # Check for duplicates
    unique_symbols = set(snapshot.symbols)
    if len(unique_symbols) != count:
        dup_count = count - len(unique_symbols)
        errors.append(f"Found {dup_count} duplicate symbols")
    
    # Check symbol format
    for symbol in snapshot.symbols:
        if not symbol or len(symbol) > 20:
            errors.append(f"Invalid symbol format: '{symbol}'")
        if ' ' in symbol:
            warnings.append(f"Symbol contains spaces: '{symbol}'")
    
    is_valid = len(errors) == 0
    return ValidationResult(is_valid, warnings, errors)


def validate_universe(index: str) -> list[ValidationResult]:
    """
    Validate all snapshots for an index.
    
    Returns list of validation results for each snapshot.
    """
    snapshots = load_all_snapshots(index)
    return [validate_snapshot(s) for s in snapshots]


def audit_universe_changes(
    prev_snapshot: ConstituentSnapshot,
    curr_snapshot: ConstituentSnapshot
) -> UniverseChange:
    """
    Analyze changes between two consecutive snapshots.
    
    Returns additions, removals, and turnover percentage.
    """
    prev_set = set(prev_snapshot.symbols)
    curr_set = set(curr_snapshot.symbols)
    
    additions = list(curr_set - prev_set)
    removals = list(prev_set - curr_set)
    
    # Turnover = (additions + removals) / avg_size
    avg_size = (len(prev_set) + len(curr_set)) / 2
    turnover = (len(additions) + len(removals)) / avg_size * 100 if avg_size > 0 else 0
    
    return UniverseChange(
        from_quarter=prev_snapshot.quarter,
        to_quarter=curr_snapshot.quarter,
        additions=sorted(additions),
        removals=sorted(removals),
        turnover_pct=round(turnover, 2)
    )


def get_all_universe_changes(index: str) -> list[UniverseChange]:
    """Get all quarterly changes for an index."""
    snapshots = load_all_snapshots(index)
    changes = []
    
    for i in range(1, len(snapshots)):
        change = audit_universe_changes(snapshots[i-1], snapshots[i])
        changes.append(change)
        
        # Warn about unusual turnover
        if change.turnover_pct > 20:
            logger.warning(
                f"High turnover in {index}: {change.from_quarter} → {change.to_quarter}: "
                f"{change.turnover_pct:.1f}% ({len(change.additions)} added, {len(change.removals)} removed)"
            )
    
    return changes


def log_rebalance_universe(
    rebalance_date: date,
    index: str,
    universe_size: int,
    used_historical: bool = True
):
    """
    Log universe size at each rebalance for audit trail.
    
    Call this from the backtest engine after getting the universe.
    """
    source = "historical" if used_historical else "current"
    logger.info(
        f"REBALANCE {rebalance_date}: {index} universe = {universe_size} stocks ({source})"
    )
