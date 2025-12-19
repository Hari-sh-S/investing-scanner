"""
Constituent Snapshot Storage

Handles saving and loading constituent snapshots as JSON files.
File structure: data/index_constituents/{index}/{quarter}.json
"""

import json
import os
from dataclasses import dataclass, asdict
from datetime import date
from pathlib import Path
from typing import Optional


@dataclass
class ConstituentSnapshot:
    """A point-in-time snapshot of index constituents."""
    index: str                    # e.g., 'NIFTY500'
    effective_date: str           # ISO format: '2024-01-01'
    quarter: str                  # e.g., '2024_Q1'
    symbols: list[str]            # Stock symbols
    isins: list[str] = None       # Optional ISIN codes
    metadata: dict = None         # Source info, parse timestamp, etc.
    
    def __post_init__(self):
        if self.isins is None:
            self.isins = []
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def effective_date_parsed(self) -> date:
        """Return effective_date as a date object."""
        return date.fromisoformat(self.effective_date)
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ConstituentSnapshot':
        return cls(**data)


# Default data directory (relative to project root)
DEFAULT_DATA_DIR = Path(__file__).parent / 'data' / 'index_constituents'


def ensure_data_dir(data_dir: Path = None) -> Path:
    """Create data directory if it doesn't exist."""
    data_dir = data_dir or DEFAULT_DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def save_snapshot(snapshot: ConstituentSnapshot, data_dir: Path = None) -> Path:
    """
    Save a constituent snapshot to JSON file.
    
    Returns: Path to saved file
    """
    data_dir = ensure_data_dir(data_dir)
    
    # Create index subdirectory
    index_dir = data_dir / snapshot.index.lower()
    index_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as quarter.json
    file_path = index_dir / f"{snapshot.quarter}.json"
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(snapshot.to_dict(), f, indent=2, ensure_ascii=False)
    
    return file_path


def load_snapshot(index: str, quarter: str, data_dir: Path = None) -> Optional[ConstituentSnapshot]:
    """
    Load a specific quarter's snapshot.
    
    Returns: ConstituentSnapshot or None if not found
    """
    data_dir = data_dir or DEFAULT_DATA_DIR
    file_path = data_dir / index.lower() / f"{quarter}.json"
    
    if not file_path.exists():
        return None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return ConstituentSnapshot.from_dict(data)


def list_snapshots(index: str, data_dir: Path = None) -> list[str]:
    """
    List all available quarters for an index.
    
    Returns: Sorted list of quarter strings ['2020_Q1', '2020_Q2', ...]
    """
    data_dir = data_dir or DEFAULT_DATA_DIR
    index_dir = data_dir / index.lower()
    
    if not index_dir.exists():
        return []
    
    quarters = []
    for f in index_dir.glob('*.json'):
        quarters.append(f.stem)  # Remove .json extension
    
    return sorted(quarters)


def load_all_snapshots(index: str, data_dir: Path = None) -> list[ConstituentSnapshot]:
    """
    Load all snapshots for an index, sorted by effective date.
    
    Returns: List of ConstituentSnapshot ordered by date (oldest first)
    """
    quarters = list_snapshots(index, data_dir)
    snapshots = []
    
    for quarter in quarters:
        snapshot = load_snapshot(index, quarter, data_dir)
        if snapshot:
            snapshots.append(snapshot)
    
    # Sort by effective date
    snapshots.sort(key=lambda s: s.effective_date)
    
    return snapshots


def get_available_indices(data_dir: Path = None) -> list[str]:
    """
    List all indices with available data.
    
    Returns: List of index names
    """
    data_dir = data_dir or DEFAULT_DATA_DIR
    
    if not data_dir.exists():
        return []
    
    indices = []
    for d in data_dir.iterdir():
        if d.is_dir() and any(d.glob('*.json')):
            indices.append(d.name.upper())
    
    return sorted(indices)
