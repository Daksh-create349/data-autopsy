"""Reusable utility functions for Data Autopsy."""

import csv
import hashlib
import time
from contextlib import contextmanager
from typing import Any, Generator, List, Optional
import numpy as np
import pandas as pd


def compute_sha256(data: bytes) -> str:
    """Calculate SHA256 hexadecimal hash for raw byte sequence.
    
    Args:
        data: Raw byte payload.
        
    Returns:
        Hexadecimal SHA256 digest string.
    """
    hasher = hashlib.sha256()
    hasher.update(data)
    return hasher.hexdigest()


def format_bytes(size_bytes: int) -> str:
    """Format an integer byte count into a human-readable string.
    
    Args:
        size_bytes: Byte count.
        
    Returns:
        Formatted string (e.g. '1.45 MB').
    """
    if size_bytes < 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    unit_index = 0
    val = float(size_bytes)
    while val >= 1024.0 and unit_index < len(units) - 1:
        val /= 1024.0
        unit_index += 1
    return f"{val:.2f} {units[unit_index]}" if unit_index > 0 else f"{int(val)} B"


def to_python_scalar(value: Any) -> Any:
    """Recursively convert NumPy, Pandas, and custom types into standard JSON-serializable Python types.
    
    Preserves exact values without inventing data. Converts NaN / NaT / pd.NA to None.
    
    Args:
        value: Any input object or primitive.
        
    Returns:
        Primitive Python object (int, float, str, bool, list, dict, or None).
    """
    if value is None:
        return None
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return value.isoformat()
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.ndarray,)):
        return [to_python_scalar(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(k): to_python_scalar(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_python_scalar(item) for item in value]
    return value


def detect_encoding(raw_bytes: bytes, candidate_encodings: Optional[List[str]] = None) -> str:
    """Attempt decoding with candidate encodings in priority order.
    
    Args:
        raw_bytes: Initial chunk of the file or full bytes.
        candidate_encodings: List of encodings to try.
        
    Returns:
        The first working encoding string.
        
    Raises:
        ValueError: If none of the candidate encodings can decode the byte sequence.
    """
    if candidate_encodings is None:
        candidate_encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1"]

    sample = raw_bytes[:65536]  # Test initial 64KB
    for enc in candidate_encodings:
        try:
            sample.decode(enc)
            return enc
        except (UnicodeDecodeError, LookupError):
            continue

    raise ValueError("Unable to determine valid text encoding for file.")


def detect_delimiter(sample_text: str, candidate_delimiters: Optional[List[str]] = None) -> str:
    """Determine the most likely CSV delimiter by analyzing character frequency and consistency.
    
    Falls back to standard comma if indeterminate.
    
    Args:
        sample_text: Sample text lines from CSV.
        candidate_delimiters: Allowed delimiter characters.
        
    Returns:
        Determined delimiter character.
    """
    if candidate_delimiters is None:
        candidate_delimiters = [",", ";", "\t", "|"]

    if not sample_text.strip():
        return ","

    try:
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample_text[:16384], delimiters="".join(candidate_delimiters))
        if dialect.delimiter in candidate_delimiters:
            return dialect.delimiter
    except Exception:
        pass

    # Frequency-based fallback across lines
    lines = [line.strip() for line in sample_text.strip().splitlines()[:10] if line.strip()]
    if not lines:
        return ","

    best_delim = ","
    max_score = -1

    for delim in candidate_delimiters:
        counts = [line.count(delim) for line in lines]
        if all(c > 0 for c in counts):
            # Check consistency of count across lines
            if len(set(counts)) == 1 and counts[0] > max_score:
                max_score = counts[0]
                best_delim = delim

    return best_delim


@contextmanager
def execution_timer() -> Generator[dict, None, None]:
    """Context manager measuring execution duration in milliseconds.
    
    Yields:
        Dictionary containing 'duration_ms'.
    """
    result = {"duration_ms": 0.0}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["duration_ms"] = round((time.perf_counter() - start) * 1000, 2)
