"""Secure, robust CSV data loader and validator for forensic tabular analysis."""

from datetime import datetime, timezone
import io
from pathlib import Path
from typing import BinaryIO, Optional, Tuple, Union
import pandas as pd

from config import AppConfig, config as default_config
from engine.models import DatasetMetadata
from engine.utils import compute_sha256, detect_delimiter, detect_encoding, format_bytes


class DataAutopsyError(Exception):
    """Base exception for all Data Autopsy forensic operations."""
    pass


class EmptyFileError(DataAutopsyError):
    """Raised when an uploaded dataset has zero bytes or contains no data rows."""
    pass


class FileTooLargeError(DataAutopsyError):
    """Raised when file size exceeds configured safety limits."""
    pass


class MalformedCSVError(DataAutopsyError):
    """Raised when the CSV structure is corrupt or cannot be parsed."""
    pass


class EncodingDetectionError(DataAutopsyError):
    """Raised when text encoding cannot be determined."""
    pass


def load_dataset(
    source: Union[str, Path, bytes, BinaryIO, io.StringIO, io.BytesIO],
    filename: Optional[str] = None,
    cfg: Optional[AppConfig] = None,
    override_delimiter: Optional[str] = None,
    override_encoding: Optional[str] = None,
) -> Tuple[pd.DataFrame, DatasetMetadata]:
    """Load and validate an untrusted tabular CSV file for forensic analysis.
    
    Preserves original row identifiers (0-indexed integer range) and exact column names.
    Does not silently alter or drop any rows, columns, or values.

    Args:
        source: File path, raw bytes, or file-like binary stream.
        filename: Optional name for metadata tracking.
        cfg: Configuration parameters (defaults to global config).
        override_delimiter: Optional explicit delimiter character.
        override_encoding: Optional explicit character encoding.

    Returns:
        Tuple of (loaded DataFrame, DatasetMetadata).

    Raises:
        EmptyFileError: If file is empty or contains no readable rows.
        FileTooLargeError: If size exceeds configured max_file_size_bytes.
        MalformedCSVError: If parser encounters structural errors.
        EncodingDetectionError: If character encoding is unsupported.
    """
    app_cfg = cfg or default_config

    # Read raw bytes to enforce security and integrity checks
    raw_bytes: bytes
    resolved_filename: str

    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise MalformedCSVError(f"Target file does not exist: {path}")
        resolved_filename = filename or path.name
        try:
            file_size = path.stat().st_size
            if file_size > app_cfg.max_file_size_bytes:
                raise FileTooLargeError(
                    f"File '{resolved_filename}' size ({format_bytes(file_size)}) "
                    f"exceeds maximum allowed limit of {format_bytes(app_cfg.max_file_size_bytes)}."
                )
            raw_bytes = path.read_bytes()
        except OSError as e:
            raise MalformedCSVError(f"Error accessing file on disk: {e}") from e

    elif isinstance(source, bytes):
        raw_bytes = source
        resolved_filename = filename or "uploaded_dataset.csv"

    elif hasattr(source, "read"):
        # File-like object (e.g. io.BytesIO or Streamlit UploadedFile)
        resolved_filename = getattr(source, "name", None) or filename or "uploaded_dataset.csv"
        # Reset pointer if possible
        if hasattr(source, "seek"):
            source.seek(0)
        data = source.read()
        if isinstance(data, str):
            raw_bytes = data.encode("utf-8")
        else:
            raw_bytes = bytes(data)
    else:
        raise MalformedCSVError(f"Unsupported source type: {type(source).__name__}")

    # 1. Check for empty payload
    total_bytes = len(raw_bytes)
    if total_bytes == 0:
        raise EmptyFileError(f"Dataset '{resolved_filename}' is empty (0 bytes).")

    if total_bytes > app_cfg.max_file_size_bytes:
        raise FileTooLargeError(
            f"File '{resolved_filename}' size ({format_bytes(total_bytes)}) "
            f"exceeds safety limit of {format_bytes(app_cfg.max_file_size_bytes)}."
        )

    # 2. Compute cryptographic SHA256 integrity hash
    sha256_hash = compute_sha256(raw_bytes)

    # 3. Detect / validate encoding
    if override_encoding:
        encoding = override_encoding
    else:
        try:
            encoding = detect_encoding(raw_bytes, app_cfg.supported_encodings)
        except ValueError as e:
            raise EncodingDetectionError(
                f"Could not determine valid text encoding for '{resolved_filename}'. "
                f"Tried encodings: {app_cfg.supported_encodings}."
            ) from e

    # 4. Decode initial sample to sniff delimiter
    try:
        sample_text = raw_bytes[:65536].decode(encoding, errors="replace")
    except Exception as e:
        raise EncodingDetectionError(f"Failed decoding initial file bytes: {e}") from e

    if not sample_text.strip():
        raise EmptyFileError(f"Dataset '{resolved_filename}' contains only whitespace.")

    if override_delimiter:
        delimiter = override_delimiter
    else:
        delimiter = detect_delimiter(sample_text, app_cfg.supported_delimiters)

    # 5. Parse CSV into DataFrame using pandas engine with strict safety settings
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=pd.errors.ParserWarning)
            df = pd.read_csv(
                io.BytesIO(raw_bytes),
                sep=delimiter,
                encoding=encoding,
                engine="python",
                index_col=False,
                on_bad_lines="error",  # Do not silently drop malformed lines in forensic mode
                keep_default_na=True,
            )
    except pd.errors.EmptyDataError as e:
        raise EmptyFileError(f"CSV file contains no columns or records: {e}") from e
    except (pd.errors.ParserError, pd.errors.ParserWarning) as e:
        raise MalformedCSVError(f"Structural syntax violation detected in CSV: {e}") from e
    except Exception as e:
        raise MalformedCSVError(f"Unable to parse CSV structure: {e}") from e

    # 6. Validate parsed DataFrame
    if df.empty and len(df.columns) == 0:
        raise EmptyFileError("Dataset parsed with zero columns and zero rows.")

    # Ensure index is 0-indexed integer range representing original row index
    df = df.reset_index(drop=True)

    metadata = DatasetMetadata(
        filename=resolved_filename,
        file_size_bytes=total_bytes,
        sha256_hash=sha256_hash,
        encoding=encoding,
        delimiter=delimiter,
        row_count=int(len(df)),
        column_count=int(len(df.columns)),
        loaded_at=datetime.now(timezone.utc).isoformat(),
    )

    return df, metadata
