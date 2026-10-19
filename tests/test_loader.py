"""Unit tests for forensic CSV loader and validator."""

import io
from pathlib import Path
import pytest
import pandas as pd

from config import AppConfig
from engine.loader import (
    load_dataset,
    EmptyFileError,
    FileTooLargeError,
    MalformedCSVError,
    EncodingDetectionError,
)


@pytest.fixture
def clean_csv_bytes() -> bytes:
    content = (
        "patient_id,age,blood_pressure,status\n"
        "P01,45,120.5,active\n"
        "P02,52,135.0,inactive\n"
        "P03,34,110.2,active\n"
    )
    return content.encode("utf-8")


@pytest.fixture
def semicolon_csv_bytes() -> bytes:
    content = (
        "sku;product_name;price;in_stock\n"
        "A101;Diagnostic Kit;199.99;True\n"
        "B202;Specimen Vial;12.50;False\n"
        "C303;Assay Plate;45.00;True\n"
    )
    return content.encode("utf-8")


def test_load_valid_csv(clean_csv_bytes):
    """Test standard valid CSV loading with metadata generation."""
    df, meta = load_dataset(clean_csv_bytes, filename="patients.csv")

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert list(df.columns) == ["patient_id", "age", "blood_pressure", "status"]
    assert meta.filename == "patients.csv"
    assert meta.row_count == 3
    assert meta.column_count == 4
    assert meta.delimiter == ","
    assert meta.encoding in ["utf-8", "utf-8-sig"]
    assert len(meta.sha256_hash) == 64


def test_load_from_bytesio_buffer(clean_csv_bytes):
    """Test loading from an in-memory buffer (mimics Streamlit UploadedFile)."""
    buffer = io.BytesIO(clean_csv_bytes)
    buffer.name = "buffer_test.csv"

    df, meta = load_dataset(buffer)
    assert len(df) == 3
    assert meta.filename == "buffer_test.csv"


def test_load_empty_bytes_raises_empty_file_error():
    """Verify 0-byte input raises EmptyFileError."""
    with pytest.raises(EmptyFileError, match="is empty"):
        load_dataset(b"", filename="empty.csv")


def test_load_whitespace_only_raises_empty_file_error():
    """Verify whitespace-only payload raises EmptyFileError."""
    with pytest.raises(EmptyFileError, match="whitespace"):
        load_dataset(b"   \n\n\t  \n", filename="blank.csv")


def test_file_too_large_raises_error(clean_csv_bytes):
    """Verify safety limit enforcement when payload exceeds configured max bytes."""
    small_cfg = AppConfig(max_file_size_mb=0)  # max_file_size_bytes = 0
    with pytest.raises(FileTooLargeError, match="exceeds"):
        load_dataset(clean_csv_bytes, cfg=small_cfg)


def test_malformed_csv_structure_raises_error():
    """Verify bad lines raise MalformedCSVError."""
    # Line 2 has 2 fields, Line 3 has 5 fields
    bad_content = (
        "col1,col2,col3\n"
        "val1,val2\n"
        "val1,val2,val3,extra1,extra2\n"
    ).encode("utf-8")

    with pytest.raises(MalformedCSVError):
        load_dataset(bad_content, filename="corrupt.csv")


def test_detect_semicolon_delimiter(semicolon_csv_bytes):
    """Verify automatic delimiter detection for semicolon-separated records."""
    df, meta = load_dataset(semicolon_csv_bytes, filename="catalog.csv")

    assert meta.delimiter == ";"
    assert len(df) == 3
    assert "sku" in df.columns
    assert "price" in df.columns


def test_tab_delimited_dataset():
    """Verify automatic delimiter detection for tab-separated values."""
    tsv_content = "id\tval\ttag\n1\t10.5\tA\n2\t20.2\tB\n".encode("utf-8")
    df, meta = load_dataset(tsv_content, filename="data.tsv")

    assert meta.delimiter == "\t"
    assert len(df) == 2
    assert list(df.columns) == ["id", "val", "tag"]


def test_latin1_encoding_fallback():
    """Verify fallback handling for non-UTF8 encoded CSVs."""
    # Latin-1 character 'é' = \xe9
    raw = "nom,ville\nRené,Montréal\n".encode("latin-1")
    df, meta = load_dataset(raw, filename="french.csv")

    assert meta.encoding == "latin-1"
    assert len(df) == 1
    assert df.iloc[0]["nom"] == "René"


def test_preserves_original_row_indices_and_columns(clean_csv_bytes):
    """Verify original row indices (0..N-1) and verbatim column names are preserved."""
    df, _ = load_dataset(clean_csv_bytes)

    assert list(df.index) == [0, 1, 2]
    assert df.iloc[0]["patient_id"] == "P01"
    assert df.iloc[2]["patient_id"] == "P03"


def test_load_sample_clean_file_from_disk():
    """Verify loading pre-packaged sample clean dataset from filesystem."""
    sample_path = Path(__file__).resolve().parent.parent / "sample_data" / "clean_patient_records.csv"
    assert sample_path.exists()

    df, meta = load_dataset(sample_path)
    assert len(df) == 10
    assert meta.column_count == 7
    assert meta.filename == "clean_patient_records.csv"
