"""Data ingestion, validation, and loading module."""
from src.data.ingest import ingest_raw_data
from src.data.validator import DataValidator, ValidationResult
from src.data.loader import prepare_data_splits, load_splits

__all__ = [
    "ingest_raw_data",
    "DataValidator",
    "ValidationResult",
    "prepare_data_splits",
    "load_splits",
]
