"""Unit tests for model lifecycle manager."""

import pytest
from src.config.settings import BASE_DIR
from src.serving.manager import ModelManager


def test_model_manager_info():
    """Test model manager provides comprehensive structured metadata."""
    manager = ModelManager()
    info = manager.get_info()

    assert "status" in info
    assert "model_name" in info
    assert "version" in info
    assert "features" in info
    assert "numerical" in info["features"]
    assert "categorical" in info["features"]
    assert "validation_metrics" in info


def test_model_manager_missing_file():
    """Test model manager raises informative error when artifact path does not exist."""
    fake_path = BASE_DIR / "non_existent_dir" / "missing_model.joblib"
    manager = ModelManager(model_path=fake_path)

    with pytest.raises(FileNotFoundError) as exc_info:
        manager.load()
    assert "not found" in str(exc_info.value).lower()
