"""Pytest configuration and fixtures"""
import pytest
import tempfile
from pathlib import Path


@pytest.fixture(scope="session")
def tmp_data_dir():
    """Session-scoped temp directory for all tests"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
