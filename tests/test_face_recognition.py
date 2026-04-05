"""Tests for faceRecognition.py — append_log() pure logic."""

from __future__ import annotations

import os

import pytest
from openpyxl import load_workbook

from faceRecognition import append_log


@pytest.fixture()
def log_path(tmp_path: pytest.TempPathFactory) -> str:
    """Return a temp path for an xlsx log that does not yet exist."""
    return str(tmp_path / "test_log.xlsx")


class TestAppendLogFileCreation:
    def test_creates_file_on_first_call(self, log_path: str) -> None:
        assert not os.path.exists(log_path)
        append_log("Alice", 42.0, log_path=log_path)
        assert os.path.exists(log_path)

    def test_creates_header_row(self, log_path: str) -> None:
        append_log("Alice", 42.0, log_path=log_path)
        wb = load_workbook(log_path)
        ws = wb.active
        headers = [ws.cell(1, c).value for c in range(1, 4)]
        assert headers == ["timestamp", "name", "confidence"]

    def test_first_data_row_written(self, log_path: str) -> None:
        append_log("Alice", 42.0, log_path=log_path)
        wb = load_workbook(log_path)
        ws = wb.active
        assert ws.cell(2, 2).value == "Alice"
        assert ws.cell(2, 3).value == 42.0


class TestAppendLogAppend:
    def test_appends_multiple_rows(self, log_path: str) -> None:
        append_log("Alice", 10.0, log_path=log_path)
        append_log("Bob", 20.0, log_path=log_path)
        append_log("Carol", 30.0, log_path=log_path)
        wb = load_workbook(log_path)
        ws = wb.active
        # Row 1 = header, rows 2-4 = data
        assert ws.max_row == 4
        assert ws.cell(3, 2).value == "Bob"
        assert ws.cell(4, 2).value == "Carol"

    def test_preserves_existing_rows(self, log_path: str) -> None:
        append_log("Alice", 1.0, log_path=log_path)
        append_log("Bob", 2.0, log_path=log_path)
        wb = load_workbook(log_path)
        ws = wb.active
        assert ws.cell(2, 2).value == "Alice"


class TestAppendLogConfidenceNone:
    def test_none_confidence_written_as_empty(self, log_path: str) -> None:
        # openpyxl reads back an empty-string cell as None
        append_log("Unknown", None, log_path=log_path)
        wb = load_workbook(log_path)
        ws = wb.active
        assert ws.cell(2, 3).value in (None, "")

    def test_none_confidence_does_not_raise(self, log_path: str) -> None:
        # Should complete without exception
        append_log("Unknown", None, log_path=log_path)

    def test_zero_confidence_stored_as_zero(self, log_path: str) -> None:
        append_log("Alice", 0.0, log_path=log_path)
        wb = load_workbook(log_path)
        ws = wb.active
        assert ws.cell(2, 3).value == 0.0


class TestAppendLogTimestamp:
    def test_timestamp_is_iso_format(self, log_path: str) -> None:
        from datetime import datetime

        append_log("Alice", 5.0, log_path=log_path)
        wb = load_workbook(log_path)
        ws = wb.active
        ts = ws.cell(2, 1).value
        # Should parse without raising
        datetime.fromisoformat(str(ts))
