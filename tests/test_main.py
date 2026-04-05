"""Tests for main.py — _get_validated_name() pure logic."""

from __future__ import annotations

import pytest

from main import _get_validated_name


class TestGetValidatedName:
    def test_accepts_simple_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("builtins.input", lambda _: "Alice")
        assert _get_validated_name() == "Alice"

    def test_accepts_name_with_underscore(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("builtins.input", lambda _: "john_doe")
        assert _get_validated_name() == "john_doe"

    def test_accepts_name_with_hyphen(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("builtins.input", lambda _: "mary-jane")
        assert _get_validated_name() == "mary-jane"

    def test_accepts_alphanumeric(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("builtins.input", lambda _: "user42")
        assert _get_validated_name() == "user42"

    def test_strips_surrounding_whitespace(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("builtins.input", lambda _: "  Alice  ")
        assert _get_validated_name() == "Alice"

    def test_rejects_empty_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("builtins.input", lambda _: "")
        with pytest.raises(SystemExit):
            _get_validated_name()

    def test_rejects_whitespace_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("builtins.input", lambda _: "   ")
        with pytest.raises(SystemExit):
            _get_validated_name()

    def test_rejects_path_traversal_dotdot(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("builtins.input", lambda _: "../etc/passwd")
        with pytest.raises(SystemExit):
            _get_validated_name()

    def test_rejects_path_separator(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("builtins.input", lambda _: "foo/bar")
        with pytest.raises(SystemExit):
            _get_validated_name()

    def test_rejects_space_in_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("builtins.input", lambda _: "john doe")
        with pytest.raises(SystemExit):
            _get_validated_name()

    def test_rejects_special_characters(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("builtins.input", lambda _: "name<script>")
        with pytest.raises(SystemExit):
            _get_validated_name()

    def test_rejects_null_byte(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("builtins.input", lambda _: "name\x00")
        with pytest.raises(SystemExit):
            _get_validated_name()
