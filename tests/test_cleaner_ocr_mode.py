"""Tests for OCR mode selection in :mod:`data.preprocessing.cleaner`."""

from __future__ import annotations

from data.models.document import Document
from data.preprocessing.cleaner import Cleaner


def _stub_cluster_calls(monkeypatch):
    calls: list[bool] = []

    def fake_cluster(
        self,
        candidates,
        *,
        use_fuzzy: bool,
        similarity: int,
        min_repeats: int,
        normalizer,
    ):
        calls.append(use_fuzzy)
        return []

    monkeypatch.setattr(Cleaner, "_cluster_repeated_lines", fake_cluster)
    return calls


def test_clean_document_respects_forced_ocr(monkeypatch):
    """When metadata sets ``is_ocr`` the cleaner honours the flag."""

    calls = _stub_cluster_calls(monkeypatch)
    cleaner = Cleaner()
    document = Document(
        content="=== PAGE 1 ===\nHeader\nBody\n",
        metadata={"is_ocr": True},
    )

    cleaner.clean_document(document)

    assert calls == [True, True]


def test_clean_document_can_disable_inferred_ocr(monkeypatch):
    """Explicit ``False`` must win over the heuristic."""

    calls = _stub_cluster_calls(monkeypatch)
    monkeypatch.setattr(Cleaner, "_infer_is_ocr_like", lambda self, pages: True)
    cleaner = Cleaner()
    document = Document(
        content="=== PAGE 1 ===\nHeader\nBody\n",
        metadata={"is_ocr": False},
    )

    cleaner.clean_document(document)

    assert calls == [False, False]


def test_clean_uses_heuristic_when_no_override(monkeypatch):
    """If no override is provided the heuristic determines the mode."""

    calls = _stub_cluster_calls(monkeypatch)
    monkeypatch.setattr(Cleaner, "_infer_is_ocr_like", lambda self, pages: True)
    cleaner = Cleaner()

    cleaner.clean("=== PAGE 1 ===\nHeader\nBody\n")

    assert calls == [True, True]
    