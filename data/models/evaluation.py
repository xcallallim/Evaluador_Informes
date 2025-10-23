"""Data structures describing the evaluation outputs produced by the evaluator.

This module centralises the schema returned by the evaluation pipeline.  Each
class offers a ``to_dict`` helper so results can be easily serialised into JSON
or sent to other layers such as repositories or presentation components.  The
``weight`` attributes exposed by questions, dimensions and sections are the
exact numeric values defined in the criteria JSON files so downstream services
can rely on perfect alignment with the source definition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional


def _weighted_average(pairs: Iterable[tuple[float, float]]) -> Optional[float]:
    """Return the weighted average of ``pairs``.

    ``pairs`` must contain tuples ``(weight, value)``.  ``value`` entries whose
    weight is ``0`` are ignored unless *all* weights are zero.  When all weights
    are zero the function falls back to a simple average of the values.  When
    ``pairs`` is empty or every value is ``None`` the function returns ``None``.
    """

    items: List[tuple[float, float]] = []
    backup_values: List[float] = []
    for weight, value in pairs:
        if value is None:
            continue
        backup_values.append(value)
        weight = float(weight)
        if weight > 0:
            items.append((weight, float(value)))

    if not backup_values:
        return None

    if not items:
        return sum(backup_values) / len(backup_values)

    total_weight = sum(weight for weight, _ in items)
    if total_weight == 0:
        return sum(value for _, value in items) / len(items)

    return sum(weight * value for weight, value in items) / total_weight


@dataclass(slots=True)
class ChunkResult:
    """Stores the evaluation returned for a single chunk of text."""

    index: int
    score: Optional[float] = None
    justification: Optional[str] = None
    relevant_text: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChunkResult":
        """Create a :class:`ChunkResult` from a dictionary."""

        return cls(
            index=data["index"],
            score=data.get("score"),
            justification=data.get("justification"),
            relevant_text=data.get("relevant_text"),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "score": self.score,
            "justification": self.justification,
            "relevant_text": self.relevant_text,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class QuestionResult:
    """Represents the consolidated score of a question across chunks."""

    question_id: str
    text: str
    weight: Optional[float] = None
    score: Optional[float] = None
    justification: Optional[str] = None
    relevant_text: Optional[str] = None
    chunk_results: List[ChunkResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QuestionResult":
        """Create a :class:`QuestionResult` from a dictionary."""

        return cls(
            question_id=data["question_id"],
            text=data["text"],
            weight=data.get("weight"),
            score=data.get("score"),
            justification=data.get("justification"),
            relevant_text=data.get("relevant_text"),
            chunk_results=[
                ChunkResult.from_dict(chunk) for chunk in data.get("chunks", [])
            ],
            metadata=dict(data.get("metadata", {})),
        )

    def best_chunk(self) -> Optional[ChunkResult]:
        if not self.chunk_results:
            return None
        sorted_chunks = sorted(
            (chunk for chunk in self.chunk_results if chunk.score is not None),
            key=lambda chunk: chunk.score,
            reverse=True,
        )
        if sorted_chunks:
            return sorted_chunks[0]
        return self.chunk_results[0]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "text": self.text,
            "weight": self.weight,
            "score": self.score,
            "justification": self.justification,
            "relevant_text": self.relevant_text,
            "chunks": [chunk.to_dict() for chunk in self.chunk_results],
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class DimensionResult:
    """Aggregates the questions that belong to the same dimension."""

    name: str
    weight: Optional[float] = None
    method: Optional[str] = None
    score: Optional[float] = None
    questions: List[QuestionResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DimensionResult":
        """Create a :class:`DimensionResult` from a dictionary."""

        return cls(
            name=data["name"],
            weight=data.get("weight"),
            method=data.get("method"),
            score=data.get("score"),
            questions=[
                QuestionResult.from_dict(question)
                for question in data.get("questions", [])
            ],
            metadata=dict(data.get("metadata", {})),
        )

    def recompute_score(self) -> Optional[float]:
        pairs = [
            (question.weight if question.weight is not None else 1.0, question.score)
            for question in self.questions
        ]
        self.score = _weighted_average(pairs)
        return self.score

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "weight": self.weight,
            "method": self.method,
            "score": self.score,
            "questions": [question.to_dict() for question in self.questions],
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class SectionResult:
    """Aggregates dimensions associated with a section in the criteria."""

    title: str
    section_id: Optional[str] = None
    weight: Optional[float] = None
    score: Optional[float] = None
    dimensions: List[DimensionResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SectionResult":
        """Create a :class:`SectionResult` from a dictionary."""

        return cls(
            section_id=data.get("section_id"),
            title=data["title"],
            weight=data.get("weight"),
            score=data.get("score"),
            dimensions=[
                DimensionResult.from_dict(dimension)
                for dimension in data.get("dimensions", [])
            ],
            metadata=dict(data.get("metadata", {})),
        )

    def recompute_score(self) -> Optional[float]:
        pairs = [
            (dimension.weight if dimension.weight is not None else 1.0, dimension.score)
            for dimension in self.dimensions
        ]
        self.score = _weighted_average(pairs)
        return self.score

    def to_dict(self) -> Dict[str, Any]:
        return {
            "section_id": self.section_id,
            "title": self.title,
            "weight": self.weight,
            "score": self.score,
            "dimensions": [dimension.to_dict() for dimension in self.dimensions],
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class EvaluationResult:
    """Top level object returned by the evaluator."""

    document_id: Optional[str]
    document_type: Optional[str]
    criteria_source: Optional[str] = None
    generated_at: datetime = field(default_factory=datetime.utcnow)
    score: Optional[float] = None
    sections: List[SectionResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def recompute_score(self) -> Optional[float]:
        pairs = [
            (section.weight if section.weight is not None else 1.0, section.score)
            for section in self.sections
        ]
        self.score = _weighted_average(pairs)
        return self.score

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationResult":
        """Create an :class:`EvaluationResult` from a dictionary."""

        generated_at = data.get("generated_at")
        if isinstance(generated_at, str):
            generated_at = datetime.fromisoformat(generated_at)
        elif generated_at is None:
            generated_at = datetime.utcnow()

        return cls(
            document_id=data.get("document_id"),
            document_type=data.get("document_type"),
            criteria_source=data.get("criteria_source"),
            generated_at=generated_at,
            score=data.get("score"),
            sections=[
                SectionResult.from_dict(section)
                for section in data.get("sections", [])
            ],
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "document_type": self.document_type,
            "criteria_source": self.criteria_source,
            "generated_at": self.generated_at.isoformat(),
            "score": self.score,
            "sections": [section.to_dict() for section in self.sections],
            "metadata": dict(self.metadata),
        }
    
__all__ = [
    "ChunkResult",
    "DimensionResult",
    "EvaluationResult",
    "QuestionResult",
    "SectionResult",
]