# © 2026 TimeWealthy Limited — DeepGuard
"""Base classes for all analyzers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class Finding:
    type: str
    confidence: float  # 0-100
    description: str
    frame_number: Optional[int] = None
    timestamp_sec: Optional[float] = None
    frames: Optional[List[int]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class AnalyzerResult:
    score: float  # 0-100, higher = more likely AI-generated
    findings: List[Finding] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def has_error(self) -> bool:
        return self.error is not None


class BaseAnalyzer(ABC):
    """Abstract base class for all analyzers."""

    @abstractmethod
    def analyze(self, *args, **kwargs) -> AnalyzerResult:
        raise NotImplementedError
