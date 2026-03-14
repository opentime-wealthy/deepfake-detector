# © 2026 TimeWealthy Limited — DeepGuard
"""Analysis model."""

import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from app.database import Base

from sqlalchemy import JSON as JSON_TYPE


class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(String(64), primary_key=True, default=lambda: "analysis_" + uuid.uuid4().hex[:12])
    user_id = Column(String(36), ForeignKey("users.id"), nullable=True)
    status = Column(String(20), default="pending", nullable=False)  # pending/processing/completed/failed
    mode = Column(String(20), default="standard", nullable=False)  # standard/war_footage
    source_type = Column(String(10), nullable=True)  # 'upload' or 'url'
    source_url = Column(Text, nullable=True)
    file_path = Column(Text, nullable=True)
    verdict = Column(String(20), nullable=True)  # ai_generated/likely_ai/uncertain/likely_human/human_made
    confidence = Column(Float, nullable=True)
    summary = Column(Text, nullable=True)
    details = Column(JSON_TYPE, nullable=True)
    duration_seconds = Column(Integer, nullable=True)
    video_duration_seconds = Column(Float, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="analyses")
    findings = relationship("Finding", back_populates="analysis", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Analysis {self.id} status={self.status} verdict={self.verdict}>"
