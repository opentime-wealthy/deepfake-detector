# © 2026 TimeWealthy Limited — DeepGuard
"""Finding model."""

import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from app.database import Base

from sqlalchemy import JSON as JSON_TYPE


class Finding(Base):
    __tablename__ = "findings"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    analysis_id = Column(String(64), ForeignKey("analyses.id"), nullable=False)
    analyzer = Column(String(30), nullable=False)  # frame/temporal/audio/metadata/war
    type = Column(String(50), nullable=False)
    confidence = Column(Float, nullable=False)
    description = Column(Text, nullable=False)
    frame_number = Column(Integer, nullable=True)
    timestamp_sec = Column(Float, nullable=True)
    extra_metadata = Column("metadata", JSON_TYPE, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    analysis = relationship("Analysis", back_populates="findings")

    def __repr__(self) -> str:
        return f"<Finding {self.analyzer}:{self.type} confidence={self.confidence}>"
