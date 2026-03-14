# © 2026 TimeWealthy Limited — DeepGuard
"""User model."""

import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, DateTime
from sqlalchemy.orm import relationship
from app.database import Base


def generate_api_key() -> str:
    return uuid.uuid4().hex + uuid.uuid4().hex  # 64 chars


class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    plan = Column(String(20), default="free", nullable=False)
    api_key = Column(String(64), unique=True, index=True)
    monthly_quota = Column(Integer, default=10, nullable=False)
    used_this_month = Column(Integer, default=0, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    analyses = relationship("Analysis", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<User {self.email} plan={self.plan}>"
