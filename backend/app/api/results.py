# © 2026 TimeWealthy Limited — DeepGuard
"""Results API routes: get analysis status and history."""

from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.analysis import Analysis
from app.models.user import User
from app.auth import get_current_user, get_optional_user

router = APIRouter(tags=["results"])


class AnalysisDetail(BaseModel):
    id: str
    status: str
    mode: str
    source_type: Optional[str]
    verdict: Optional[str]
    confidence: Optional[float]
    summary: Optional[str]
    details: Optional[dict]
    error_message: Optional[str]
    video_duration_seconds: Optional[float]
    created_at: datetime
    completed_at: Optional[datetime]
    poll_url: str

    class Config:
        from_attributes = True


class HistoryItem(BaseModel):
    id: str
    status: str
    verdict: Optional[str]
    confidence: Optional[float]
    source_type: Optional[str]
    created_at: datetime
    completed_at: Optional[datetime]

    class Config:
        from_attributes = True


@router.get("/results/{analysis_id}", response_model=AnalysisDetail)
def get_result(
    analysis_id: str,
    db: Session = Depends(get_db),
    user: Optional[User] = Depends(get_optional_user),
):
    """Get the current status and result of an analysis."""
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"解析ID '{analysis_id}' が見つかりません",
        )

    # Users can only see their own analyses (or anonymous analyses)
    if analysis.user_id and user and analysis.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="アクセス権限がありません")

    return AnalysisDetail(
        id=analysis.id,
        status=analysis.status,
        mode=analysis.mode,
        source_type=analysis.source_type,
        verdict=analysis.verdict,
        confidence=analysis.confidence,
        summary=analysis.summary,
        details=analysis.details,
        error_message=analysis.error_message,
        video_duration_seconds=analysis.video_duration_seconds,
        created_at=analysis.created_at,
        completed_at=analysis.completed_at,
        poll_url=f"/api/v1/results/{analysis.id}",
    )


@router.get("/history", response_model=List[HistoryItem])
def get_history(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Get the authenticated user's analysis history."""
    analyses = (
        db.query(Analysis)
        .filter(Analysis.user_id == user.id)
        .order_by(Analysis.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    return [
        HistoryItem(
            id=a.id,
            status=a.status,
            verdict=a.verdict,
            confidence=a.confidence,
            source_type=a.source_type,
            created_at=a.created_at,
            completed_at=a.completed_at,
        )
        for a in analyses
    ]
