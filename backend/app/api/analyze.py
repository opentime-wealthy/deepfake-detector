# © 2026 TimeWealthy Limited — DeepGuard
"""Analysis API routes: upload and URL-based analysis."""

import os
import tempfile
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, HttpUrl
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db
from app.models.analysis import Analysis
from app.models.user import User
from app.auth import get_current_user, get_optional_user

router = APIRouter(tags=["analysis"])
settings = get_settings()

ALLOWED_MIME_TYPES = {
    "video/mp4", "video/quicktime", "video/webm",
    "video/x-msvideo", "video/x-matroska", "video/mpeg",
}
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".webm", ".avi", ".mkv", ".mpeg", ".mpg"}


class AnalyzeUrlRequest(BaseModel):
    url: str
    mode: str = "standard"


class AnalysisResponse(BaseModel):
    id: str
    status: str
    estimated_seconds: int
    poll_url: str


def _check_quota(user: Optional[User], db: Session) -> None:
    """Raise 429 if the user has exceeded their monthly quota."""
    if user is None:
        return  # anonymous — no quota enforced in MVP
    if user.used_this_month >= user.monthly_quota:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"月間クォータを超過しました ({user.monthly_quota}回/月)。プランをアップグレードしてください。",
        )


def _increment_quota(user: Optional[User], db: Session) -> None:
    if user is None:
        return
    user.used_this_month += 1
    db.commit()


def _create_analysis(db: Session, user: Optional[User], mode: str, source_type: str) -> Analysis:
    analysis = Analysis(
        id="analysis_" + uuid.uuid4().hex[:12],
        user_id=user.id if user else None,
        status="processing",
        mode=mode,
        source_type=source_type,
    )
    db.add(analysis)
    db.commit()
    db.refresh(analysis)
    return analysis


@router.post("/analyze", response_model=AnalysisResponse, status_code=status.HTTP_202_ACCEPTED)
async def analyze_upload(
    file: UploadFile = File(...),
    mode: str = Form(default="standard"),
    db: Session = Depends(get_db),
    user: Optional[User] = Depends(get_optional_user),
):
    """Upload a video file and start AI generation analysis."""
    # Validate format
    content_type = file.content_type or ""
    ext = os.path.splitext(file.filename or "")[1].lower()

    if content_type not in ALLOWED_MIME_TYPES and ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"サポートされていないファイル形式です。MP4/MOV/WebM のみ対応しています。(received: {content_type})",
        )

    # Validate mode
    if mode not in ("standard", "war_footage"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="modeは 'standard' または 'war_footage' を指定してください",
        )

    # Check quota
    _check_quota(user, db)

    # Save uploaded file to temp directory
    suffix = ext or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix="deepguard_") as tmp:
        content = await file.read()

        # Check file size
        max_bytes = settings.max_upload_mb * 1024 * 1024
        if len(content) > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"ファイルサイズが制限({settings.max_upload_mb}MB)を超えています",
            )

        tmp.write(content)
        tmp_path = tmp.name

    # Create analysis record
    analysis = _create_analysis(db, user, mode, "upload")
    analysis.file_path = tmp_path
    db.commit()

    # Increment quota
    _increment_quota(user, db)

    # Dispatch async task
    try:
        from app.tasks.analyze import run_analysis_task
        run_analysis_task.delay(analysis.id, tmp_path, mode)
    except Exception as e:
        # If Celery is not available, run synchronously (dev mode)
        import threading
        def _run_sync():
            from app.tasks.analyze import _execute_analysis
            _execute_analysis(analysis.id, tmp_path, mode)
        threading.Thread(target=_run_sync, daemon=True).start()

    return AnalysisResponse(
        id=analysis.id,
        status="processing",
        estimated_seconds=45,
        poll_url=f"/api/v1/results/{analysis.id}",
    )


@router.post("/analyze-url", response_model=AnalysisResponse, status_code=status.HTTP_202_ACCEPTED)
async def analyze_url(
    request: AnalyzeUrlRequest,
    db: Session = Depends(get_db),
    user: Optional[User] = Depends(get_optional_user),
):
    """Provide a video URL and start AI generation analysis."""
    if request.mode not in ("standard", "war_footage"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="modeは 'standard' または 'war_footage' を指定してください",
        )

    _check_quota(user, db)

    # Create analysis record
    analysis = _create_analysis(db, user, request.mode, "url")
    analysis.source_url = request.url
    db.commit()

    _increment_quota(user, db)

    # Dispatch async task (download + analyze)
    try:
        from app.tasks.analyze import run_url_analysis_task
        run_url_analysis_task.delay(analysis.id, request.url, request.mode)
    except Exception as e:
        import threading
        def _run_sync():
            from app.tasks.analyze import _execute_url_analysis
            _execute_url_analysis(analysis.id, request.url, request.mode)
        threading.Thread(target=_run_sync, daemon=True).start()

    return AnalysisResponse(
        id=analysis.id,
        status="processing",
        estimated_seconds=90,
        poll_url=f"/api/v1/results/{analysis.id}",
    )
