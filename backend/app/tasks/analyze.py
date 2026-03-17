# © 2026 TimeWealthy Limited — DeepGuard
"""Celery tasks for video analysis."""

import logging
import os
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# Try to import Celery; fall back gracefully if Redis is unavailable
try:
    from celery import Celery
    from app.config import get_settings

    settings = get_settings()
    celery_app = Celery(
        "deepguard",
        broker=settings.redis_url,
        backend=settings.redis_url,
    )
    celery_app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        timezone="UTC",
        enable_utc=True,
    )

    @celery_app.task(bind=True, max_retries=2, name="analyze_video")
    def run_analysis_task(self, analysis_id: str, file_path: str, mode: str):
        """Celery task: analyze a video file."""
        return _execute_analysis(analysis_id, file_path, mode)

    @celery_app.task(bind=True, max_retries=2, name="analyze_url")
    def run_url_analysis_task(self, analysis_id: str, url: str, mode: str):
        """Celery task: download then analyze a video URL."""
        return _execute_url_analysis(analysis_id, url, mode)

except Exception as e:
    logger.warning(f"Celery setup skipped: {e}")

    # Stub tasks that can be called with .delay() without error
    class _StubTask:
        def delay(self, *args, **kwargs):
            logger.warning("Celery not configured; running synchronously")

    run_analysis_task = _StubTask()
    run_url_analysis_task = _StubTask()


def _get_db_session():
    """Create a standalone DB session for use inside tasks."""
    from app.database import SessionLocal
    return SessionLocal()


def _execute_analysis(analysis_id: str, file_path: str, mode: str) -> dict:
    """Core analysis logic (called by Celery task or directly)."""
    db = _get_db_session()
    try:
        from app.models.analysis import Analysis
        from app.analyzers import (
            ReStraVAnalyzer, C2PAAnalyzer, TemporalAnalyzer, AudioAnalyzer,
            MetadataAnalyzer, WarFootageAnalyzer, EnsembleScorer,
        )
        from app.utils.video import extract_frames, extract_audio, get_video_duration

        analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
        if not analysis:
            logger.error(f"Analysis {analysis_id} not found in DB")
            return {"error": "Analysis not found"}

        analysis.status = "processing"
        db.commit()

        # Extract video content
        frames, fps = extract_frames(file_path, fps=2.0, max_frames=60)
        audio, sr = extract_audio(file_path)
        duration = get_video_duration(file_path)

        analyzer_results = {}

        # ReStraV: DINOv2 perceptual trajectory analysis (replaces SigLIP FrameAnalyzer)
        try:
            restrav_result = ReStraVAnalyzer().analyze(frames, fps=fps)
            analyzer_results["restrav"] = restrav_result
        except Exception as e:
            logger.warning(f"ReStraVAnalyzer failed: {e}")

        # Temporal analysis
        try:
            temporal_result = TemporalAnalyzer().analyze(frames, fps=fps)
            analyzer_results["temporal"] = temporal_result
        except Exception as e:
            logger.warning(f"TemporalAnalyzer failed: {e}")

        # Audio analysis
        if audio is not None and len(audio) > 0:
            try:
                audio_result = AudioAnalyzer().analyze(audio, sr)
                analyzer_results["audio"] = audio_result
            except Exception as e:
                logger.warning(f"AudioAnalyzer failed: {e}")

        # C2PA Content Credentials verification
        try:
            c2pa_result = C2PAAnalyzer().analyze(file_path)
            analyzer_results["c2pa"] = c2pa_result
        except Exception as e:
            logger.warning(f"C2PAAnalyzer failed: {e}")

        # Metadata analysis
        try:
            metadata_result = MetadataAnalyzer().analyze(file_path)
            analyzer_results["metadata"] = metadata_result
        except Exception as e:
            logger.warning(f"MetadataAnalyzer failed: {e}")

        # War footage analysis (only if mode == 'war_footage')
        if mode == "war_footage":
            try:
                war_result = WarFootageAnalyzer().analyze(frames, audio, sr)
                analyzer_results["war"] = war_result
            except Exception as e:
                logger.warning(f"WarFootageAnalyzer failed: {e}")

        # Ensemble scoring
        scorer = EnsembleScorer()
        verdict_data = scorer.score(analyzer_results, mode=mode)

        # Save results
        analysis.status = "completed"
        analysis.verdict = verdict_data["verdict"]
        analysis.confidence = verdict_data["score"]
        analysis.summary = verdict_data["summary"]
        analysis.details = verdict_data["details"]
        analysis.video_duration_seconds = duration
        analysis.completed_at = datetime.utcnow()
        db.commit()

        # Clean up temp file
        if file_path and os.path.exists(file_path) and "/tmp/" in file_path:
            try:
                os.unlink(file_path)
            except Exception:
                pass

        logger.info(f"Analysis {analysis_id} completed: {verdict_data['verdict']} ({verdict_data['score']})")
        return {"id": analysis_id, "verdict": verdict_data["verdict"], "score": verdict_data["score"]}

    except Exception as e:
        logger.exception(f"Analysis {analysis_id} failed: {e}")
        db.query(__import__('app.models.analysis', fromlist=['Analysis']).Analysis).filter(
            __import__('app.models.analysis', fromlist=['Analysis']).Analysis.id == analysis_id
        ).update({"status": "failed", "error_message": str(e)})
        db.commit()
        return {"error": str(e)}
    finally:
        db.close()


def _execute_url_analysis(analysis_id: str, url: str, mode: str) -> dict:
    """Download video from URL then analyze."""
    from app.utils.download import download_video

    file_path, error = download_video(url)
    if error or not file_path:
        db = _get_db_session()
        try:
            from app.models.analysis import Analysis
            analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
            if analysis:
                analysis.status = "failed"
                analysis.error_message = f"動画のダウンロードに失敗しました: {error}"
                db.commit()
        finally:
            db.close()
        return {"error": error}

    return _execute_analysis(analysis_id, file_path, mode)
