# © 2026 TimeWealthy Limited — DeepGuard
"""Celery worker entry point.

Usage:
    celery -A celery_worker.celery_app worker --loglevel=info
"""

from app.tasks.analyze import celery_app  # noqa: F401
