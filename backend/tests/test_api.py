# © 2026 TimeWealthy Limited — DeepGuard
"""Tests for REST API endpoints."""

import io
import json
import pytest
from unittest.mock import patch, MagicMock


class TestAuth:

    def test_signup_returns_201(self, client):
        """POST /api/v1/auth/signup → 201"""
        resp = client.post("/api/v1/auth/signup", json={
            "email": "newuser@example.com",
            "password": "SecurePass123",
        })
        assert resp.status_code == 201
        data = resp.json()
        assert "access_token" in data
        assert data["email"] == "newuser@example.com"
        assert data["plan"] == "free"
        assert "api_key" in data

    def test_signup_duplicate_email_returns_409(self, client):
        """重複メールアドレス → 409 Conflict"""
        payload = {"email": "dup@example.com", "password": "Password123"}
        client.post("/api/v1/auth/signup", json=payload)
        resp = client.post("/api/v1/auth/signup", json=payload)
        assert resp.status_code == 409

    def test_signup_short_password_returns_422(self, client):
        """短すぎるパスワード → 422"""
        resp = client.post("/api/v1/auth/signup", json={
            "email": "test@example.com",
            "password": "short",
        })
        assert resp.status_code == 422

    def test_login_success(self, client, registered_user):
        """POST /api/v1/auth/login → 200 with token"""
        resp = client.post("/api/v1/auth/login", json={
            "email": "test@example.com",
            "password": "TestPass123",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data

    def test_login_wrong_password_returns_401(self, client, registered_user):
        """Wrong password → 401"""
        resp = client.post("/api/v1/auth/login", json={
            "email": "test@example.com",
            "password": "WrongPassword",
        })
        assert resp.status_code == 401

    def test_login_unknown_email_returns_401(self, client):
        """Unknown email → 401"""
        resp = client.post("/api/v1/auth/login", json={
            "email": "nobody@example.com",
            "password": "AnyPassword",
        })
        assert resp.status_code == 401


class TestAnalyzeUpload:

    def _make_mp4_bytes(self) -> bytes:
        """Minimal fake MP4-like bytes."""
        return b"\x00\x00\x00\x18ftypisom\x00\x00\x00\x00isomiso2" + b"\x00" * 100

    def test_upload_valid_mp4_returns_202(self, client):
        """MP4アップロード → 202 Accepted"""
        # run_analysis_task is imported locally inside the endpoint,
        # so patch at its source module
        with patch("app.tasks.analyze.run_analysis_task") as mock_task, \
             patch("threading.Thread") as mock_thread:
            # Make .delay() a no-op
            mock_task.delay = MagicMock()
            mock_thread.return_value.start = MagicMock()

            resp = client.post(
                "/api/v1/analyze",
                files={"file": ("test.mp4", self._make_mp4_bytes(), "video/mp4")},
                data={"mode": "standard"},
            )
        assert resp.status_code == 202
        data = resp.json()
        assert data["status"] == "processing"
        assert "id" in data
        assert "poll_url" in data

    def test_upload_invalid_format_returns_400(self, client):
        """非動画ファイル → 400 Bad Request"""
        resp = client.post(
            "/api/v1/analyze",
            files={"file": ("image.jpg", b"fake jpeg data", "image/jpeg")},
            data={"mode": "standard"},
        )
        assert resp.status_code == 400

    def test_upload_invalid_mode_returns_400(self, client):
        """不正なmode → 400"""
        resp = client.post(
            "/api/v1/analyze",
            files={"file": ("test.mp4", self._make_mp4_bytes(), "video/mp4")},
            data={"mode": "invalid_mode"},
        )
        assert resp.status_code == 400

    def test_quota_exceeded_returns_429(self, client, db_session):
        """クォータ超過 → 429 Too Many Requests"""
        from app.models.user import User
        from app.auth import hash_password, generate_api_key

        # Create user with exhausted quota
        user = User(
            email="quota@example.com",
            password_hash=hash_password("Password123"),
            plan="free",
            api_key=generate_api_key(),
            monthly_quota=10,
            used_this_month=10,  # Already at limit
        )
        db_session.add(user)
        db_session.commit()

        # Login
        from app.auth import create_access_token
        token = create_access_token({"sub": user.id})
        headers = {"Authorization": f"Bearer {token}"}

        resp = client.post(
            "/api/v1/analyze",
            files={"file": ("test.mp4", self._make_mp4_bytes(), "video/mp4")},
            data={"mode": "standard"},
            headers=headers,
        )
        assert resp.status_code == 429

    def test_upload_response_has_poll_url(self, client):
        """レスポンスに poll_url が含まれる"""
        with patch("threading.Thread") as mock_thread:
            mock_thread.return_value.start = MagicMock()
            resp = client.post(
                "/api/v1/analyze",
                files={"file": ("test.mp4", self._make_mp4_bytes(), "video/mp4")},
                data={"mode": "standard"},
            )
        if resp.status_code == 202:
            assert "/api/v1/results/" in resp.json()["poll_url"]


class TestAnalyzeUrl:

    def test_analyze_url_returns_202(self, client):
        """POST /api/v1/analyze-url → 202"""
        with patch("threading.Thread") as mock_thread:
            mock_thread.return_value.start = MagicMock()
            resp = client.post("/api/v1/analyze-url", json={
                "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                "mode": "standard",
            })
        assert resp.status_code == 202

    def test_analyze_url_invalid_mode_returns_400(self, client):
        """不正なmode → 400"""
        resp = client.post("/api/v1/analyze-url", json={
            "url": "https://www.youtube.com/watch?v=test",
            "mode": "bad_mode",
        })
        assert resp.status_code == 400


class TestResults:

    def test_results_not_found_returns_404(self, client):
        """存在しないID → 404"""
        resp = client.get("/api/v1/results/nonexistent_id_xyz")
        assert resp.status_code == 404

    def test_results_processing(self, client, db_session):
        """処理中のanalysis → status: processing"""
        from app.models.analysis import Analysis

        analysis = Analysis(
            id="analysis_test123",
            status="processing",
            mode="standard",
            source_type="upload",
        )
        db_session.add(analysis)
        db_session.commit()

        resp = client.get("/api/v1/results/analysis_test123")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "processing"
        assert data["verdict"] is None

    def test_results_completed(self, client, db_session):
        """完了済み → verdict + confidence"""
        from app.models.analysis import Analysis

        analysis = Analysis(
            id="analysis_done456",
            status="completed",
            mode="standard",
            source_type="upload",
            verdict="ai_generated",
            confidence=87.3,
            summary="この動画はAI生成の可能性が高い",
            details={"frame_analysis": {"score": 82.1, "findings": []}},
        )
        db_session.add(analysis)
        db_session.commit()

        resp = client.get("/api/v1/results/analysis_done456")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert data["verdict"] == "ai_generated"
        assert data["confidence"] == 87.3

    def test_results_failed(self, client, db_session):
        """失敗 → status: failed, error_message あり"""
        from app.models.analysis import Analysis

        analysis = Analysis(
            id="analysis_fail789",
            status="failed",
            mode="standard",
            error_message="ffprobe not found",
        )
        db_session.add(analysis)
        db_session.commit()

        resp = client.get("/api/v1/results/analysis_fail789")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "failed"
        assert "error_message" in data


class TestHistory:

    def test_history_requires_auth(self, client):
        """GET /api/v1/history → 認証なしで 401/403"""
        resp = client.get("/api/v1/history")
        assert resp.status_code in (401, 403)

    def test_history_returns_user_analyses(self, client, auth_headers, db_session, registered_user):
        """認証済みユーザーの履歴を返す"""
        from app.models.analysis import Analysis

        analysis = Analysis(
            id="analysis_hist001",
            user_id=registered_user["user_id"],
            status="completed",
            mode="standard",
            verdict="human_made",
            confidence=15.0,
        )
        db_session.add(analysis)
        db_session.commit()

        resp = client.get("/api/v1/history", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        ids = [item["id"] for item in data]
        assert "analysis_hist001" in ids

    def test_history_empty_for_new_user(self, client, auth_headers):
        """新規ユーザー → 空のリスト"""
        resp = client.get("/api/v1/history", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json() == []


class TestHealthAndRoot:

    def test_root_returns_service_info(self, client):
        """GET / → サービス情報"""
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert "DeepGuard" in data["service"]

    def test_health_returns_ok(self, client):
        """GET /health → {"status": "ok"}"""
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
