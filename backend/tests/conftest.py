# © 2026 TimeWealthy Limited — DeepGuard
"""Shared test fixtures for DeepGuard backend tests."""

import os
import pytest
import numpy as np

# Must be set BEFORE any app imports
os.environ["DATABASE_URL"] = "sqlite:///./test_temp.db"
os.environ["REDIS_URL"] = "redis://localhost:6379"
os.environ["SECRET_KEY"] = "test-secret-key-for-testing-only"

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.main import app
from app.database import Base, get_db


# ── Test database (file-based temp SQLite to avoid in-memory sharing issues) ─

TEST_DATABASE_URL = "sqlite:///./test_temp.db"

engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
)
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    db = TestSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(autouse=True)
def setup_db():
    """Create all tables before each test, drop them after."""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def db_session():
    """Provide a DB session for direct model tests."""
    db = TestSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def client():
    """FastAPI TestClient with DB override."""
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c
    app.dependency_overrides.clear()


# ── Video / Audio fixtures ────────────────────────────────────────────────────

def make_frame(width=320, height=240, channels=3, value=None) -> np.ndarray:
    """Create a synthetic BGR frame."""
    if value is not None:
        return np.full((height, width, channels), value, dtype=np.uint8)
    rng = np.random.default_rng(seed=42)
    return rng.integers(0, 255, (height, width, channels), dtype=np.uint8)


@pytest.fixture
def random_frame() -> np.ndarray:
    return make_frame()


@pytest.fixture
def uniform_frame() -> np.ndarray:
    """A perfectly uniform gray frame (AI-like texture)."""
    return make_frame(value=128)


@pytest.fixture
def random_frames() -> list:
    """10 random frames (simulates ~5 seconds at 2fps)."""
    rng = np.random.default_rng(seed=0)
    return [rng.integers(0, 255, (240, 320, 3), dtype=np.uint8) for _ in range(10)]


@pytest.fixture
def uniform_frames() -> list:
    """10 identical uniform frames (very AI-like)."""
    base = make_frame(value=128)
    return [base.copy() for _ in range(10)]


@pytest.fixture
def random_audio():
    """1-second of random noise audio at 22050 Hz."""
    rng = np.random.default_rng(seed=7)
    return rng.uniform(-0.5, 0.5, 22050).astype(np.float32), 22050


@pytest.fixture
def silent_audio():
    """3 seconds of near-silence."""
    return np.zeros(22050 * 3, dtype=np.float32), 22050


@pytest.fixture
def mock_video_path(tmp_path) -> str:
    """Fake video path (file exists but is empty; used for metadata mock tests)."""
    p = tmp_path / "test_video.mp4"
    p.write_bytes(b"\x00" * 1024)
    return str(p)


# ── Auth helpers ──────────────────────────────────────────────────────────────

@pytest.fixture
def registered_user(client):
    """Register a test user and return credentials."""
    resp = client.post("/api/v1/auth/signup", json={
        "email": "test@example.com",
        "password": "TestPass123",
    })
    assert resp.status_code == 201, resp.text
    data = resp.json()
    return data


@pytest.fixture
def auth_headers(registered_user):
    """Authorization headers for a registered test user."""
    return {"Authorization": f"Bearer {registered_user['access_token']}"}
