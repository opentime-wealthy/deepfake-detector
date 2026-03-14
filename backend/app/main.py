# © 2026 TimeWealthy Limited — DeepGuard
"""DeepGuard FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.database import init_db
from app.api.auth import router as auth_router
from app.api.analyze import router as analyze_router
from app.api.results import router as results_router

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "DeepGuard: AI生成動画検出SaaS API\n\n"
        "動画がAI生成か人間制作かを判定します。\n\n"
        "© 2026 TimeWealthy Limited — DeepGuard"
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — adjust origins for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routes under /api/v1
app.include_router(auth_router, prefix="/api/v1")
app.include_router(analyze_router, prefix="/api/v1")
app.include_router(results_router, prefix="/api/v1")


@app.on_event("startup")
def startup_event():
    """Initialize DB tables on startup (dev mode)."""
    init_db()


@app.get("/")
def root():
    return {
        "service": "DeepGuard API",
        "version": settings.app_version,
        "copyright": "© 2026 TimeWealthy Limited — DeepGuard",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok"}
