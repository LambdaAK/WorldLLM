"""
Async database models + connection helpers for TinyGPT request logging.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, String, func
from sqlalchemy.ext.asyncio import AsyncAttrs, AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


DEFAULT_DATABASE_URL = "sqlite+aiosqlite:///./tinygpt.db"

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None
_database_url: str | None = None


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def normalize_database_url(raw_url: str | None) -> str | None:
    if raw_url is None:
        return None
    url = raw_url.strip()
    if not url:
        return None

    if url.startswith("postgres://"):
        return "postgresql+asyncpg://" + url[len("postgres://") :]
    if url.startswith("postgresql://") and not url.startswith("postgresql+asyncpg://"):
        return "postgresql+asyncpg://" + url[len("postgresql://") :]
    if url.startswith("sqlite:///"):
        return "sqlite+aiosqlite:///" + url[len("sqlite:///") :]
    return url


class Base(AsyncAttrs, DeclarativeBase):
    pass


class RequestLog(Base):
    __tablename__ = "request_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    request_id: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(32), default="ok")
    temperature: Mapped[float] = mapped_column(Float, default=0.1)
    top_k: Mapped[int] = mapped_column(Integer, default=5)
    max_tokens: Mapped[int] = mapped_column(Integer, default=40)
    token_events: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, server_default=func.now())


async def init_db(database_url: str | None = None) -> bool:
    """Initialize async engine + tables. Returns True when DB is enabled."""
    global _engine, _session_factory, _database_url

    resolved_url = normalize_database_url(database_url or os.getenv("DATABASE_URL"))
    if resolved_url is None:
        return False

    _database_url = resolved_url
    _engine = create_async_engine(resolved_url, future=True, pool_pre_ping=True)
    _session_factory = async_sessionmaker(_engine, expire_on_commit=False)

    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return True


async def close_db() -> None:
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _session_factory = None


def is_db_enabled() -> bool:
    return _session_factory is not None


def get_database_url() -> str | None:
    return _database_url


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    if _session_factory is None:
        raise RuntimeError("Database not initialized")
    return _session_factory
