import pytest
import pytest_asyncio
import os
import asyncpg
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.pool import NullPool
from sqlalchemy.engine import make_url
from src.database.db import Base, get_session

from dotenv import load_dotenv

load_dotenv()

# The app config is evaluated at import time (configure_settings()),
# so set required env vars BEFORE importing src.api.app.
os.environ.setdefault("OPENAI_API_KEY", "test")
os.environ.setdefault("EMBEDDINGS_MODEL", "text-embedding-3-small")
os.environ.setdefault("EMBEDDINGS_DIM", "1536")

os.environ.setdefault("LLM_PROVIDER_QUERY", "groq")
os.environ.setdefault("LLM_PROVIDER_SYNTHESIS", "groq")
os.environ.setdefault("GROQ_API_KEY", "test")

os.environ.setdefault("DEMO_PASSWORD", "test")

from src.api.app import app


if "TEST_DATABASE_URL" in os.environ:
    TEST_DATABASE_URL = os.environ["TEST_DATABASE_URL"]
else:
    POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
    POSTGRES_DB = os.getenv("POSTGRES_DB_TEST", "smartdocs_test")
    TEST_DATABASE_URL = f"postgresql+asyncpg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"


async def _ensure_database_exists(database_url: str) -> None:
    """Create the target database if it doesn't exist.

    This makes the test suite more robust on fresh machines/CI.
    """

    url = make_url(database_url)
    db_name = url.database
    if not db_name or db_name in {"postgres", "template1"}:
        return

    admin_url = url.set(database="postgres")

    conn = await asyncpg.connect(
        user=admin_url.username,
        password=admin_url.password,
        host=admin_url.host,
        port=admin_url.port,
        database=admin_url.database,
    )
    try:
        exists = await conn.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1",
            db_name,
        )
        if not exists:
            # Identifier quoting: double quotes; db_name is from config/env.
            await conn.execute(f'CREATE DATABASE "{db_name}"')
    finally:
        await conn.close()

@pytest_asyncio.fixture(scope="function")
async def async_db_engine():
    await _ensure_database_exists(TEST_DATABASE_URL)
    engine = create_async_engine(
        TEST_DATABASE_URL,
        poolclass=NullPool, 
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()

@pytest_asyncio.fixture(scope="function")
async def db_session(async_db_engine):
    async_session = async_sessionmaker(
        async_db_engine, class_=AsyncSession, expire_on_commit=False
    )

    async with async_session() as session:
        yield session
        
@pytest.fixture(scope="function")
def override_get_session(db_session):
    async def _override_get_session():
        yield db_session
    return _override_get_session

@pytest_asyncio.fixture(scope="function")
async def client(db_session):
    async def override_get_session():
        yield db_session

    app.dependency_overrides[get_session] = override_get_session
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        headers={"X-Demo-Password": os.environ["DEMO_PASSWORD"]},
    ) as ac:
        yield ac

    app.dependency_overrides = {}