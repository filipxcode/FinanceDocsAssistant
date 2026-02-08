import pytest
import pytest_asyncio
import os
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.pool import NullPool
from src.database.db import Base, get_session
from src.api.app import app 

from dotenv import load_dotenv

load_dotenv()


if "TEST_DATABASE_URL" in os.environ:
    TEST_DATABASE_URL = os.environ["TEST_DATABASE_URL"]
else:
    POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
    POSTGRES_DB = os.getenv("POSTGRES_DB_TEST", "smartdocs_test")
    TEST_DATABASE_URL = f"postgresql+asyncpg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

@pytest_asyncio.fixture(scope="function")
async def async_db_engine():
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
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac

    app.dependency_overrides = {}