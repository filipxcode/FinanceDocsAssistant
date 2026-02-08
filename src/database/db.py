from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from typing import AsyncGenerator
from sqlalchemy.orm import DeclarativeBase 
from src.config.settings import get_settings

settings = get_settings()

database_url = settings.database_url_async

engine = create_async_engine(database_url, echo=True, future=True)
async_session_maker = async_sessionmaker(engine,  expire_on_commit=False)

class Base(DeclarativeBase):
    pass

async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for getting async database session"""
    async with async_session_maker() as session:
        yield session

async def create_db_and_tables():
    """Creates database tables based on SQLAlchemy models"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)