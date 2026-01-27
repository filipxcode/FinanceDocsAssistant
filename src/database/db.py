from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from typing import AsyncGenerator
from sqlalchemy.orm import DeclarativeBase 
from dotenv import load_dotenv
import os

load_dotenv()

user = os.getenv("POSTGRES_USER", "postgres")
password = os.getenv("POSTGRES_PASSWORD", "postgres")
host = os.getenv("POSTGRES_HOST", "localhost")
port = os.getenv("POSTGRES_PORT", "5432")
db_name = os.getenv("POSTGRES_DB", "vector_db")

database_url = f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{db_name}"

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