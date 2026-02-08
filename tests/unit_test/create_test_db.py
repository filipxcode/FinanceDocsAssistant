import asyncio
import asyncpg
import os
from dotenv import load_dotenv

load_dotenv()

async def create_db():
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "postgres")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db_name = os.getenv("POSTGRES_DB_TEST", "smartdocs_test")

    try:
        # Connect to default postgres database to run Create DB command
        conn = await asyncpg.connect(user=user, password=password, host=host, port=port, database="postgres")
        
        # Check if exists
        exists = await conn.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", db_name)
        if not exists:
            print(f"Creating database {db_name}...")
            await conn.execute(f'CREATE DATABASE "{db_name}"')
            print(f"Database {db_name} created successfully.")
        else:
            print(f"Database {db_name} already exists.")
            
        await conn.close()
    except Exception as e:
        print(f"Failed to create database: {e}")

if __name__ == "__main__":
    asyncio.run(create_db())
