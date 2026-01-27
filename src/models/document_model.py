from sqlalchemy import Column, String, DateTime, Integer
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB
from datetime import datetime
from src.database.db import Base
from uuid import uuid4

class Document(Base):
    __tablename__ = "raporty_finansowe_hybrid"
    id = Column(PG_UUID(as_uuid=True), primary_key=True)
    metadata_ = Column("metadata_", JSONB)
    
class UploadedFile(Base):
    __tablename__ = "uploaded_files"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4())
    filename = Column(String, nullable=False) 
    original_filename = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow) 
    size_bytes = Column(Integer) 

