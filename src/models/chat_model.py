from sqlalchemy import Column, String, DateTime, ForeignKey, Numeric, Boolean
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB
from sqlalchemy.orm import relationship
from datetime import datetime
from uuid import uuid4
from src.database.db import Base

class ChatSession(Base):
    __tablename__ = "chat"
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    title = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_deleted = Column(Boolean, default=False)
    messages = relationship("Message", back_populates="chat", cascade="all, delete-orphan")

class Message(Base):
    __tablename__ = "message"
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    chat_id = Column(PG_UUID(as_uuid=True), ForeignKey("chat.id"), nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    role = Column(String(10))
    text = Column(String)
    sources = Column(JSONB, nullable=True) 
    chat = relationship("ChatSession", back_populates="messages")
    metrics = relationship("Metrics", back_populates="message", cascade="all, delete-orphan")

class Metrics(Base):
    __tablename__ = "metrics"
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    message_id = Column(PG_UUID(as_uuid=True), ForeignKey("message.id"), nullable=False)
    label = Column(String(300), nullable=True)
    amount = Column(Numeric(12,2), nullable=True)
    unit = Column(String(10), nullable=True)
    currency = Column(String(3), nullable=True)
    date = Column(String(15), nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    message = relationship("Message", back_populates="metrics")