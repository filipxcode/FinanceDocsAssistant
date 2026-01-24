from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from src.models.chat_model import ChatSession, Message, Metrics as DbMetrics
from src.schemas.schemas import ChatTemplate
from uuid import UUID
from fastapi import HTTPException, status
from datetime import datetime

async def get_chat(db: AsyncSession, chat_id: UUID) -> ChatSession:
    res = await db.execute(select(ChatSession).where(ChatSession.id == chat_id, ChatSession.is_deleted == False))
    chat = res.scalar_one_or_none()
    if not chat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail=f"Chat not found"
        )
    return chat

async def create_chat_session(db: AsyncSession, title: str| None) -> tuple[UUID, str]:
    if not title:
        title=f"Rozmowa {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}"
    new_chat = ChatSession(title=title)
    db.add(new_chat)
    
    await db.commit()
    await db.refresh(new_chat)
    
    return new_chat.id, new_chat.title

async def save_message(db: AsyncSession, chat_id: UUID, mess: ChatTemplate) -> Message:
    sources_json = [s.model_dump() for s in mess.sources] if mess.sources else None
    await get_chat(db=db, chat_id=chat_id)
    
    mess_db = Message(chat_id=chat_id, role=mess.role, text=mess.text, sources=sources_json) 
    db.add(mess_db)
    await db.flush()
    
    if mess.metrics:
        for m in mess.metrics:
            metrics = m.model_dump(exclude_defaults=True)
            db.add(DbMetrics(message_id=mess_db.id, **metrics))
    await db.commit()
    await db.refresh(mess_db)
    return mess_db

async def list_chats(db: AsyncSession, limit = 10) -> list[ChatSession]:
    chats = await db.execute(select(ChatSession).where(ChatSession.is_deleted==False).order_by(ChatSession.created_at.desc()).limit(limit))
    return list(chats.scalars().all())

async def get_chat_history(db: AsyncSession, chat_id: UUID, limit: int | None = None) -> list[Message]: 
    query = (
        select(Message)
        .where(Message.chat_id == chat_id)
        .options(selectinload(Message.metrics))
        .order_by(Message.created_at.asc())      
    )
    
    if limit:
        query = (
            select(Message)
            .where(Message.chat_id == chat_id)
            .options(selectinload(Message.metrics))
            .order_by(Message.created_at.desc())
            .limit(limit)
        )
        res = await db.execute(query)
        messages = res.scalars().all()
        return list(reversed(messages))
        
    res = await db.execute(query)
    messages = res.scalars().all()
    return list(messages)

async def update_chat_title(db: AsyncSession, chat_id: UUID, new_title: str) -> ChatSession:
    chat = await get_chat(db, chat_id)
    chat.title = new_title
    await db.commit()
    return chat

async def soft_delete_chat(db: AsyncSession, chat_id: UUID) -> bool:
    chat = await get_chat(db=db, chat_id=chat_id)
    chat.is_deleted = True
    await db.commit()
    return True