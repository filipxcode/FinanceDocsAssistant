from fastapi import FastAPI, UploadFile, BackgroundTasks, File, HTTPException, Query, Depends
from fastapi.responses import FileResponse
from pathlib import Path
import os
from src.services.RAGSerivce import RAGService
from src.config.settings import configure_settings
import shutil
from uuid import uuid4
from src.schemas.schemas import InputQuery, ResponseOutputFinal, ChatTemplate, ChatSessionOut, ChatSessionFull, ChatUpdate, ChatCreate
from cachetools import TTLCache
from enum import Enum
from contextlib import asynccontextmanager
import logging
import nest_asyncio
from src.database.db import get_session
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID
from src.database.db import create_db_and_tables
from src.services.chat_service import save_message, create_chat_session, list_chats, get_chat_history, soft_delete_chat, update_chat_title, get_chat

nest_asyncio.apply()
configure_settings()


UPLOAD_DIR = Path("files")
UPLOAD_DIR.mkdir(exist_ok=True)
JOB_STATUS = TTLCache(maxsize=1000, ttl=3600)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JobStatus(str, Enum):
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

rag_service: RAGService = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_service
    logger.info("Uruchamianie aplikacji i ładowanie modeli")
    
    configure_settings()
    await create_db_and_tables()     
    rag_service = RAGService()
    yield


app = FastAPI(lifespan=lifespan)

async def process_rag_tasks(file_paths: list[str], job_id: str):
    JOB_STATUS[job_id] = JobStatus.PROCESSING
    try:
        for p in file_paths:
            rag_service.process_file(p)
        JOB_STATUS[job_id] = JobStatus.COMPLETED
    except Exception as e:
        JOB_STATUS[job_id] = JobStatus.FAILED
        raise SystemError(f"Processing file error, job_id: {job_id}")

async def message_history_preprocessor(chat_history: ChatSessionFull) -> list[str]:
    messages = []
    for mess in chat_history:
        role_label = mess.role 
        text_content = mess.text
        entry = f"{role_label}: {text_content}"
        messages.append(entry)
    return messages

@app.get("/status")
async def check_status():
    if rag_service is None:
        return {"status": "loading"}
    return {"status": "ok"}

@app.post("/upload")
async def upload_files(files: list[UploadFile] = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    saved_paths = []
    job_id = str(uuid4())
    for file in files:
        safe_filename = f'{job_id}_{file.filename}'
        file_path = UPLOAD_DIR / safe_filename
        try: 
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_paths.append(str(file_path))
        except Exception as e:
            raise HTTPException(f"Error in saving file {file.filename}")
    background_tasks.add_task(process_rag_tasks, saved_paths, job_id)
    JOB_STATUS[job_id] = JobStatus.PROCESSING
    
    return {"message":"Processing", "job_id": job_id}

@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    status = JOB_STATUS.get(job_id)
    if not status:
        return {"status": "unknown"}
    return {"status": status}

@app.post("/query", response_model=ResponseOutputFinal)
async def query(q: InputQuery, session: AsyncSession = Depends(get_session)):
    await save_message(db=session, chat_id=q.chat_id, mess=ChatTemplate.from_input(q))
    try:
        chat_history = await get_chat_history(db=session, chat_id=q.chat_id, limit = 20)
        history_context = chat_history[:-1] 
        messages_processed = await message_history_preprocessor(history_context)
        response = await rag_service.aget_answear(q.query, messages_processed)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying LLM: {e}")
    await save_message(db=session, chat_id=q.chat_id, mess=ChatTemplate.from_response(response))
    return response

@app.get("/files/{filename}")
async def file_context(filename: str):
    safe_name = os.path.basename(filename)
    file_path = UPLOAD_DIR / safe_name
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(
        path=file_path, 
        media_type="application/pdf", 
        filename=filename,
        content_disposition_type="inline"
    )
    
@app.post("/chats")
async def create_chat(chat_data: ChatCreate, session: AsyncSession = Depends(get_session)):
    id, title = await create_chat_session(db=session, title=chat_data.title)
    return {"id": id, "title": title}

@app.get("/chats", response_model=list[ChatSessionOut])
async def list_chats_sessions(limit: int = Query(ge=1, le=40, default=10), session: AsyncSession = Depends(get_session)):
    response = await list_chats(db=session, limit=limit)
    return response

@app.get("/chats/{chat_id}", response_model=ChatSessionFull)
async def show_chat(chat_id: UUID, session: AsyncSession = Depends(get_session)):
    chat_info = await get_chat(db=session, chat_id=chat_id)
    messages = await get_chat_history(db=session, chat_id=chat_id)
    return ChatSessionFull(
        id=chat_info.id,
        created_at=chat_info.created_at,
        title=chat_info.title,
        messages=messages
    )

@app.delete("/chats/{chat_id}")
async def delete_chat(chat_id: UUID, session: AsyncSession = Depends(get_session)):
    response = await soft_delete_chat(db=session, chat_id=chat_id)
    return response

@app.patch("/chats/{chat_id}")
async def update_title(chat_id: UUID, chat_data:ChatUpdate, session: AsyncSession = Depends(get_session)):
    response = await update_chat_title(db=session, chat_id=chat_id, new_title=chat_data.title)
    return response