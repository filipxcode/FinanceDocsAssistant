from fastapi import FastAPI, UploadFile, BackgroundTasks, File, HTTPException, Query, Depends, Header, Request
from fastapi.responses import FileResponse
from pathlib import Path
import os
from src.services.RAGSerivce import RAGService
from src.config.settings import configure_settings, get_settings
import shutil
from uuid import uuid4
from src.schemas.schemas import InputQuery, ResponseOutputFinal, ChatTemplate, ChatSessionOut, ChatSessionFull, ChatUpdate, ChatCreate, DocumentOut
from cachetools import TTLCache
from enum import Enum
from contextlib import asynccontextmanager
import logging
import nest_asyncio
from src.database.db import get_session, async_session_maker
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID
from src.database.db import create_db_and_tables
from src.services.chat_service import save_message, create_chat_session, list_chats, get_chat_history, soft_delete_chat, update_chat_title, get_chat
from src.services.document_service import register_file, list_doc_filenames, delete_full_doc
from src.services.language_gate import fast_check_llama_native
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.responses import JSONResponse
from slowapi.middleware import SlowAPIMiddleware
import secrets

nest_asyncio.apply()
configure_settings()
settings = get_settings()

UPLOAD_DIR = settings.UPLOAD_DIR
UPLOAD_DIR.mkdir(exist_ok=True)
MAX_UPLOAD_FILE_SIZE_MB = settings.MAX_UPLOAD_FILE_SIZE_MB
MAX_UPLOAD_FILE_SIZE_BYTES = MAX_UPLOAD_FILE_SIZE_MB * 1024 * 1024
MAX_UPLOAD_FILES = settings.MAX_UPLOAD_FILES
MAX_TOTAL_UPLOAD_SIZE_BYTES = settings.MAX_TOTAL_UPLOAD_SIZE_MB * 1024 * 1024
JOB_STATUS = TTLCache(maxsize=1000, ttl=3600)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JobStatus(str, Enum):
    """Enum for background tasking handling"""
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

rag_service: RAGService = None

def verify_demo_password(request: Request, x_demo_password: str | None = Header(default=None, alias="X-Demo-Password")):
    expected = (settings.demo_password or "").strip()
    provided = (x_demo_password or "").strip()
    
    if request.url.path == "/status":
        return True
    # Fail closed if demo password is not configured
    if not expected:
        raise HTTPException(status_code=503, detail="Demo access is not configured.")

    if not secrets.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return True

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Life span function to initialize app"""
    global rag_service
    logger.info("Uruchamianie aplikacji i ładowanie modeli")
    
    configure_settings()
    await create_db_and_tables()     
    rag_service = RAGService()
    yield


docs_url = None if settings.DEMO_DISABLE_DOCS else "/docs"
redoc_url = None if settings.DEMO_DISABLE_DOCS else "/redoc"
openapi_url = None if settings.DEMO_DISABLE_DOCS else "/openapi.json"

app = FastAPI(
    lifespan=lifespan,
    dependencies=[Depends(verify_demo_password)],
    docs_url=docs_url,
    redoc_url=redoc_url,
    openapi_url=openapi_url,
)

limiter = Limiter(key_func=get_remote_address, default_limits=[settings.RATE_LIMIT_DEFAULT])
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)
app.add_exception_handler(
    RateLimitExceeded,
    lambda request, exc: JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Please try again later."},
    ),
)

async def process_rag_tasks(file_paths: list[str], original_paths: list[str], file_sizes: list[int], job_id: str):
    """Function for processing documents which handling the file uploading to rag service layer and file registration to DB."""
    JOB_STATUS[job_id] = JobStatus.PROCESSING
    try:
        async with async_session_maker() as session:
            for p, o, s in zip(file_paths, original_paths, file_sizes):
                file_obj = await register_file(db=session, original_filename=o, filename=p, size_bytes=s)
                await rag_service.process_file(file_path=p, file_id=str(file_obj.id), original_path = o)
        JOB_STATUS[job_id] = JobStatus.COMPLETED
    except Exception as e:
        JOB_STATUS[job_id] = JobStatus.FAILED
        logger.error(f"Processing file error, job_id: {job_id}, error: {e}")

async def message_history_preprocessor(chat_history: ChatSessionFull) -> list[str]:
    """Preprocesses chat history into list of strings"""
    messages = []
    for mess in chat_history:
        role_label = mess.role 
        text_content = mess.text
        entry = f"{role_label}: {text_content}"
        messages.append(entry)
    return messages

@app.get("/status", dependencies=[])
async def check_status():
    """Checks RAG service loading status"""
    if rag_service is None:
        return {"status": "loading"}
    return {"status": "ok"}

@app.post("/upload")
@limiter.limit(settings.RATE_LIMIT_UPLOAD)
async def upload_files(request: Request, files: list[UploadFile] = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    """Uploads files and starts background processing task"""
    saved_paths = []
    file_sizes = []
    original_paths = []
    errors = []
    job_id = str(uuid4())

    if len(files) > MAX_UPLOAD_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files. Max allowed per request: {MAX_UPLOAD_FILES}.",
        )

    total_written_all = 0
    for file in files:
        filename = f"{uuid4()}_{file.filename}" 
        original_filename = file.filename
        file_path = UPLOAD_DIR / filename
        try: 
            with open(file_path, "wb") as buffer:
                total_written = 0
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        break
                    total_written += len(chunk)
                    if total_written > MAX_UPLOAD_FILE_SIZE_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail=f"Plik {original_filename} przekracza limit {MAX_UPLOAD_FILE_SIZE_MB} MB.",
                        )
                    total_written_all += len(chunk)
                    if total_written_all > MAX_TOTAL_UPLOAD_SIZE_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail=f"Łączny rozmiar uploadu przekracza limit {settings.MAX_TOTAL_UPLOAD_SIZE_MB} MB.",
                        )
                    buffer.write(chunk)
            
            check_result = fast_check_llama_native(str(file_path))
            if "error" in check_result:
                os.remove(file_path)
                errors.append(f"File {original_filename} rejected: {check_result['error']}")
                continue
            
            saved_paths.append(str(file_path))
            file_sizes.append(int(total_written))
            original_paths.append(str(original_filename))
        except HTTPException as e:
            if os.path.exists(file_path):
                os.remove(file_path)
            errors.append(e.detail)
            continue
        except Exception as e:
            raise HTTPException(f"Error in saving file {file.filename}")
        finally:
            try:
                await file.close()
            except Exception:
                pass

    if not saved_paths:
        JOB_STATUS[job_id] = JobStatus.FAILED
        return {"message": "No files accepted", "job_id": job_id, "errors": errors}

    background_tasks.add_task(process_rag_tasks, saved_paths, original_paths, file_sizes, job_id)
    JOB_STATUS[job_id] = JobStatus.PROCESSING
    response = {"message":"Processing", "job_id": job_id}
    if errors:
        response["errors"] = errors
    return response

@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Returns status of the file processing job"""
    status = JOB_STATUS.get(job_id)
    if not status:
        return {"status": "unknown"}
    return {"status": status}

@app.post("/query", response_model=ResponseOutputFinal)
@limiter.limit(settings.RATE_LIMIT_QUERY)
async def query(request: Request, q: InputQuery, session: AsyncSession = Depends(get_session)):
    """Main RAG query endpoint"""
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
    """Serves uploaded PDF files"""
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
    """Creates a new chat session"""
    id, title = await create_chat_session(db=session, title=chat_data.title)
    return {"id": id, "title": title}

@app.get("/chats", response_model=list[ChatSessionOut])
async def list_chats_sessions(limit: int = Query(ge=1, le=40, default=10), session: AsyncSession = Depends(get_session)):
    """Lists active chat sessions"""
    response = await list_chats(db=session, limit=limit)
    return response

@app.get("/chats/{chat_id}", response_model=ChatSessionFull)
async def show_chat(chat_id: UUID, session: AsyncSession = Depends(get_session)):
    """Returns full chat history"""
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
    """Soft deletes a chat session"""
    response = await soft_delete_chat(db=session, chat_id=chat_id)
    return response

@app.patch("/chats/{chat_id}")
async def update_title(chat_id: UUID, chat_data:ChatUpdate, session: AsyncSession = Depends(get_session)):
    """Updates chat session title"""
    response = await update_chat_title(db=session, chat_id=chat_id, new_title=chat_data.title)
    return response

@app.get("/documents", response_model=list[DocumentOut])
async def list_documents(limit: int = Query(ge=1, default=20), session: AsyncSession = Depends(get_session)):
    """Lists processed documents"""
    response = await list_doc_filenames(db=session, limit=limit)
    return response

@app.delete("/documents/{document_id}")
async def delete_document(document_id: UUID,session: AsyncSession = Depends(get_session)):
    """Deletes document and vectors"""
    response = await delete_full_doc(db=session, id=document_id, upload_dir=UPLOAD_DIR)
    return response
