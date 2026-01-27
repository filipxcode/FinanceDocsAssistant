from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import delete, select
from src.models.document_model import UploadedFile, Document
from uuid import UUID
from fastapi import HTTPException, status
from datetime import datetime
import os
async def register_file(db: AsyncSession, original_filename: str, filename: str, size_bytes: int) -> UploadedFile:
    """Registers uploaded file in the database"""
    new_file_record = UploadedFile(filename=filename, original_filename=original_filename, size_bytes=size_bytes)
    db.add(new_file_record)
    await db.commit()
    await db.refresh(new_file_record)
    return new_file_record

async def list_doc_filenames(db: AsyncSession, limit: int = 20) -> list[UploadedFile]:
    """Lists files registered in the database"""
    result = await db.execute(select(UploadedFile).order_by(UploadedFile.created_at.desc()).limit(limit))
    return result.scalars().all()


async def delete_full_doc(db: AsyncSession, id: UUID, upload_dir: str) -> dict:
    """Deletes document and its associated vector data, deleting document from upload directory"""
    try:
        # Retrieve the filename before deletion
        result = await db.execute(select(UploadedFile).where(UploadedFile.id == id))
        file_record = result.scalars().first()
        filename_to_delete = file_record.filename if file_record else None

        res_vector = await db.execute(delete(Document).where(Document.metadata_['file_id'].astext == str(id)))
        res_db = await db.execute(delete(UploadedFile).where(UploadedFile.id == id))
        await db.commit()

        # Delete file from filesystem if it existed in DB
        if filename_to_delete:
            clean_filename = os.path.basename(filename_to_delete)
            file_path = os.path.join(str(upload_dir), clean_filename)
            if os.path.exists(file_path):
                os.remove(file_path)

        return {
            "deleted_files": res_db.rowcount,
            "deleted_chunks": res_vector.rowcount
        }
    except Exception as e:
        await db.rollback() 
        raise e