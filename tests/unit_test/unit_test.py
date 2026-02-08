import pytest
import uuid
from unittest.mock import patch, AsyncMock
from src.models.document_model import UploadedFile
from datetime import datetime

import pytest_asyncio

# Fixture to seed a file into the DB directly
@pytest_asyncio.fixture
async def seeded_file(db_session):
    file_id = uuid.uuid4()
    new_file = UploadedFile(
        id=file_id,
        filename=f"{file_id}_test.pdf",
        original_filename="test_report.pdf",
        size_bytes=1024,
        created_at=datetime.utcnow()
    )
    db_session.add(new_file)
    await db_session.commit()
    return new_file

@pytest.mark.asyncio
async def test_list_documents_seeded(client, seeded_file):
    res = await client.get("/documents")
    assert res.status_code == 200
    data = res.json()
    assert len(data) >= 1
    found = any(d['id'] == str(seeded_file.id) for d in data)
    assert found

@pytest.mark.asyncio
async def test_delete_document_seeded(client, seeded_file):
    res = await client.delete(f"/documents/{seeded_file.id}")
    assert res.status_code == 200
    
    res_list = await client.get("/documents")
    data = res_list.json()
    found = any(d['id'] == str(seeded_file.id) for d in data)
    assert not found

@pytest.mark.asyncio
async def test_query_empty(client):
    query_id = str(uuid.uuid4())
    res = await client.post(f"/query", json={"query":"", "chat_id": query_id})
    assert res.status_code == 422

@pytest.mark.asyncio
async def test_list_chat_negative(client):
    res1 = await client.post(f"/chats", json={"title": "Chat 1"})
    res2 = await client.post(f"/chats", json={"title": "Chat 2"})
    assert res1.status_code == 200 and res2.status_code == 200
    res = await client.get(f"/chats", params={"limit":-1})
    assert res.status_code == 422

@pytest.mark.asyncio
async def test_list_chat_big_amount(client):
    res1 = await client.post(f"/chats", json={"title": "Chat 1"})
    res2 = await client.post(f"/chats", json={"title": "Chat 2"})
    res = await client.get(f"/chats", params={"limit":100000000})

    assert res.status_code == 422

@pytest.mark.asyncio
async def test_show_chat_random(client):
    random_id = uuid.uuid4()
    res1 = await client.get(f"/chats/{random_id}")
    assert res1.status_code == 404

@pytest.mark.asyncio
async def test_delete_chat(client):
    res1 = await client.post(f"/chats", json={"title": "Chat to Delete"})
    assert res1.status_code == 200
    data = res1.json()
    chat_id = data["id"]
    res = await client.delete(f"/chats/{chat_id}")
    assert res.status_code == 200

@pytest.mark.asyncio
async def test_delete_chat_random(client):
    random_id = uuid.uuid4()
    res = await client.delete(f"/chats/{random_id}")
    assert res.status_code == 404

@pytest.mark.asyncio
async def test_update_chat_title(client):
    # 1. Create chat
    res_create = await client.post("/chats", json={"title": "Old Title"})
    assert res_create.status_code == 200
    chat_id = res_create.json()["id"]

    # 2. Update title
    new_title = "New Title Updated"
    res_update = await client.patch(f"/chats/{chat_id}", json={"title": new_title})
    assert res_update.status_code == 200
    
    # 3. Verify update
    res_get = await client.get(f"/chats/{chat_id}")
    assert res_get.status_code == 200
    assert res_get.json()["title"] == new_title

@pytest.mark.asyncio
async def test_upload_file_mocked(client):
    file_content = b"Fake PDF content"
    files = {'files': ('test.pdf', file_content, 'application/pdf')}
    
    with patch("src.api.app.rag_service") as mock_service:
        mock_service.process_file = AsyncMock(return_value=True)
        with patch("src.api.app.shutil.copyfileobj"):
            with patch("builtins.open"):
                response = await client.post("/upload", files=files)
                
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert data["message"] == "Processing"

