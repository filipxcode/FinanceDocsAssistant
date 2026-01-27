from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core import VectorStoreIndex
from src.services.parser import parse_document
import os
from src.config.settings import configure_settings
from src.services.FinQueryEngine import FinancialEngineBuilder
from llama_index.core import Settings
import logging
from datetime import datetime
from uuid import uuid4
import asyncio

logger = logging.getLogger(__name__)

class RAGService:
    """Service for RAG operations and vector store management"""
    def __init__(self):
        self.index = None
        self._initialize_index()
        self.engine = FinancialEngineBuilder(self.index)
        
    def _initialize_index(self):
        """Initializes PostgreSQL vector store connection"""
        user = os.getenv("POSTGRES_USER")
        password = os.getenv("POSTGRES_PASSWORD")
        host = os.getenv("POSTGRES_HOST")
        port = os.getenv("POSTGRES_PORT")
        db_name = os.getenv("POSTGRES_DB")

        connection_string = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db_name}"
        async_connection_string = f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{db_name}"
        
        vector_store = PGVectorStore(
            connection_string=connection_string,
            async_connection_string=async_connection_string, 
            table_name="raporty_finansowe_hybrid", 
            embed_dim=1024,
            hybrid_search=True,
            text_search_config="simple"
        )

        self.index = VectorStoreIndex.from_vector_store(vector_store)
    
    async def process_file(self, file_path: str, file_id: str, original_path: str, meta_info: dict = None) -> bool:
        """Sending to parsing helper function, processes uploaded file, ingests into vector store"""
        logger.info(f"[RAGService] Rozpoczynam przetwarzanie pliku: {file_path}")
        documents = await parse_document(file_path)
        logger.info(f"[RAGService] Pobrarno {len(documents)} dokumentów z LlamaParse.")
        for i, doc in enumerate(documents):
            doc.id_ = file_id
            doc.metadata["filename"] = original_path
            doc.metadata["file_id"] = file_id   
            page_number = i + 1
            doc.metadata["page_label"] = str(page_number)
            if meta_info:
                doc.metadata.update(meta_info)
            doc.excluded_embed_metadata_keys = ["page_label", "filename", "file_id"]
        nodes = Settings.node_parser.get_nodes_from_documents(documents)
        logger.info(f"[RAGService] Dokument podzielony na {len(nodes)} fragmentów (chunks).")
        
        for node in nodes:
            node.excluded_embed_metadata_keys = [
                "window",       
                "original_text", 
                "filename", 
                "page_label",
                "file_id"
            ]
            node.excluded_llm_metadata_keys = ["original_text"]
            
        logger.info(f"[RAGService] Zapisywanie węzłów do bazy wektorowej...")
        await asyncio.to_thread(self.index.insert_nodes, nodes)
        
        logger.info(f"[RAGService] Plik {original_path} przetworzony pomyślnie.")
        return True
    
    async def aget_answear(self, query: str, chat_history: list[str]):
        """Generates answer using RAG engine"""
        logger.info(f"[RAGService] Otrzymano zapytanie (ASYNC): '{query}'")
        return await self.engine.query_async(query, chat_history)
