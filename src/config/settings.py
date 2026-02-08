from pydantic import BaseModel, Field
from functools import lru_cache
from pathlib import Path
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.llms.groq import Groq
from llama_index.core.node_parser import SentenceWindowNodeParser
from dotenv import load_dotenv
import os
import sys
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
import logging

load_dotenv()
llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger("llama_index.core.retrievers").setLevel(logging.DEBUG)

class AppSettings(BaseModel):
    # App Config
    UPLOAD_DIR: Path = Path("files")
    
    # LLM Config
    LLM_PROVIDER: str = Field(default_factory=lambda: os.getenv("LLM_PROVIDER", "ollama").lower())
    GROQ_API_KEY: str | None = Field(default_factory=lambda: os.getenv("GROQ_API_KEY"))
    
    # Database Config
    POSTGRES_USER: str = Field(default_factory=lambda: os.getenv("POSTGRES_USER", "postgres"))
    POSTGRES_PASSWORD: str = Field(default_factory=lambda: os.getenv("POSTGRES_PASSWORD", "postgres"))
    POSTGRES_HOST: str = Field(default_factory=lambda: os.getenv("POSTGRES_HOST", "localhost"))
    POSTGRES_PORT: str = Field(default_factory=lambda: os.getenv("POSTGRES_PORT", "5432"))
    POSTGRES_DB: str = Field(default_factory=lambda: os.getenv("POSTGRES_DB", "vector_db"))
    
    # Vector Helper
    VECTOR_TABLE_NAME: str = "raporty_finansowe_hybrid"

    @property
    def database_url_async(self) -> str:
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    @property
    def database_url_sync(self) -> str:
        return f"postgresql+psycopg2://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    class Config:
        arbitrary_types_allowed = True

@lru_cache()
def get_settings() -> AppSettings:
    return AppSettings()

@lru_cache()
def get_query_llm(provider: str, api_key: str | None = None):
    """Initializes and returns the LLM instance based on provider, cached."""
    
    if provider == "ollama":
        return Ollama(
            model="llama3.1", 
            request_timeout=360.0, 
            context_window=8192, 
            temperature=0.1, 
            top_p=0.1
        )
    
    elif provider == "groq":
        if not api_key:
            raise ValueError("Wybrano providera GROQ, ale brak GROQ_API_KEY")
        return Groq(
            model="llama-3.1-8b-instant",
            api_key=api_key,
            temperature=0.1
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

@lru_cache()
def get_synthesis_llm(api_key: str | None = None):
    if not api_key:
        pass
        
    return Groq(
            model="llama-3.3-70b-versatile",
            api_key=api_key,
            temperature=0.1
        )
    
def configure_settings():
    """Configures global LlamaIndex settings"""
    settings = get_settings()
    
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-m3"
    )
    
    # Pass necessary args to helper functions
    Settings.query_llm = get_query_llm(settings.LLM_PROVIDER, settings.GROQ_API_KEY)
    Settings.synthesis_llm = get_synthesis_llm(settings.GROQ_API_KEY)
    
    Settings.node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=5,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )
    Settings.callback_manager = callback_manager
    