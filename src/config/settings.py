from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from pathlib import Path
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import SentenceWindowNodeParser
import sys
import logging
from lingua import Language, LanguageDetectorBuilder

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger("llama_index.core.retrievers").setLevel(logging.DEBUG)

class AppSettings(BaseSettings):
    # App Config
    UPLOAD_DIR: Path = Path("files")
    MAX_UPLOAD_FILE_SIZE_MB: int = 100
    
    # LLM Config
    LLM_PROVIDER_QUERY: str = "groq"
    LLM_PROVIDER_SYNTHESIS: str = "openai"
    GROQ_API_KEY: str | None = None
    GROQ_MODEL_QUERY: str = "llama-3.1-8b-instant"
    GROQ_MODEL_SYNTHESIS: str = "llama-3.3-70b-versatile"
    LLAMAPARSE_API_KEY: str | None = None
    OPENAI_API_KEY: str | None = None
    OPENAI_MODEL_QUERY: str = "o3-mini"
    OPENAI_MODEL_SYNTHESIS: str = "o3-mini"
    # Database Config
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: str = "5432"
    POSTGRES_DB: str = "vector_db"
    
    # Vector Helper
    VECTOR_TABLE_NAME: str = "raporty_finansowe_hybrid"

    #Languages allowed
    LANGUAGES_ALLOWED: list[str] = ["POLISH"]

    #Demo access credential
    DEMO_PASSWORD: str = ""
    DEMO_PASS: str =""
    
    model_config = SettingsConfigDict(
        env_file=".env", 
        extra="ignore" 
    )

    @property
    def database_url_async(self) -> str:
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    @property
    def database_url_sync(self) -> str:
        return f"postgresql+psycopg2://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    @property
    def demo_password(self) -> str:
        return self.DEMO_PASSWORD or self.DEMO_PASS

@lru_cache()
def get_settings() -> AppSettings:
    return AppSettings()

@lru_cache()
def get_query_llm():
    settings = get_settings()
    provider = (settings.LLM_PROVIDER_QUERY or "").lower()

    if provider == "groq":
        if not settings.GROQ_API_KEY:
            raise ValueError("Choosed groq, but no GROQ_API_KEY")
        return Groq(model=settings.GROQ_MODEL_QUERY, api_key=settings.GROQ_API_KEY, temperature=0.1)

    if provider == "openai":
        if not settings.OPENAI_API_KEY:
            raise ValueError("Choosed openai, but no OPENAI_API_KEY")
        return OpenAI(model=settings.OPENAI_MODEL_QUERY, api_key=settings.OPENAI_API_KEY, temperature=0.1)

    raise ValueError(f"Unknown provider: {provider}. Use 'groq' or 'openai'.")

@lru_cache()
def get_synthesis_llm():
    settings = get_settings()
    provider = (settings.LLM_PROVIDER_SYNTHESIS or "").lower()

    if provider == "groq":
        if not settings.GROQ_API_KEY:
            raise ValueError("Choosed groq, but no GROQ_API_KEY")
        return Groq(model=settings.GROQ_MODEL_SYNTHESIS, api_key=settings.GROQ_API_KEY, temperature=0.1)

    if provider == "openai":
        if not settings.OPENAI_API_KEY:
            raise ValueError("Choosed openai, but no OPENAI_API_KEY")
        return OpenAI(model=settings.OPENAI_MODEL_SYNTHESIS, api_key=settings.OPENAI_API_KEY, temperature=0.1)

    raise ValueError(f"Unknown provider: {provider}. Use 'groq' or 'openai'.")
    
def configure_settings():
    """Configures global LlamaIndex settings"""
    settings = get_settings()
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-m3"
    )
    
    # Pass necessary args to helper functions
    Settings.query_llm = get_query_llm()
    Settings.synthesis_llm = get_synthesis_llm()
    
    Settings.node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=5,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )
    Settings.language_detector = LanguageDetectorBuilder.from_languages(
        Language.POLISH,
        Language.ENGLISH,
    ).build() 