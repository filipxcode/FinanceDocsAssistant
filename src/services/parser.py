from llama_cloud_services import LlamaParse
from llama_index.core.schema import Document
import os
import logging
logger = logging.getLogger(__name__)
async def parse_document(filepath: list[str] | str, lang: str = "pl", no_photos: bool = True, target_p: str | None = None, max_p: int | None = None) -> list[Document]:
    """Parses documents using LlamaParse API"""
    parser = LlamaParse(
            api_key=os.getenv("API_KEY"),
            language=lang,
            num_workers=4,
            verbose=True,
            disable_ocr=no_photos,
            skip_diagonal_text=True,
            target_pages=target_p,
            max_pages=max_p,
            result_type="markdown",  
            premium_mode=True,       
            parsing_instruction="To jest raport finansowy. Zachowaj wszystkie tabele w formacie Markdown. Nie pomijaj żadnych liczb."
        )
    try:
        docs = await parser.aload_data(filepath) 
    except Exception as e:
        raise RuntimeError(f"LLamaindex API error: {e}")
    logger.info("Parsed!!!!!!!")
    return docs

