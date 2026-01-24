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

def get_llm():
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    
    if provider == "ollama":
        return Ollama(
            model="llama3.1", 
            request_timeout=360.0, 
            context_window=8192, 
            temperature=0.1, 
            top_p=0.1
        )
    
    elif provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Wybrano providera GROQ, ale brak GROQ_API_KEY")
        return Groq(
            model="llama-3.1-8b-instant",
            api_key=api_key,
            temperature=0.1
        )
    else:
        raise ValueError(f"Unknown provider")
    
def configure_settings():
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-m3"
    )
    
    Settings.llm = Ollama(model="llama3.1", request_timeout=360.0, context_window=8192, temperature=0.1, top_p=0.1)
    
    Settings.node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )
    Settings.callback_manager = callback_manager
    