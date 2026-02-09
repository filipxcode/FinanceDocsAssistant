from llama_index.core import SimpleDirectoryReader, Settings
from src.config.settings import get_settings

def fast_check_llama_native(file_path: str) -> dict:
    try:
        settings = get_settings()
        if not file_path.lower().endswith((".pdf",".pptx",".txt",".docx")):
            return {"error":"Extension not allowed"}
        reader = SimpleDirectoryReader(input_files=[file_path])
        docs = reader.load_data()
        
        if not docs:
            return {"error":"Failed to parse"} 
            
        sample_text = docs[0].text[:2000]
        detected = Settings.detector.detect_language_of(sample_text)
        if detected is None:
            return {"error": "Language detection failed (unknown content)"}
        if detected.name in settings.LANGUAGES_ALLOWED:
            return {"parsed":"Allowed"} 
        return {"error":"Extension not allowed"}
    except Exception as e:
        return {"error":"Failed to parse"}  