import re
from llama_index.core import SimpleDirectoryReader, Settings
from src.config.settings import get_settings

def fast_check_llama_native(file_path: str) -> dict:
    try:
        settings = get_settings()
        POLISH_STOP_WORDS = {"się", "jest", "dla", "oraz", "przez", "spółka", "zł", "tys.", "r/r"}
        if not file_path.lower().endswith((".pdf",".pptx",".txt",".docx")):
            return {"error":"Nieobsługiwany format pliku"}
            
        reader = SimpleDirectoryReader(input_files=[file_path])
        docs = reader.load_data()
        
        if not docs:
            return {"error":"Pusty plik lub brak warstwy tekstowej (skan?)"}
            

        meaningful_text = ""
        target_length = 1000  
        for doc in docs:
            page_text = doc.text or ""
            
            letters_only = re.sub(r'[^a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ]', '', page_text)
            
            if len(letters_only) < 15:
                continue
                
            meaningful_text += page_text + " "
            
            if len(meaningful_text) > target_length * 2:
                break
        
        if len(meaningful_text) < 20:
            meaningful_text = " ".join([d.text for d in docs[:10]])

        clean_text = re.sub(r'\d+', ' ', meaningful_text)
        clean_text = re.sub(r'[^\w\s\%\$€zł]', ' ', clean_text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        text_to_analyze = clean_text[:2000] # Maksymalnie 2000 znaków do analizy

        if not text_to_analyze:
            return {"error": "Plik zawiera same obrazy lub liczby (brak zdań)."}

        words_set = set(text_to_analyze.lower().split())
        common_pl = words_set.intersection(POLISH_STOP_WORDS)
        
        if len(common_pl) >= 2 and "POLISH" in settings.LANGUAGES_ALLOWED:
            return {"parsed": "Allowed"}

        confidence_values = Settings.language_detector.compute_language_confidence_values(text_to_analyze)
        
        if not confidence_values:
            return {"error": "Nie udało się wykryć języka"}

        top_res = confidence_values[0]
        
        if top_res.language.name in settings.LANGUAGES_ALLOWED and top_res.value > 0.05: # Obniżony próg
            return {"parsed": "Allowed"}
            
        for res in confidence_values:
            if res.language.name in settings.LANGUAGES_ALLOWED and res.value > 0.15:
                return {"parsed": "Allowed"}

        return {"error": f"Nierozpoznany język. Wykryto: {top_res.language.name} ({top_res.value:.2f})"}

    except Exception as e:
        return {"error": f"Błąd weryfikacji: {str(e)}"}