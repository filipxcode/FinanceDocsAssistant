from llama_index.core import SimpleDirectoryReader

def fast_check_llama_native(file_path: str) -> dict:
    try:
        if not file_path.lower().endswith((".pdf", ".pptx", ".txt", ".docx")):
            return {"error": "Nieobsługiwany format pliku"}

        reader = SimpleDirectoryReader(input_files=[file_path])
        docs = reader.load_data()

        if not docs:
            return {"error": "Pusty plik lub brak warstwy tekstowej (skan?)"}

        total_text = sum(len(doc.text or "") for doc in docs)
        if total_text < 50:
            return {"error": "Plik zawiera za mało tekstu do analizy."}

        return {"parsed": "Allowed"}
    except Exception as e:
        return {"error": f"Błąd weryfikacji: {str(e)}"}
