# FinanceDocsAssistant

Asystent RAG do analizy dokumentów finansowych (PL/EN) oparty na **LlamaIndex + PGVector + FastAPI + Streamlit**.

## Opis

Aplikacja umożliwia wgrywanie dokumentów finansowych (PDF, PPTX, DOCX, TXT), ich automatyczne parsowanie, indeksowanie w bazie wektorowej i zadawanie pytań w języku naturalnym. Odpowiedzi zawierają **źródła** (nazwa pliku + numer strony) oraz wyekstrahowane **kluczowe dane liczbowe** prezentowane w formie tabeli.

### Główne funkcjonalności

- Wgrywanie dokumentów i indeksowanie w Postgres (pgvector)
- Zadawanie pytań w języku naturalnym z kontekstem historii czatu
- Odpowiedzi ze wskazaniem źródeł (nazwa pliku + strona)
- Ekstrakcja kluczowych danych finansowych do tabeli (eksport CSV)
- Wyszukiwanie hybrydowe (wektorowe + pełnotekstowe) z rerankingiem
- Walidacja języka dokumentu (filtr akceptuje dokumenty w języku polskim)
- Zarządzanie dokumentami i sesjami czatu przez GUI
- Przetwarzanie plików w tle (background tasks)

## Stos technologiczny

| Warstwa | Technologia |
|---|---|
| Backend API | FastAPI (async) |
| Interfejs użytkownika | Streamlit |
| Framework RAG | LlamaIndex |
| Baza wektorowa | PostgreSQL + pgvector (`llama-index-vector-stores-postgres`) |
| Embeddingi | `BAAI/bge-m3` (HuggingFace, wymiar 1024) |
| LLM — zapytania | Groq `llama-3.1-8b-instant` |
| LLM — synteza | OpenAI 'o3-mini'/Groq `llama-3.3-70b-versatile` |
| Parser dokumentów | LlamaParse |
| Node parser | SentenceWindowNodeParser (window_size=5) |
| Detekcja języka | lingua-language-detector |
| ORM / baza danych | SQLAlchemy + asyncpg |
| Tracing / ewaluacja | LangSmith (opcjonalnie) |
| Konteneryzacja | Docker Compose |

## Architektura

```
Użytkownik (Streamlit GUI)
        │
        ▼
   FastAPI (async)
   ├── POST /upload     → zapis pliku + rejestracja w DB + walidacja języka
   │                       → background task: parsowanie → chunking → insert do PGVector
   ├── POST /query      → pobranie historii czatu → retrieval hybrydowy + reranking → synteza LLM
   ├── CRUD /chats      → tworzenie, lista, historia, edycja tytułu, soft-delete
   ├── CRUD /documents  → lista, usuwanie (plik + wektory)
   └── GET  /files/{f}  → serwowanie PDF-ów (podgląd w przeglądarce)
        │
        ▼
   PostgreSQL + pgvector
   (tabele: raporty_finansowe_hybrid, chat_sessions, messages, documents)
```

### Pipeline przetwarzania dokumentu

1. `POST /upload` — zapisuje plik do `files/`, waliduje język (`lingua`), rejestruje w tabeli `documents`
2. Background task — parsuje dokument (`LlamaParse`), dzieli na węzły (`SentenceWindowNodeParser`)
3. Węzły z metadanymi (filename, page_label, file_id) zapisywane do PGVector

### Pipeline odpowiedzi na pytanie

1. `POST /query` — zapisuje wiadomość użytkownika w historii czatu
2. Pobiera ostatnie 20 wiadomości jako kontekst konwersacji
3. Retrieval:
   -- Query fusion - Rozbija pytanie nad podzapytania
   -- Hybrid retrival - Każde podzapytanie przetwarzane jest przez Semantic Search i Keyword Search, wynikiem jest ranking najbardziej pasujących danych
   -- Reranker - Wyniki wszystkich podzapytań przetwarzane są przez Reranker, który podaje końcowy wynik
5. Synteza odpowiedzi przez LLM
6. Zwraca `summary_text` + `key_numbers` + `source_data` (filename + page_ref + node_content)

## Struktura projektu

```
├── src/
│   ├── api/
│   │   └── app.py                 # Endpointy FastAPI
│   ├── config/
│   │   └── settings.py            # Konfiguracja (Pydantic Settings, LlamaIndex)
│   ├── database/
│   │   └── db.py                  # Sesje SQLAlchemy (async)
│   ├── gui/
│   │   └── gui.py                 # Interfejs Streamlit
│   ├── models/                    # Modele SQLAlchemy (tabele DB)
│   ├── schemas/
│   │   └── schemas.py             # Schematy Pydantic (request/response)
│   ├── services/
│   │   ├── RAGSerivce.py          # Serwis RAG (indeksowanie, odpowiedzi)
│   │   ├── FinQueryEngine.py      # Budowa pipeline'u retrieval + synteza
│   │   ├── parser.py              # Parsowanie dokumentów (LlamaParse)
│   │   ├── chat_service.py        # Operacje CRUD na czatach
│   │   ├── document_service.py    # Operacje CRUD na dokumentach
│   │   └── language_gate.py       # Walidacja języka dokumentu
│   └── docker-compose.yml         # Docker Compose (DB + opcjonalnie API/UI)
├── tests/
│   └── rag_test/
│       └── rag_test.py            # Testy RAG / ewaluacja
├── requirements.txt
├── .env.example
└── README.md
```

## Uruchomienie

### 1. Konfiguracja zmiennych środowiskowych

```bash
cp .env.example .env
```

Uzupełnij klucze API w pliku `.env`:

```dotenv
GROQ_API_KEY=twoj_klucz
LLAMAPARSE_API_KEY=twoj_klucz
OPENAI_API_KEY=twoj_klucz
LLM_PROVIDER=groq / openai #Dla lepszych wyników wybierz openai
```

### 2. Uruchomienie bazy danych (pgvector)

```bash
docker compose -f src/docker-compose.yml up -d db
```

### 3. Instalacja zależności

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Uruchomienie backendu

```bash
PYTHONPATH=. uvicorn src.api.app:app --reload --port 8000
```

API dostępne pod: `http://localhost:8000`

### 5. Uruchomienie interfejsu

```bash
streamlit run src/gui/gui.py
```

UI dostępne pod: `http://localhost:8501`

### Docker (cały stack)

```bash
docker compose -f src/docker-compose.yml up --build
```

## Endpointy API

| Metoda | Ścieżka | Opis |
|---|---|---|
| `GET` | `/status` | Status gotowości serwisu RAG |
| `POST` | `/upload` | Wgranie plików (background processing) |
| `GET` | `/jobs/{job_id}` | Status zadania przetwarzania |
| `POST` | `/query` | Zapytanie RAG z historią czatu |
| `GET` | `/files/{filename}` | Podgląd pliku PDF |
| `POST` | `/chats` | Utworzenie nowej sesji czatu |
| `GET` | `/chats` | Lista aktywnych czatów |
| `GET` | `/chats/{chat_id}` | Pełna historia czatu |
| `PATCH` | `/chats/{chat_id}` | Zmiana tytułu czatu |
| `DELETE` | `/chats/{chat_id}` | Usunięcie czatu (soft-delete) |
| `GET` | `/documents` | Lista zaindeksowanych dokumentów |
| `DELETE` | `/documents/{document_id}` | Usunięcie dokumentu i powiązanych wektorów |

## Testy

Skrypty testowe i ewaluacyjne znajdują się w katalogu `tests/`.

```bash
PYTHONPATH=. python tests/rag_test/rag_test.py
```

---

## Ewaluacja

Testy przeprowadziłem używając **LangSmith**. Do oceny wykorzystałem podejście **LLM-as-a-Judge** z metryką **Relevance Answer** (ocena trafności odpowiedzi, a nie samego kontekstu). Zbadałem wyniki pod następującymi aspektami:

| Metryka | Opis |
|---|---|
| **Relevance** | Czy znaleziono odpowiedni dokument (ocena retrievera) |
| **Correctness** | Wiarygodność względem prawdy (Ground Truth) |
| **Faithfulness** | Wiarygodność względem otrzymanych informacji ze źródła |
| **Latency** | Czas generacji odpowiedzi |
| **JSON Accuracy** | Zgodność otrzymanych wyników `FinancialMetrics` względem Ground Truth (mój customowy feature, wymagający poprawek) |
| **Errors** | Błędy podczas ewaluacji |

### Wyniki ewaluacji

Ewaluacje przeprowadziłem dla dwóch modeli: **Llama 3.3 70B** i **GPT o3-mini**. Widać zdecydowaną przewagę o3-mini kosztem droższych tokenów oraz dłuższego opóźnienia (Latency).






#### GPT o3-mini (OpenAI)
![Ewaluacja GPT o3-mini — szczegóły](photos/Zrzut%20ekranu%202026-02-11%20133532.png)


#### Llama 3.3 70B (Groq)
![Ewaluacja Llama 3.3 70B — wyniki](photos/Zrzut%ekranu%2026-02-11%155125.png.png)

### Tabela porównawcza


| Metryka | Llama 3.3 70B | GPT o3-mini |
|---|---|---|
| Relevance | 0.86 | 1.00 |
| Correctness | 0.69 | 0.89 |
| Faithfulness | 0.57 | 1.00 |
| Latency (avg) | 12.13s | 22.07s |
| JSON Accuracy | 0.61 | 0.6 |
| Errors | 0 | 0 |

![Porównanie wykres — szczegóły](photos/Zrzut%20ekranu%202026-02-11%20133811.png)

### Zużycie tokenów


| Model | Input tokens | Output tokens | Koszt przybliżony |
|---|---|---|---|
| Llama 3.3 70B (Groq) | — | — | — |
| GPT o3-mini (OpenAI) | — | — | — |

