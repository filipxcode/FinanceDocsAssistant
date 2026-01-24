from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core import PromptTemplate
from llama_index.core import Settings
from llama_index.core import get_response_synthesizer
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import QueryBundle
import logging
from src.schemas.schemas import ResponseOutput, SourceData, ResponseOutputFinal
import asyncio

logger = logging.getLogger(__name__)
logging.getLogger("llama_index.core.retrievers").setLevel(logging.DEBUG)
logging.getLogger("llama_index.core.indices.vector_store").setLevel(logging.DEBUG)
logging.getLogger("httpx").setLevel(logging.DEBUG)

QUERY_GEN_PROMPT = (
    """Jesteś ekspertem ds. wyszukiwania informacji finansowych. 
    Twoim zadaniem jest przygotowanie listy zapytań do bazy wektorowej na podstawie pytania użytkownika.

    ZASADY ANALIZY:
    1. DEKOMPOZYCJA: Jeśli pytanie jest złożone (np. "Porównaj X i Y"), rozbij je na osobne zapytania o X i osobne o Y.
       Przykład: "Porównaj wyniki 2024 i 2025" -> "Wyniki finansowe rok 2024", "Wyniki finansowe rok 2025".
    2. STRESZCZANIE I SŁOWA KLUCZOWE: Jeśli pytanie jest długie, wyciągnij z niego tylko kluczowe podmioty i wskaźniki (np. "EBITDA", "Marża netto").
    3. SYNONIMY: Jeśli wskaźnik ma różne nazwy (np. Przychody/Sprzedaż, Zysk/Wynik netto), użyj wariantów w kolejnych zapytaniach.
    4. PRECYZJA: Używaj pełnych nazw wskaźników finansowych i konkretnych dat/okresów (Q1, H1, FY).
    5. JĘZYK: Generuj zapytania wyłącznie po polsku.

    FORMATOWANIE:
    - Wygeneruj dokładnie {num_queries} zapytań.
    - Każde zapytanie w nowej linii.
    - Nie dodawaj numeracji, myślników ani cudzysłowów na początku linii.
    - NIE dodawaj zbędnych wstępów jak "Oto lista zapytań". Rozpocznij od razu od pierwszego zapytania.

    Pytanie użytkownika: {query}

    Lista zapytań do bazy:
    """
)
ANSWEAR_GEN_PROMPT = (
    """
    Jesteś ekspertem w analizie finansowej. Twoim celem jest precyzyjna odpowiedź na pytanie oraz ekstrakcja kluczowych danych liczbowych.
    
    ### FORMAT DANYCH WEJŚCIOWYCH
        Otrzymasz fragmenty tekstu. Każdy z nich zawiera metadane (nazwę pliku i numer strony).
        Przykład fragmentu:
        "File: raport_2024.pdf, Page: 15 | Zysk netto wyniósł 500 mln zł."
        
    ### ZASADY ODPOWIEDZI (Summary):
    1. Odpowiadaj w formie podsumowania w polu 'summary_text', 
    2. Stwórz spójne, merytoryczne podsumowanie, skup się na płynności i poprawności danych.
    ### ZASADY EKSTRAKCJI DANYCH LICZBOWYCH ('key_numbers'):
    1. Jeśli w tekście znajdziesz dane liczbowe, dodaj je do listy `key_numbers`.
    2. Pola: 'amount' (sama liczba float), 'unit' (jednostka, %, mld), 'currency' (PLN, EUR), 'date' (rok/okres).
    3. Jeśli brakuje danych (np. jednostki), zostaw null.

    
    ### PRZYKŁAD ODPOWIEDZI JSON:
    {
        "summary_text": "W 2024 roku przychody spółki osiągnęły poziom 100 mln PLN [1]. Jednocześnie odnotowano spadek zysku netto o 5%.",
        "key_numbers": [
            {"label": "Przychody 2024", "amount": 100.0, "unit": "mln", "currency": "PLN", "date": "2024"},
            {"label": "Spadek zysku netto", "amount": -5.0, "unit": "%", "currency": null, "date": "2024"}
        ]
    }
    
    Jeśli nie ma dostarczonych informacji, odpowiadaj tylko i wyłącznie "Niestety, nie znalazłem informacji na ten temat".
    
    Podsumowanie jest dla analityków finansowych, którzy potrzebują odpowiedzi z dostarczonych danych finansowych.
    Twój ton ma być jasny i profesjonalny.
    -------------------
    PYTANIE:{query_str}
    -------------------
    DOSTARCZONE INFORMACJE:{context_str}
    """
)

CONDENSE_QUESTION_PROMPT = (
"""Biorąc pod uwagę poniższą historię rozmowy oraz nowe pytanie, 
    sformułuj je na nowo tak, aby było samodzielnym pytaniem, idealnym do wyszukania w raporcie finansowym.

    ZASADY TRANSFORMATORA PYTAŃ:

    1. ROZWIĄZYWANIE CZASU (NAJWAŻNIEJSZE!):
        - Jeśli użytkownik używa pojęć względnych ("rok wcześniej", "poprzedni kwartał", "analogiczny okres"), 
         MUSISZ znaleźć ostatnią datę w Historii i obliczyć konkretną nową datę.
        - Przykład: Jeśli w historii jest "Q1 2025", a user pyta "a rok temu?", zamień to na "Q1 2024".
        - Nigdy nie zostawiaj w pytaniu słów "tamten rok/kwartał".

    2. UTRZYMANIE SEGMENTU/KONTEKSTU:
        - Jeśli nowe pytanie brzmi jak KONTYNUACJA (np. "a ile zarobili?", "jaki był wynik?"), a historia dotyczy konkretnego segmentu - zachowaj ten segment.
        - Jeśli użytkownik pyta o ogólny wskaźnik (np. "EBITDA", "Zysk"), NIE zakładaj automatycznie, że chodzi o segment z historii.
        - Domyślnie pytania o finanse dotyczą CAŁEJ GRUPY, chyba że kontekst wskazuje wyraźnie inaczej.

    3. SŁOWNIK FINANSOWY (MAPPING):
        - Zamieniaj potoczne słowa na terminologię z raportów giełdowych:
        - "Ile zarobili" -> "Zysk netto / EBITDA"
        - "Ile sprzedali" -> "Przychody / GMV"
        - "Ile wydali" -> "Koszty operacyjne / CAPEX"
        - Jeśli użytkownik pyta nieprecyzyjnie, użyj fachowej nazwy wskaźnika, o którym była mowa.

    4. FORMATOWANIE I CZYSTOŚĆ:
        - Usuń z pytania wszelkie zwroty grzecznościowe ("cześć", "proszę"), wstępy ("chciałbym wiedzieć") i dygresje.
        - Zwróć TYLKO samo sformułowane pytanie. Żadnych wstępów typu "Oto zmodyfikowane pytanie:".
        - NIE ODPOWIADAJ na pytanie. Twoim zadaniem jest tylko redakcja tekstu.
        - Jeśli pytanie jest niezrozumiałe, bezsensowne lub za krótkie ZWRÓĆ JE BEZ ZMIAN lub jako pusty string.
        - Absolutnie NIE PISZ "Nie znalazłem odpowiedzi", bo nie jesteś od odpowiadania.
    ---------------------
    Historia rozmowy:
    {chat_history}
    
    Nowe pytanie użytkownika: {question}
    ---------------------
    
    ZMODYFIKOWANE PYTANIE:"""
)

class FinancialQueryEngine(CustomQueryEngine):
    retriever: BaseRetriever
    response_synthesizer: BaseSynthesizer
    postprocessors: list[BaseNodePostprocessor]
    
    async def acustom_query(self, query: str, chat_history: list[str] = None):
        if len(query) < 3:
            return self._empty_response()
        chat_history = chat_history or []
        if len(chat_history) > 0:
            history_str = "\n".join(chat_history)
            prompt = CONDENSE_QUESTION_PROMPT.format(chat_history=history_str, question=query)
            response = await Settings.llm.acomplete(prompt)
            logger.info(f"\n\nzmodyfikowane query{response}\n\n")
            query = response.text.strip()
        logger.info(f"\n\CZATY {chat_history}\n\n")
        #logger.info(" [KROK 1] Rozpoczynam Fusion Retriever...")
        nodes = await self.retriever.aretrieve(query)
        #logger.info(f" [KROK 1] Zakończono. Pobranno {len(nodes)} surowych node.")
        if not nodes:
            #logger.warning("Pusto. Wracam.")
            return self._empty_response()
        
        query_bundle = QueryBundle(query)
        #logger.info(f"\n\n1\n {nodes}\n\n")
        #logger.info(" [KROK 2] Rozpoczynam postprocessing...")
        loop = asyncio.get_running_loop()
        
        for p in self.postprocessors:
            nodes = await loop.run_in_executor(
                None, 
                p.postprocess_nodes, 
                nodes, 
                query_bundle
            )
        #logger.info(f"[KROK 2] Zakończono. Zostało {len(nodes)} node'ów.")
        if not nodes:
            logger.warning("Pusto. Wracam.")
            return self._empty_response()
        #logger.info("[KROK 3] Rozpoczynam Syntezę (LLM generuje odpowiedź)...")
        response = await self.response_synthesizer.asynthesize(query, nodes=nodes)
        #logger.info("[KROK 3] LLM odpowiedział!")
        #logger.info(f"\n\n2\n{nodes}\n\n")
        return self._create_final_response(response, nodes)

    def custom_query(self, query: str):
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.acustom_query(query))

    def _create_final_response(self, response, nodes):
        sources_data_list = [] 
        llm_output = response.response
        logger.info(f"[SafeFinEngine] Znaleziono {len(nodes)} pasujących fragmentów po rerankingu.")
        for idx, node_with_score in enumerate(nodes):
            actual_node = node_with_score.node
            score_val = node_with_score.score
            
            file = actual_node.metadata.get('filename', 'N/A')
            page = actual_node.metadata.get('page_label', '0')
            text = actual_node.get_content()
            try:
                page_int = int(page)
            except ValueError:
                page_int = 0
                
            score_str = round(score_val, 4) if score_val is not None else "N/A"
            
            logger.info(f"   [{idx+1}] Plik: {file} | Str: {page} | Score: {score_str} | {text[:200]}...")

            source_obj = SourceData(
                fragment_number=int(idx+1),
                page_ref=page_int,
                filename=str(file),
                node_content=text
            )
            sources_data_list.append(source_obj)
        return ResponseOutputFinal(llm_output=llm_output, source_data=sources_data_list)
    
    def _empty_response(self):
        return ResponseOutputFinal(llm_output=
                    ResponseOutput(summary_text="Niestety nie udało mi się znaleźć szukanych informacji, prosze doprecyzuj pytanie."), 
                    source_data = [])

class FinancialEngineBuilder:
    def __init__(self, index):
        self.index = index
        
        self.replace_postprocessor = MetadataReplacementPostProcessor(target_metadata_key="window")
        
        self.treshold_postprocessor = SimilarityPostprocessor(similarity_cutoff=0.1)
        
        self.reranker = SentenceTransformerRerank(
            model="BAAI/bge-reranker-v2-m3", 
            top_n=5
        )

    async def query_async(self, question: str, chat_history: list[str]):
        from src.schemas.schemas import ResponseOutput
        
        hybrid_retriver = self.index.as_retriever(
            similarity_top_k=20, 
            vector_store_query_mode="hybrid"
        )

        fusion_retriever = QueryFusionRetriever(
            [hybrid_retriver],
            similarity_top_k=20,
            num_queries=3,
            query_gen_prompt=QUERY_GEN_PROMPT,
            llm=Settings.llm, 
            mode="reciprocal_rerank",
            verbose=True
        )
        
        response_synthesizer = get_response_synthesizer(
            llm=Settings.llm,
            text_qa_template=PromptTemplate(ANSWEAR_GEN_PROMPT),
            response_mode="tree_summarize",
            output_cls=ResponseOutput
        )
        engine = FinancialQueryEngine(
            retriever=fusion_retriever,
            response_synthesizer=response_synthesizer,
            postprocessors=[
                self.replace_postprocessor,
                self.reranker,              
                self.treshold_postprocessor 
            ]
        )
        
        logger.info("[FinQueryEngine] Uruchamianie ASYNC aquery...")
        response = await engine.acustom_query(question, chat_history)
        return response