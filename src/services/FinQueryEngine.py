import logging
import asyncio
from datetime import datetime
from llama_index.core import Settings, PromptTemplate, get_response_synthesizer
from llama_index.core.retrievers import QueryFusionRetriever, BaseRetriever
from llama_index.core.postprocessor import (
    LLMRerank,
    MetadataReplacementPostProcessor, 
    SimilarityPostprocessor
)
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import QueryBundle

from src.schemas.schemas import ResponseOutput, SourceData, ResponseOutputFinal
from src.config.prompts import (
    QUERY_GEN_PROMPT, 
    ANSWEAR_GEN_PROMPT, 
    CONDENSE_QUESTION_PROMPT
)
from src.config.settings import get_settings

logger = logging.getLogger(__name__)

# Configure logging levels
logging.getLogger("llama_index.core.retrievers").setLevel(logging.DEBUG)
logging.getLogger("llama_index.core.indices.vector_store").setLevel(logging.DEBUG)
logging.getLogger("httpx").setLevel(logging.WARNING) 


class FinancialQueryEngine(CustomQueryEngine):
    retriever: BaseRetriever
    response_synthesizer: BaseSynthesizer
    postprocessors: list[BaseNodePostprocessor]
    
    def custom_query(self, query_str: str):
        """Dummy implementation of abstract method"""
        raise NotImplementedError("Use acustom_query for async execution")

    async def acustom_query(self, query: str, chat_history: list[str] = None):
        """Executes the custom query logic"""
        if len(query) < 3:
            return self._empty_response()
            
        original_query = query
        chat_history = chat_history or []
        
        effective_history = [h for h in chat_history if query.strip() not in h]

        if len(effective_history) > 0:
            history_str = "\n".join(effective_history)
            current_date_str = datetime.now().strftime("%Y-%m-%d")
            prompt = CONDENSE_QUESTION_PROMPT.format(
                chat_history=history_str, 
                question=query, 
                current_date=current_date_str
            )
            response = await Settings.query_llm.acomplete(prompt)
            
            rewritten_query = response.text.strip()
            logger.info(f"\n\n[REWRITE] Oryginał: '{query}' || LLM: '{rewritten_query}'\n\n")
            
            # Fallback logic
            if len(rewritten_query) > 3 and "nie znalazłem" not in rewritten_query.lower():
                query = rewritten_query
            else:
                logger.warning(f"[REWRITE] Nieudane przepisanie ('{rewritten_query}'). Wracam do oryginału.")
                query = original_query
        
        logger.info(f"\n[FUSION INPUT] Query: {query}\n")
        nodes = await self.retriever.aretrieve(query)
        if not nodes:
            return self._empty_response()
        
        query_bundle = QueryBundle(query)

        loop = asyncio.get_running_loop()
        
        for p in self.postprocessors:
            nodes = await loop.run_in_executor(
                None, 
                p.postprocess_nodes, 
                nodes, 
                query_bundle
            )
        if not nodes:
            return self._empty_response()
        response = await self.response_synthesizer.asynthesize(query, nodes=nodes)

        return self._create_final_response(response, nodes)


    def _create_final_response(self, response, nodes):
        """Creates the final response object with source citations"""
        sources_data_list = [] 
        llm_output = response.response
        logger.info(f"[SafeFinEngine] Znaleziono {len(nodes)} pasujących fragmentów po rerankingu.")
        for idx, node_with_score in enumerate(nodes):
            actual_node = node_with_score.node
            score_val = node_with_score.score
            
            file = actual_node.metadata.get('filename', 'N/A')
            real_file = actual_node.metadata.get('real_filename')
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
                real_filename=real_file,
                node_content=text
            )
            sources_data_list.append(source_obj)
        return ResponseOutputFinal(llm_output=llm_output, source_data=sources_data_list)
    
    def _empty_response(self):
        """Returns a standardized empty response"""
        return ResponseOutputFinal(llm_output=
                    ResponseOutput(summary_text="Niestety nie udało mi się znaleźć szukanych informacji, prosze doprecyzuj pytanie."), 
                    source_data = [])

class FinancialEngineBuilder:
    """Builder for FinancialQueryEngine with configured components"""
    def __init__(self, index):
        self.index = index
        self._initialize_pipeline()

    def _initialize_pipeline(self):
        """Initializes all RAG components once during startup"""
        logger.info("[FinancialEngineBuilder] Initializing RAG pipeline components...")

        app_settings = get_settings()
        
        # 1. Postprocessors
        self.replace_postprocessor = MetadataReplacementPostProcessor(target_metadata_key="window")
        self.treshold_postprocessor = SimilarityPostprocessor(similarity_cutoff=0.1)
        rerank_provider = (app_settings.RERANK_PROVIDER or "").lower()
        if rerank_provider == "llm":
            self.reranker = LLMRerank(llm=Settings.query_llm, top_n=app_settings.RERANK_TOP_N)
        elif rerank_provider == "none":
            self.reranker = None
        else:
            raise ValueError(f"Unknown RERANK_PROVIDER: {rerank_provider}. Use 'llm' or 'none'.")
        
        # 2. Retrievers
        self.hybrid_retriever = self.index.as_retriever(
            similarity_top_k=30, 
            vector_store_query_mode="hybrid"
        )

        self.fusion_retriever = QueryFusionRetriever(
            [self.hybrid_retriever],
            similarity_top_k=30,
            num_queries=3,
            query_gen_prompt=QUERY_GEN_PROMPT,
            llm=Settings.query_llm, 
            mode="reciprocal_rerank",
            verbose=True
        )
        
        # 3. Synthesizer
        self.response_synthesizer = get_response_synthesizer(
            llm=Settings.synthesis_llm,
            text_qa_template=PromptTemplate(ANSWEAR_GEN_PROMPT),
            response_mode="tree_summarize",
            output_cls=ResponseOutput
        )

        # 4. Engine Assembly
        self.engine = FinancialQueryEngine(
            retriever=self.fusion_retriever,
            response_synthesizer=self.response_synthesizer,
            postprocessors=[
                self.replace_postprocessor,
                *([self.reranker] if self.reranker is not None else []),
                self.treshold_postprocessor 
            ]
        )
        logger.info("[FinancialEngineBuilder] Engine initialized successfully.")

    async def query_async(self, question: str, chat_history: list[str]):
        """Executes the complete query pipeline asynchronously using pre-initialized engine"""
        logger.info("[FinQueryEngine] Processing query with initialized engine...")
        return await self.engine.acustom_query(question, chat_history)