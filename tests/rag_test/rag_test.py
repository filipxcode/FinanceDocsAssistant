import asyncio
from dotenv import load_dotenv
load_dotenv()
from langsmith import Client, aevaluate
from llama_index.core import Settings
from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    CorrectnessEvaluator,
    RelevancyEvaluator
)
from src.services.RAGSerivce import RAGService
from langsmith import evaluate
from llama_index.core import Response
from llama_index.core.schema import NodeWithScore, TextNode
from src.schemas.schemas import FinancialMetric
from pydantic import BaseModel, Field
import json
from src.config.settings import configure_settings
from src.schemas.schemas import ResponseOutputFinal
from langsmith.schemas import Run, Example
from llama_index.core import PromptTemplate
configure_settings()

class SingleMetricScore(BaseModel):
    label_score: int = Field(description="1 jeśli etykieta pasuje semantycznie (jest synonimem), w przeciwnym razie 0")
    amount_score: int = Field(description="1 jeśli kwota zgadza się matematycznie (różnica < 1%), w przeciwnym razie 0")
    unit_currency_score: int = Field(description="1 jeśli jednostka i waluta są zgodne (np. PLN == pln), w przeciwnym razie 0")
    date_score: int = Field(description="1 jeśli data wskazuje ten sam okres czasu, w przeciwnym razie 0")
    reason: str = Field(description="Krótkie uzasadnienie przyznanych punktów. Wymień konkretne niezgodności (np. 'Zła waluta: oczekiwano EUR, jest PLN').")

class EvalOutputList(BaseModel):
    results: list[SingleMetricScore] = Field(description="Lista ocen odpowiadająca dokładnie (1 do 1) liście metryk oczekiwanych (Ground Truth)")
class FinancialMetricsEvaluator:
    def __init__(self):
        self.llm = Settings.synthesis_llm 
        
    def evaluate(self, metric_out: list[FinancialMetric], metric_exp: list[dict]) -> EvalOutputList:
        clean_metrics = []
        for m in metric_out:
            if hasattr(m, "model_dump"):
                clean_metrics.append(m.model_dump())
            elif isinstance(m, dict):
                clean_metrics.append(m)
            else:
                clean_metrics.append({"raw_value": str(m)})
                
        candidates_json = json.dumps(clean_metrics, ensure_ascii=False)
        ground_truth_json = json.dumps(metric_exp, ensure_ascii=False)
        if not metric_exp and not metric_out:
            return EvalOutputList(results=[]) 
            
        if not metric_exp:
            return EvalOutputList(results=[])
        
        eval_prompt = f"""
            Jesteś starszym audytorem finansowym, zajmujesz się sprawdzaniem poprawności danych finansowych.
            Dostaniesz dane wzorca i dane przychodzące, twoim zadaniem jest porównanie wyników i ocena jakości.
            Każde pole porównawcze ma te same zmienne, których wynik powinien się pokrywać.
            Oceń każde pole wynikiem 0 lub 1, według kryteriów.
            ### DANE WEJŚCIOWE:
                1. Ground Truth (Oczekiwane): {ground_truth_json}
                2. Candidate (Wygenerowane): {candidates_json}
            ### KRYTERIA dla pól:
                1. LABEL (Etykieta):
                - Ocena 1.0: Etykiety są identyczne LUB są oczywistymi synonimami w kontekście finansowym (np. "Przychody ze sprzedaży" == "Sprzedaż", "EBITDA" == "Wynik operacyjny powiększony o amortyzację").
                - Ocena 0.0: Etykiety oznaczają co innego (np. "Przychód" vs "Zysk", "Netto" vs "Brutto").

                2. AMOUNT (Kwota):
                - Ocena 1.0: Wartości liczbowe są matematycznie równe. Ignoruj różnice w formacie zapisu (np. "1 000" == "1000", "10.0" == "10").
                - Akceptowalna tolerancja błędu: < 1% różnicy (błędy zaokrągleń są dopuszczalne).
                - Ocena 0.0: Wartości różnią się znacząco.

                3. UNIT & CURRENCY (Jednostka i Waluta):
                - Ocena 1.0: Są tożsame, ignorując wielkość liter i kropki (np. "mln" == "Mln.", "PLN" == "pln", "tys." == "k").
                - Ocena 0.0: Różne waluty (USD vs PLN) lub rzędy wielkości (tys. vs mln) bez przeliczenia kwoty.

                4. DATE (Data):
                - Ocena 1.0: Wskazują na ten sam okres sprawozdawczy.
                - Przykłady zgodności: "2024" == "2024-12-31" (dla danych rocznych), "Q3 2024" == "3 kwartał 2024".
                - Ocena 0.0: Różne lata lub okresy.
            
        """
        eval_template = PromptTemplate(eval_prompt)
        try:
            response_obj = self.llm.structured_predict(
                EvalOutputList, 
                prompt=eval_template
            )
            return response_obj
            
        except Exception as e:
            print(f"LLM Eval Error: {e}")
            return EvalOutputList(results=[])
client = Client()
dataset_name = "dataset-rag-finance"

judge_llm =Settings.synthesis_llm 
faithfulness_eval = FaithfulnessEvaluator(llm=judge_llm)
correctness_eval = CorrectnessEvaluator(llm=judge_llm)
relevance_eval = RelevancyEvaluator(llm=judge_llm) 
metrics_eval = FinancialMetricsEvaluator()
ragservice = RAGService()

async def rag_target(inputs: dict) -> dict:

    question = inputs["question"]
    response_obj = await ragservice.aget_answear(question, chat_history=[])
    return response_obj.model_dump()

async def rag_evaluator(run, example) -> dict:
    question = example.inputs["question"]
    ground_truth = example.outputs["ground_truth"]
    expected_metrics = example.outputs.get("expected_metrics", [])
    
    try:
        response_obj = ResponseOutputFinal(**run.outputs)
    except Exception as e:
        return {"results": [{"key": "error", "score": 0, "comment": str(e)}]}

    print(f"\n Oceniam: {question}")
    
    has_sources = len(response_obj.source_data) > 0

    summary_text = response_obj.llm_output.summary_text or "Brak odpowiedzi."
    
    llama_response_object = Response(
        response=summary_text,
        source_nodes=[
            NodeWithScore(
                node=TextNode(text=src.node_content, metadata={"filename": src.filename}), 
                score=1.0 
            ) 
            for src in response_obj.source_data
        ]
    )

    tasks = []
    
    if has_sources:
        tasks.append(faithfulness_eval.aevaluate_response(response=llama_response_object))
    else:
        tasks.append(asyncio.sleep(0, result="SKIP"))

    tasks.append(correctness_eval.aevaluate_response(query=question, response=llama_response_object, reference=ground_truth))

    if has_sources:
        tasks.append(relevance_eval.aevaluate_response(query=question, response=llama_response_object))
    else:
        tasks.append(asyncio.sleep(0, result="SKIP"))

    results_list = await asyncio.gather(*tasks)
    
    res_faith = results_list[0]
    res_corr = results_list[1]
    res_rel = results_list[2]


    if res_faith == "SKIP":
        score_faith = 1.0 
        comment_faith = "Brak źródeł - model poprawnie nie wygenerował odpowiedzi na podstawie pustego kontekstu."
    else:
        score_faith = 1.0 if res_faith.passing else 0.0
        comment_faith = res_faith.feedback


    if res_rel == "SKIP":
        score_rel = 1.0 
        comment_rel = "Brak pobranych dokumentów."
    else:
        score_rel = 1.0 if res_rel.passing else 0.0
        comment_rel = res_rel.feedback


    score_corr = res_corr.score / 5.0
    comment_corr = res_corr.feedback

    metric_out = response_obj.llm_output.key_numbers or []
    
    if not expected_metrics and not metric_out:
        final_metric_score = 1.0
        reasons = ["Poprawny brak metryk."]
    else:
        eval_result_m = metrics_eval.evaluate(metric_out=metric_out, metric_exp=expected_metrics)
        if not eval_result_m.results:
            final_metric_score = 0.0 if expected_metrics else 1.0
            reasons = ["Brak wyników oceny metryk."]
        else:
            total_possible = len(expected_metrics) * 4 
            current_points = sum([r.label_score + r.amount_score + r.unit_currency_score + r.date_score for r in eval_result_m.results])
            reasons = [r.reason for r in eval_result_m.results if r.reason]
            final_metric_score = current_points / total_possible if total_possible > 0 else 0

    return {
        "results": [
            {"key": "faithfulness", "score": score_faith, "comment": comment_faith},
            {"key": "correctness", "score": score_corr, "comment": comment_corr},
            {"key": "relevance", "score": score_rel, "comment": comment_rel},
            {"key": "json_accuracy_llm", "score": final_metric_score, "comment": "; ".join(reasons)}
        ]
    }

async def main():
    results = await aevaluate(
        rag_target,            
        data=dataset_name,      
        evaluators=[rag_evaluator], 
        client=client,
        max_concurrency=5      
    )
    print(results)
    
if __name__ == "__main__":
    asyncio.run(main())