from fastapi.testclient import TestClient
from datasets import Dataset
import os 
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.run_config import RunConfig
import time

from langchain_community.chat_models import ChatOllama
from src.api.app import app 

load_dotenv()
judge_llm = ChatOllama(
    model="llama3.1",  
    temperature=0,  
    base_url="http://localhost:11434",
    #format="json"
)
embed_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3"
    )

my_run_config = RunConfig(
    max_workers=1,        # Absolutnie konieczne: jeden po drugim
    timeout=360,          # Dajemy mu 3 minuty na odpowiedź (duży bufor)
    max_retries=10,       # Jak dostanie 429, niech próbuje do skutku
    max_wait=30           # Czekaj do 60s między próbami
)

test_data = [
    {
        "question": "Jaka była wartość przychodów 'Total revenue' Allegro w Q1 2025?",
        "ground_truth": "2,622.4 mln PLN",
    },
    {
        "question": "O ile procent zmieniły się przychody 'Polish Operations' w Allegro r/r?",
        "ground_truth": "Wzrosły o 15.0%.",
    },
    {
        "question": "Kto jest Prezesem Zarządu ING Banku Śląskiego?",
        "ground_truth": "Michał Bolesławski",
    },
    {
        "question": "Jaki był zysk netto Allegro w Q4 2025?",
        "ground_truth": "Brak danych w raporcie (raport dotyczy tylko Q1).",
    }
]
questions = []
answers = []
contexts = []
ground_truths = []

with TestClient(app) as client:

    for item in test_data:
        print(f"Testuję pytanie: {item['question']}")
        setup_response = client.post("/chats", json={"title": "Test Ragas"})
        if setup_response.status_code == 200:
            active_chat_id = setup_response.json()["id"]
            print(f"Test session: {active_chat_id}")
        else:
            raise Exception("Failed to create session.")
        time.sleep(2)
        
        response = client.post("/query", json={
            "query": item["question"],
            "chat_id": active_chat_id 
        })
        
        if response.status_code == 200:
            data = response.json()
            
            generated_answer = data["llm_output"]["summary_text"]
            
            retrieved_contexts = [source["node_content"] for source in data["source_data"]]
            
            questions.append(item["question"])
            answers.append(generated_answer)
            contexts.append(retrieved_contexts)
            ground_truths.append(item["ground_truth"])
        else:
            print(f"Błąd API: {response.text}")

print("Generowanie raportu Ragas...")

dataset = Dataset.from_dict({
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths
})


result = evaluate(
    dataset=dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision, 
        context_recall,
    ],
    llm=judge_llm,            
    embeddings=embed_model, 
    run_config=my_run_config
)

print(result)
df = result.to_pandas()
df.to_csv("ragas_results.csv")