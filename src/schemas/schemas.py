from pydantic import BaseModel, Field, ConfigDict
from typing import Literal
from uuid import UUID
from datetime import datetime

class InputQuery(BaseModel):
    query: str
    chat_id: UUID

class FinancialMetric(BaseModel):
    label: str = Field(description="Krótka etykieta danej, np. 'Przychody 2024', 'PKB 2025'")
    amount: float = Field(description="Czysta wartość liczbowa. Jeśli w tekście jest '5 mld', wpisz 5.0. Żadnych napisów!")
    unit: str | None = Field(default=None, description="Jednostka np: 'mld', '%', 'pkt proc.'")
    currency: str | None = Field(default=None, description="Waluta np PLN USD EUR")
    date: str | None = Field(default=None, description="Data lub okres (RRRR, MM-RRRR lub DD-MM-RRRR)")


class ResponseOutput(BaseModel):
    summary_text: str = Field(description="Merytoryczne podsumowanie odpowiedzi na pytanie użytkownika.")
    key_numbers: list[FinancialMetric] | None = Field(
        default=None, 
        description="Lista kluczowych danych do wykresu"
    )


class SourceData(BaseModel):
    fragment_number: int
    page_ref: int
    filename: str
    node_content: str
    
class ResponseOutputFinal(BaseModel):
    llm_output: ResponseOutput
    source_data: list[SourceData]
    

class ChatTemplate(BaseModel):
    created_at: datetime | None = None
    role: Literal["user", "assistant"]
    text: str 
    
    metrics: list[FinancialMetric] | None = Field(default=None)
    sources: list[SourceData] | None = Field(default=None)
    
    @classmethod
    def from_input(cls, inp: InputQuery):
        return cls(role="user", text=inp.query)

    @classmethod
    def from_response(cls, resp: ResponseOutputFinal):
        return cls(
            role="assistant",
            text=resp.llm_output.summary_text,
            metrics=resp.llm_output.key_numbers,
            sources=resp.source_data
        )
    model_config = ConfigDict(from_attributes=True) 

class ChatSessionOut(BaseModel):
    id: UUID
    created_at: datetime
    title: str | None = Field(default=None, min_length=1, max_length=100)
    
    model_config = ConfigDict(from_attributes=True)

class ChatSessionFull(ChatSessionOut):
    messages: list[ChatTemplate] = [] 

class ChatUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=100)

class ChatCreate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=100)
