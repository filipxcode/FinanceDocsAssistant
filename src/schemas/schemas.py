from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import Literal
from uuid import UUID
from datetime import datetime
import re

class InputQuery(BaseModel):
    query: str = Field(min_length=1)
    chat_id: UUID

class FinancialMetric(BaseModel):
    label: str = Field(description="Krótka etykieta danej, np. 'Przychody 2024', 'PKB 2025'")
    amount: float = Field(description="Czysta wartość liczbowa. Jeśli w tekście jest '5 mld', wpisz 5.0. Żadnych napisów!")
    unit: str | None = Field(default=None, description="Jednostka np: 'mld', '%', 'pkt proc.', 'tys.'")
    currency: str | None = Field(default=None, description="Tylko kod waluty: 'PLN', 'USD', 'EUR'. Jeśli brak, to null.")
    date: str | None = Field(default=None, description="Data okresu. Format: 'RRRR' lub 'RRRR-MM' lub 'RRRR-Qx'. Żadnych długich opisów!")

    @field_validator('currency')
    @classmethod
    def validate_currency(cls, v: str | None) -> str | None:
        if v:
            clean = v.strip().upper()
            if len(clean) > 3:
                return clean[:3] 
            return clean
        return v

    @field_validator('date')
    @classmethod
    def validate_date(cls, v: str | None) -> str | None:
        if v:
            
            clean = re.sub(r'(okres|rok|r\.|kwartał|kw\.|styczeń|lutego|marca|...|-)', '', v, flags=re.IGNORECASE).strip()
            
            match = re.search(r'(\d{4})', v) 
            if match:
                if len(v) > 10:
                    return match.group(1)
            return v[:10] if len(v) > 10 else v
        return v

    @field_validator('unit')
    @classmethod
    def validate_unit(cls, v: str | None) -> str | None:
        if v and len(v) > 10:
            return v[:10]
        return v



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
    real_filename: str | None = None
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

class DocumentOut(BaseModel):
    id: UUID
    filename: str
    original_filename: str
    created_at: datetime
    size_bytes: int
    
    model_config = ConfigDict(from_attributes=True)