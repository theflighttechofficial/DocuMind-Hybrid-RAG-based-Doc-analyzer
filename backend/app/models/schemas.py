from pydantic import BaseModel
from datetime import datetime


class DocumentOut(BaseModel):
    id: str
    name: str
    file_type: str
    file_size: int
    page_count: int
    word_count: int
    sentence_count: int
    paragraph_count: int
    processing_status: str
    error_message: str | None = None
    created_at: datetime

    class Config:
        from_attributes = True


class DocumentDetailOut(DocumentOut):
    entities_json: dict | None = None
    keywords_json: dict | None = None
    pos_json: dict | None = None
    stats_json: dict | None = None
    toc_json: list | None = None
    title: str | None = None
    structure_json: dict | None = None


class PageOut(BaseModel):
    page_number: int
    text: str

    class Config:
        from_attributes = True


class AskRequest(BaseModel):
    question: str
    conversation_id: str | None = None


class EvidenceOut(BaseModel):
    page: int
    section: str | None
    text: str
    relevance: float


class AskResponse(BaseModel):
    conversation_id: str
    answer: str
    grounded: bool
    evidence: list[EvidenceOut]


class SearchRequest(BaseModel):
    query: str
    method: str = "hybrid"  # semantic | bm25 | hybrid
    top_k: int = 8


class SearchResultOut(BaseModel):
    page: int
    section: str | None
    text: str
    relevance: float


class SummaryRequest(BaseModel):
    length: str = "standard"  # brief | standard | detailed


class CompareRequest(BaseModel):
    document_id_a: str
    document_id_b: str


class SettingsOut(BaseModel):
    gemini_api_key_set: bool
    gemini_api_key_masked: str
    gemini_model: str


class UpdateGeminiKeyRequest(BaseModel):
    api_key: str


class TestConnectionOut(BaseModel):
    ok: bool
    message: str


class RegisterRequest(BaseModel):
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserOut(BaseModel):
    id: str
    email: str

    class Config:
        from_attributes = True


class WorkspaceCreate(BaseModel):
    name: str


class WorkspaceOut(BaseModel):
    id: str
    name: str
    created_at: datetime
    documents: list[DocumentOut] = []

    class Config:
        from_attributes = True


class WorkspaceDocumentsRequest(BaseModel):
    document_ids: list[str]


class WorkspaceAskRequest(BaseModel):
    question: str


class WorkspaceEvidenceOut(BaseModel):
    document_id: str
    document_name: str
    page: int
    section: str | None
    text: str
    relevance: float


class WorkspaceAskResponse(BaseModel):
    answer: str
    grounded: bool
    evidence: list[WorkspaceEvidenceOut]
