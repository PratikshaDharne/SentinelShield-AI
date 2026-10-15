from __future__ import annotations
from typing import Any, Optional
from pydantic import BaseModel, Field


class Turn(BaseModel):
    role: str
    content: str


class GuardRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=20000)
    history: list[Turn] = []


class ChatRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=20000)
    history: list[Turn] = []
    system: Optional[str] = None            # the CLIENT's own (trusted) system prompt


class ToolRequest(BaseModel):
    tool_name: str
    instruction: str = ""
    arguments: dict[str, Any] = {}
    sensitivity: Optional[str] = None


class OpenAIChatRequest(BaseModel):
    model: str = "sentinelshield"
    messages: list[Turn]
    stream: bool = False
    temperature: Optional[float] = None


class LoginRequest(BaseModel):
    username: str
    password: str


class KeyCreate(BaseModel):
    name: str
    rate_limit_per_min: int = 120


class ReviewUpdate(BaseModel):
    status: str                  # confirmed | false_positive | pending
    note: str = ""


class RLMode(BaseModel):
    mode: str                      # off | shadow | active | auto


class RLReset(BaseModel):
    warm: bool = True              # False = cold start (random network) to watch it learn from scratch


class RLSim(BaseModel):
    steps: int = 600
    rate: float = 8.0
    source: str = "val"            # val | val_unseen | unseen_test_drift
    seed: int = 0


class RLFeedback(BaseModel):
    audit_id: int
    label: str                     # CONFIRM | ALLOW | REWRITE | BLOCK
