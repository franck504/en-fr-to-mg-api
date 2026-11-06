from __future__ import annotations

import time
from typing import Any
from pydantic import BaseModel, ConfigDict, Field

# ==============================================================================
# 1. OpenAI Compatibility Schemas (/v1/chat/completions, /v1/models)
# ==============================================================================

class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    role: str
    content: str | list[dict[str, Any]] | None = ""

class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model: str = Field(default="default")
    messages: list[ChatMessage]
    temperature: float | None = 0.0
    max_tokens: int | None = 256
    stream: bool | None = False

class ChatCompletionMessage(BaseModel):
    role: str = "assistant"
    content: str

class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatCompletionMessage
    finish_reason: str = "stop"

class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageInfo = Field(default_factory=UsageInfo)

class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = 1700000000
    owned_by: str = "en-fr-to-mg-api"

class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelCard]


# ==============================================================================
# 2. Anthropic Compatibility Schemas (/v1/messages)
# ==============================================================================

class AnthropicContentBlock(BaseModel):
    type: str = "text"
    text: str

class AnthropicMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    role: str
    content: str | list[dict[str, Any]] | list[AnthropicContentBlock]

class AnthropicMessageRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model: str = Field(default="default")
    messages: list[AnthropicMessage]
    max_tokens: int = 256
    system: str | None = None
    temperature: float | None = None
    stream: bool | None = False

class AnthropicUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0

class AnthropicMessageResponse(BaseModel):
    id: str
    type: str = "message"
    role: str = "assistant"
    content: list[AnthropicContentBlock]
    model: str
    stop_reason: str = "end_turn"
    stop_sequence: str | None = None
    usage: AnthropicUsage = Field(default_factory=AnthropicUsage)


# ==============================================================================
# 3. Ollama Compatibility Schemas (/api/chat, /api/generate, /api/tags)
# ==============================================================================

class OllamaMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    role: str
    content: str

class OllamaChatRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model: str = Field(default="default")
    messages: list[OllamaMessage]
    stream: bool = False

class OllamaChatResponse(BaseModel):
    model: str
    created_at: str
    message: OllamaMessage
    done: bool = True
    total_duration: int | None = None
    load_duration: int | None = None
    prompt_eval_count: int | None = None
    eval_count: int | None = None

class OllamaGenerateRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model: str = Field(default="default")
    prompt: str
    stream: bool = False

class OllamaGenerateResponse(BaseModel):
    model: str
    created_at: str
    response: str
    done: bool = True

class OllamaModelDetails(BaseModel):
    parent_model: str = ""
    format: str = "pytorch"
    family: str = "seq2seq"
    parameter_size: str = "600M"
    quantization_level: str = "fp16"

class OllamaModelItem(BaseModel):
    name: str
    model: str
    modified_at: str
    size: int
    digest: str
    details: OllamaModelDetails

class OllamaTagsResponse(BaseModel):
    models: list[OllamaModelItem]
