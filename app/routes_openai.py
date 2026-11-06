import time
import uuid
from fastapi import APIRouter, Depends, HTTPException
from app.config import get_settings
from app.schemas_compat import (
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelCard,
    ModelListResponse,
    UsageInfo,
)
from app.services.dispatcher import (
    execute_translation,
    extract_text_from_messages,
    resolve_service_by_model,
)
from app.services.security import verify_api_key

router = APIRouter(
    prefix="/v1",
    tags=["OpenAI Compatibility"],
    dependencies=[Depends(verify_api_key)],
)
settings = get_settings()

@router.get("/models", response_model=ModelListResponse)
def list_models():
    """Returns available models formatted for OpenAI SDK compatibility."""
    now = int(time.time())
    models = [
        ModelCard(id=settings.hf_model_name, created=now),
        ModelCard(id="nllb", created=now),
        ModelCard(id="m2m100", created=now),
        ModelCard(id=settings.gemma4_model_name, created=now),
        ModelCard(id="gemma4", created=now),
        ModelCard(id="default", created=now),
        # Popular aliases so existing clients work out of the box
        ModelCard(id="gpt-4o", created=now),
        ModelCard(id="gpt-3.5-turbo", created=now),
    ]
    return ModelListResponse(data=models)

@router.post("/chat/completions", response_model=ChatCompletionResponse)
def create_chat_completion(request: ChatCompletionRequest):
    """
    OpenAI-compatible Chat Completions endpoint.
    Translates user messages from English or French to Malagasy.
    """
    text_to_translate = extract_text_from_messages(request.messages)
    if not text_to_translate:
        raise HTTPException(
            status_code=400,
            detail="No text to translate found in user messages.",
        )

    service, model_id = resolve_service_by_model(request.model)

    try:
        result = execute_translation(
            text=text_to_translate,
            service=service,
            source_lang="auto",
            target_lang="mg",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    prompt_tokens = max(1, len(text_to_translate.split()))
    completion_tokens = max(1, len(result.translated_text.split()))

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        model=request.model or model_id,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content=result.translated_text,
                ),
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )
