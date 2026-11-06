import uuid
from fastapi import APIRouter, Depends, HTTPException
from app.schemas_compat import (
    AnthropicContentBlock,
    AnthropicMessageRequest,
    AnthropicMessageResponse,
    AnthropicUsage,
)
from app.services.dispatcher import (
    execute_translation,
    extract_text_from_messages,
    resolve_service_by_model,
)
from app.services.security import verify_api_key

router = APIRouter(
    prefix="/v1",
    tags=["Anthropic Compatibility"],
    dependencies=[Depends(verify_api_key)],
)

@router.post("/messages", response_model=AnthropicMessageResponse)
def create_message(request: AnthropicMessageRequest):
    """
    Anthropic Messages API compatible endpoint.
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

    input_tokens = max(1, len(text_to_translate.split()))
    output_tokens = max(1, len(result.translated_text.split()))

    return AnthropicMessageResponse(
        id=f"msg_{uuid.uuid4().hex[:16]}",
        model=request.model or model_id,
        content=[
            AnthropicContentBlock(
                type="text",
                text=result.translated_text,
            )
        ],
        stop_reason="end_turn",
        usage=AnthropicUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ),
    )
