from __future__ import annotations

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from app.config import get_settings

bearer_security = HTTPBearer(auto_error=False)
header_security = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(
    bearer: HTTPAuthorizationCredentials | None = Security(bearer_security),
    header_key: str | None = Security(header_key_security := header_security),
) -> str:
    """
    Validates API key from request headers.
    Supports:
      1. 'Authorization: Bearer <API_KEY>' (Standard for OpenAI, Anthropic, LangChain, SDKs)
      2. 'X-API-Key: <API_KEY>' (Standard for Anthropic, custom REST headers)

    The server API key must be configured in Google Colab / Kaggle Secrets as 'API_KEY'.
    No automatic generation is performed.
    """
    settings = get_settings()
    expected_key = settings.api_key

    if not expected_key or not expected_key.strip():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "API_KEY is not configured on this server. "
                "Please configure the 'API_KEY' secret in Google Colab (volet Clé 🔑) "
                "or Kaggle (Add-ons > Secrets)."
            ),
        )

    provided_token: str | None = None
    if bearer and bearer.credentials:
        provided_token = bearer.credentials.strip()
    elif header_key:
        provided_token = header_key.strip()

    if not provided_token or provided_token != expected_key.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                "Invalid or missing API key. "
                "Provide a valid key via header 'Authorization: Bearer <API_KEY>' or 'X-API-Key: <API_KEY>'."
            ),
            headers={"WWW-Authenticate": "Bearer"},
        )

    return provided_token
