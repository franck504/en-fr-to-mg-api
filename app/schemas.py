from pydantic import BaseModel, Field, field_validator

# Supported languages
SUPPORTED_SOURCE_LANGUAGES = {"auto", "en", "fr"}
SUPPORTED_TARGET_LANGUAGES = {"mg"}

class TranslateRequest(BaseModel):
    """
    Request model for translation tasks.
    Validates text length and ensures source/target languages are supported.
    """
    text: str = Field(..., min_length=1, max_length=5000)
    source_lang: str = Field(default="auto")
    target_lang: str = Field(default="mg")

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("The text to translate cannot be empty.")
        return value

    @field_validator("source_lang")
    @classmethod
    def validate_source_lang(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in SUPPORTED_SOURCE_LANGUAGES:
            raise ValueError(f"source_lang must be one of: {', '.join(SUPPORTED_SOURCE_LANGUAGES)}")
        return normalized

    @field_validator("target_lang")
    @classmethod
    def validate_target_lang(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in SUPPORTED_TARGET_LANGUAGES:
            raise ValueError(f"target_lang must be: {', '.join(SUPPORTED_TARGET_LANGUAGES)}")
        return normalized

class TranslateResponse(BaseModel):
    """
    Response model returned after a successful translation.
    Contains the original text, its translation, and provider metadata.
    """
    text: str
    translated_text: str
    source_lang: str
    target_lang: str
    provider: str
    model_name: str

class HealthResponse(BaseModel):
    """
    Health and status model for a specific translation provider.
    """
    status: str
    provider: str
    model_name: str
    model_family: str
    model_loaded: bool
    runtime_device: str
    runtime_dtype: str
