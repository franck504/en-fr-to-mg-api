from pydantic import BaseModel, Field, field_validator

# Langues prises en charge
SUPPORTED_SOURCE_LANGUAGES = {"auto", "en", "fr"}
SUPPORTED_TARGET_LANGUAGES = {"mg"}

class TranslateRequest(BaseModel):
    """
    Modèle de requête pour une demande de traduction.
    On valide la longueur du texte et les langues demandées.
    """
    text: str = Field(..., min_length=1, max_length=5000)
    source_lang: str = Field(default="auto")
    target_lang: str = Field(default="mg")

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Le texte à traduire ne peut pas être vide.")
        return value

    @field_validator("source_lang")
    @classmethod
    def validate_source_lang(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in SUPPORTED_SOURCE_LANGUAGES:
            raise ValueError(f"La langue source doit être l'une des suivantes : {', '.join(SUPPORTED_SOURCE_LANGUAGES)}")
        return normalized

    @field_validator("target_lang")
    @classmethod
    def validate_target_lang(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in SUPPORTED_TARGET_LANGUAGES:
            raise ValueError(f"La langue cible doit être : {', '.join(SUPPORTED_TARGET_LANGUAGES)}")
        return normalized

class TranslateResponse(BaseModel):
    """
    Réponse renvoyée après une traduction réussie.
    Contient le texte original, sa traduction et des métadonnées sur le provider utilisé.
    """
    text: str
    translated_text: str
    source_lang: str
    target_lang: str
    provider: str
    model_name: str

class HealthResponse(BaseModel):
    """
    État de santé d'un provider de traduction spécifique.
    """
    status: str
    provider: str
    model_name: str
    model_family: str
    model_loaded: bool
    runtime_device: str
    runtime_dtype: str
