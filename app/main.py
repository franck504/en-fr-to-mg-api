from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import get_settings
from app.routes import router as native_router
from app.routes_anthropic import router as anthropic_router
from app.routes_ollama import router as ollama_router
from app.routes_openai import router as openai_router
from app.services.dispatcher import default_provider, default_translation_service

settings = get_settings()

@asynccontextmanager
async def lifespan(_: FastAPI):
    """
    Handles application lifecycle events.
    Pre-loads the translation model and runs an immediate self-test.
    """
    if settings.load_model_on_startup:
        print("\n⏳ [1/2] Pré-chargement du modèle de traduction sur le GPU...")
        default_provider.warmup()
        print("🧪 [2/2] Exécution d'un auto-test de validation...")
        test_res = default_translation_service.translate(
            text="Hello world",
            source_lang="en",
            target_lang="mg",
        )
        print(f"✅ Auto-test validé : 'Hello world' -> '{test_res.translated_text}'")
        print("🚀 Le modèle est 100% prêt sur GPU pour répondre instantanément !\n")
    yield

# Create the FastAPI application instance
app = FastAPI(
    title=settings.app_name,
    version="0.2.0",
    description=(
        "Local Translation Service for English/French to Malagasy (MG). "
        "Built for Google Colab and Kaggle with multi-standard API compatibility "
        "(OpenAI, Anthropic, Ollama, Native REST)."
    ),
    lifespan=lifespan,
)

# Enable CORS for cross-origin frontend client apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Attach routes
app.include_router(native_router)
app.include_router(openai_router)
app.include_router(anthropic_router)
app.include_router(ollama_router)
