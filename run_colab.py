import os
from pathlib import Path
import threading
import time
import uvicorn


def as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def start_keep_alive_thread():
    """
    Runs in the background and prints a heartbeat every 3 minutes.
    This maintains stdout activity in Colab/Kaggle and prevents kernel idle suspension.
    """
    def _worker():
        start_time = time.time()
        while True:
            time.sleep(180)  # Every 3 minutes
            uptime_min = int((time.time() - start_time) / 60)
            vram_info = ""
            try:
                import torch
                if torch.cuda.is_available():
                    alloc = torch.cuda.memory_allocated() / 1024 / 1024
                    vram_info = f" | GPU VRAM: {alloc:.1f} MB"
            except Exception:
                pass
            print(f"💓 [Keep-Alive Heartbeat] Serveur actif | Uptime: {uptime_min} min{vram_info}", flush=True)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()


def resolve_api_key() -> str | None:
    """
    Attempts to retrieve the API key from:
    1. Environment variable API_KEY
    2. Google Colab Secrets (userdata)
    3. Kaggle Secrets (UserSecretsClient)
    4. Local configuration / .env file
    """
    token = os.getenv("API_KEY")
    if token and token != "replace_me":
        return token

    # Attempt retrieval from Google Colab Secrets
    try:
        from google.colab import userdata
        token = userdata.get("API_KEY")
        if token:
            return token
    except Exception:
        pass

    # Attempt retrieval from Kaggle Secrets
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        token = user_secrets.get_secret("API_KEY")
        if token:
            return token
    except Exception:
        pass

    # Attempt retrieval from local .env via Pydantic settings
    try:
        from app.config import get_settings
        cfg_key = get_settings().api_key
        if cfg_key and cfg_key != "replace_me":
            return cfg_key
    except Exception:
        pass

    return None


def resolve_ngrok_token() -> str | None:
    """
    Attempts to retrieve the ngrok authtoken from:
    1. Environment variable NGROK_AUTHTOKEN
    2. Google Colab Secrets (userdata)
    3. Kaggle Secrets (UserSecretsClient)
    4. Local .env file
    """
    token = os.getenv("NGROK_AUTHTOKEN")
    if token and token != "replace_me":
        return token

    # Attempt retrieval from Google Colab Secrets
    try:
        from google.colab import userdata
        token = userdata.get("NGROK_AUTHTOKEN")
        if token:
            return token
    except Exception:
        pass

    # Attempt retrieval from Kaggle Secrets
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        token = user_secrets.get_secret("NGROK_AUTHTOKEN")
        if token:
            return token
    except Exception:
        pass

    # Attempt retrieval from .env file
    if os.path.exists(".env"):
        try:
            with open(".env", "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip().startswith("NGROK_AUTHTOKEN="):
                        val = line.strip().split("=", 1)[1].strip()
                        if val and val != "replace_me":
                            return val
        except Exception:
            pass

    return None


def configure_environment() -> tuple[str, int]:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))

    # Auto-detect Kaggle vs Colab environment
    if os.path.exists("/kaggle/working") or "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
        default_cache = "/kaggle/working/hf_models"
    else:
        default_cache = "/content/hf_models"

    model_cache_dir = os.getenv("MODEL_CACHE_DIR", default_cache)
    Path(model_cache_dir).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MODEL_CACHE_DIR", model_cache_dir)
    # Preload the model on startup so the first external request is instant
    os.environ.setdefault("LOAD_MODEL_ON_STARTUP", "true")

    # Verify and set API_KEY from Secrets
    api_key = resolve_api_key()
    if not api_key:
        raise RuntimeError(
            "API_KEY introuvable. "
            "Sur Google Colab : ajoutez le secret 'API_KEY' dans le volet Secrets (icône clé 🔑 à gauche). "
            "Sur Kaggle : ajoutez le secret 'API_KEY' via le menu Add-ons > Secrets."
        )
    os.environ["API_KEY"] = api_key

    return host, port


def _close_existing_ngrok_listeners(ngrok_module) -> int:
    closed = 0
    listeners_attr = getattr(ngrok_module, "listeners", None)
    listeners = None
    if callable(listeners_attr):
        try:
            listeners = listeners_attr()
        except Exception:
            listeners = None
    elif listeners_attr is not None:
        listeners = listeners_attr

    if listeners is not None:
        try:
            for listener in list(listeners):
                try:
                    listener.close()
                    closed += 1
                except Exception:
                    continue
        except Exception:
            pass

    kill_fn = getattr(ngrok_module, "kill", None)
    if callable(kill_fn):
        try:
            kill_fn()
        except Exception:
            pass

    return closed


def start_ngrok_tunnel(port: int):
    try:
        import ngrok
    except ImportError as exc:
        raise RuntimeError(
            "ngrok is missing. Install requirements-colab.txt before starting."
        ) from exc

    auth_token = resolve_ngrok_token()
    if not auth_token:
        raise RuntimeError(
            "NGROK_AUTHTOKEN introuvable. "
            "Sur Google Colab : ajoutez le secret 'NGROK_AUTHTOKEN' dans le volet Secrets (icône clé 🔑 à gauche). "
            "Sur Kaggle : ajoutez le secret 'NGROK_AUTHTOKEN' via le menu Add-ons > Secrets."
        )

    try:
        domain = os.getenv("NGROK_DOMAIN")
        kwargs = {"authtoken": auth_token}
        if domain:
            kwargs["domain"] = domain

        listener = ngrok.forward(port, **kwargs)
    except Exception as exc:
        message = str(exc)
        if "ERR_NGROK_334" in message or "already online" in message:
            closed = _close_existing_ngrok_listeners(ngrok)
            try:
                listener = ngrok.forward(port, authtoken=auth_token)
            except Exception as retry_exc:
                raise RuntimeError(
                    "ngrok n'a pas pu démarrer car une session précédente est encore active. "
                    "Arrêtez l'ancienne cellule / session et réessayez. "
                    f"Listeners locaux fermés : {closed}. Erreur d'origine : {retry_exc}"
                ) from retry_exc
        else:
            raise RuntimeError(f"ngrok n'a pas pu démarrer : {exc}") from exc

    public_url = listener.url()
    return listener, public_url


def print_banner(public_url: str | None, port: int):
    base = public_url if public_url else f"http://127.0.0.1:{port}"
    print("=" * 70)
    print("🚀 EN/FR -> Malagasy (MG) Translation Service (Local Models)")
    print("=" * 70)
    print(f"📡 Base URL:          {base}")
    print(f"📖 Swagger Docs:      {base}/docs")
    print(f"🔍 Health Check:      {base}/health")
    print(f"🔒 Authentification:  Actif (Clé API requise via header 'Authorization' ou 'X-API-Key')")
    print(f"💓 Keep-Alive:        Actif (Heartbeat toutes les 3 minutes)")
    print("-" * 70)
    print("API Endpoints & Compatibilités :")
    print(f"  • Native REST:  POST {base}/translate")
    print(f"  • OpenAI:       POST {base}/v1/chat/completions")
    print(f"  • Anthropic:    POST {base}/v1/messages")
    print(f"  • Ollama:       POST {base}/api/chat")
    print("=" * 70)


def main() -> None:
    host, port = configure_environment()
    enable_ngrok = as_bool(os.getenv("ENABLE_NGROK", "true"))
    tunnel = None
    public_url = None

    try:
        if enable_ngrok:
            tunnel, public_url = start_ngrok_tunnel(port)

        print_banner(public_url, port)
        # Start heartbeat to keep Colab/Kaggle active
        start_keep_alive_thread()
        uvicorn.run("app.main:app", host=host, port=port, reload=False)
    finally:
        if tunnel is not None:
            try:
                tunnel.close()
            except AttributeError:
                pass


if __name__ == "__main__":
    main()
