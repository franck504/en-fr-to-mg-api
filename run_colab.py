import os
from pathlib import Path
import uvicorn


def as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def resolve_ngrok_token() -> str | None:
    """
    Attempts to retrieve the ngrok authtoken from:
    1. Environment variable NGROK_AUTHTOKEN
    2. Google Colab Secrets (userdata)
    3. Kaggle Secrets (UserSecretsClient)
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
    os.environ.setdefault("LOAD_MODEL_ON_STARTUP", "false")

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
        listener = ngrok.forward(port, authtoken=auth_token)
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
        uvicorn.run("app.main:app", host=host, port=port, reload=False)
    finally:
        if tunnel is not None:
            try:
                tunnel.close()
            except AttributeError:
                pass


if __name__ == "__main__":
    main()
