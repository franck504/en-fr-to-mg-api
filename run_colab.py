import os
from pathlib import Path

import uvicorn


def as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def configure_environment() -> tuple[str, int]:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    model_cache_dir = os.getenv("MODEL_CACHE_DIR", "/content/hf_models")

    Path(model_cache_dir).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MODEL_CACHE_DIR", model_cache_dir)
    os.environ.setdefault("LOAD_MODEL_ON_STARTUP", "false")

    return host, port


def start_ngrok_tunnel(port: int):
    try:
        import ngrok
    except ImportError as exc:
        raise RuntimeError(
            "ngrok is missing. Install requirements-colab.txt before starting on Colab."
        ) from exc

    auth_token = os.getenv("NGROK_AUTHTOKEN")
    if not auth_token:
        raise RuntimeError(
            "NGROK_AUTHTOKEN is missing. Create a free ngrok account and set the token in Colab."
        )

    listener = ngrok.forward(port, authtoken=auth_token)
    public_url = listener.url()
    print(f"Public URL: {public_url}")
    print(f"Docs URL: {public_url}/docs")
    return listener


def main() -> None:
    host, port = configure_environment()
    enable_ngrok = as_bool(os.getenv("ENABLE_NGROK", "true"))
    tunnel = None

    try:
        if enable_ngrok:
            tunnel = start_ngrok_tunnel(port)

        print(f"Local URL: http://127.0.0.1:{port}")
        uvicorn.run("app.main:app", host=host, port=port, reload=False)
    finally:
        if tunnel is not None:
            try:
                tunnel.close()
            except AttributeError:
                pass


if __name__ == "__main__":
    main()
