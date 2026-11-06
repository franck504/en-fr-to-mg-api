#!/usr/bin/env python3
"""
Exemple de script client démontrant l'utilisation des 4 formats d'API supportés
avec authentification par clé API :
1. REST Natif (/translate)
2. OpenAI (/v1/chat/completions)
3. Anthropic (/v1/messages)
4. Ollama (/api/chat)

Utilisation :
    python examples/client_demo.py --url https://votre-url-ngrok.ngrok-free.app --api-key VOTRE_CLE_API
    python examples/client_demo.py --url http://127.0.0.1:8000 --api-key ma_cle --text "Bonjour docteur."
"""

import argparse
import json
import os
import sys
from urllib import error, request


def build_headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        headers["X-API-Key"] = api_key
    return headers


def post_json(endpoint: str, payload: dict, api_key: str | None) -> dict:
    req = request.Request(
        url=endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=build_headers(api_key),
        method="POST",
    )
    with request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_json(endpoint: str, api_key: str | None = None) -> dict:
    req = request.Request(
        url=endpoint,
        headers=build_headers(api_key),
        method="GET",
    )
    with request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def test_native(base_url: str, text: str, api_key: str | None):
    print("\n--- 1. Test Format REST Natif (/translate) ---")
    payload = {"text": text, "source_lang": "auto", "target_lang": "mg"}
    res = post_json(f"{base_url}/translate", payload, api_key)
    print(f"Source   : {res.get('text')} (langue: {res.get('source_lang')})")
    print(f"Malgache : {res.get('translated_text')}")
    print(f"Moteur   : {res.get('provider')} ({res.get('model_name')})")


def test_openai(base_url: str, text: str, api_key: str | None):
    print("\n--- 2. Test Format OpenAI (/v1/chat/completions) ---")
    payload = {
        "model": "nllb",
        "messages": [{"role": "user", "content": text}],
    }
    res = post_json(f"{base_url}/v1/chat/completions", payload, api_key)
    translated = res["choices"][0]["message"]["content"]
    print(f"Malgache : {translated}")
    print(f"ID Réponse: {res.get('id')}")


def test_anthropic(base_url: str, text: str, api_key: str | None):
    print("\n--- 3. Test Format Anthropic (/v1/messages) ---")
    payload = {
        "model": "nllb",
        "max_tokens": 256,
        "messages": [{"role": "user", "content": text}],
    }
    res = post_json(f"{base_url}/v1/messages", payload, api_key)
    translated = res["content"][0]["text"]
    print(f"Malgache : {translated}")
    print(f"Stop Reason: {res.get('stop_reason')}")


def test_ollama(base_url: str, text: str, api_key: str | None):
    print("\n--- 4. Test Format Ollama (/api/chat) ---")
    payload = {
        "model": "nllb",
        "messages": [{"role": "user", "content": text}],
    }
    res = post_json(f"{base_url}/api/chat", payload, api_key)
    translated = res["message"]["content"]
    print(f"Malgache : {translated}")


def main():
    parser = argparse.ArgumentParser(description="Test des endpoints de traduction vers le malgache.")
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000",
        help="URL de base du serveur (ex: https://xxx.ngrok-free.app ou http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("API_KEY", "your_secret_api_key_here"),
        help="Clé API configurée sur le serveur (dans Colab/Kaggle Secrets)",
    )
    parser.add_argument(
        "--text",
        default="Hello, how are you? The doctor recommends to take this medicine twice a day.",
        help="Texte en anglais ou français à traduire en malgache.",
    )
    args = parser.parse_args()
    base_url = args.url.rstrip("/")

    print("=" * 65)
    print(f"🌍 Test du serveur : {base_url}")
    print(f"🔑 Clé API         : {'*' * (len(args.api_key) - 4) + args.api_key[-4:] if len(args.api_key) > 4 else '****'}")
    print(f"📝 Texte source    : {args.text}")
    print("=" * 65)

    try:
        health = get_json(f"{base_url}/health")
        print(f"🩺 État serveur : {health.get('status')} (Default: {health.get('default_provider')})")
    except Exception as exc:
        print(f"⚠️ Avertissement : Impossible de joindre /health ({exc}). Vérifiez que le serveur est bien lancé.")

    try:
        test_native(base_url, args.text, args.api_key)
        test_openai(base_url, args.text, args.api_key)
        test_anthropic(base_url, args.text, args.api_key)
        test_ollama(base_url, args.text, args.api_key)
        print("\n" + "=" * 65)
        print("✅ Tous les tests d'API sécurisés ont réussi avec succès !")
        print("=" * 65)
    except error.HTTPError as exc:
        print(f"\n❌ Erreur HTTP {exc.code} : {exc.read().decode('utf-8')}")
        sys.exit(1)
    except Exception as exc:
        print(f"\n❌ Erreur inattendue : {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
