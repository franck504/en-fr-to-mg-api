# Service de traduction EN/FR vers MG

Ce projet fournit un backend `FastAPI` pour traduire du texte depuis l'anglais (`en`) ou le francais (`fr`) vers le malgache (`mg`).

Le meme service expose plusieurs moteurs de traduction en parallele:

- `gemini_api` pour utiliser Gemini via API
- `local_llm` pour utiliser un modele Hugging Face seq2seq local
- `gemma4` pour utiliser Gemma 4 localement

Le but est simple:

- garder une seule API cote client
- pouvoir comparer plusieurs approches de traduction
- rester utilisable sur une machine faible grace a `Google Colab + ngrok`

## Ce Que Fait Le Backend

Le backend accepte un JSON du type:

```json
{
  "text": "Hello, how are you?",
  "source_lang": "en",
  "target_lang": "mg"
}
```

Et retourne une reponse du type:

```json
{
  "text": "Hello, how are you?",
  "translated_text": "Salama, manao ahoana ianao?",
  "source_lang": "en",
  "target_lang": "mg",
  "provider": "gemini_api",
  "model_name": "gemini-2.5-flash-lite"
}
```

Langues supportees:

- `source_lang`: `auto`, `en`, `fr`
- `target_lang`: `mg`

## Endpoints

Etat du service:

- `GET /health` retourne l'etat des trois providers
- `GET /health/gemini` retourne l'etat du provider Gemini
- `GET /health/local_llm` retourne l'etat du provider Hugging Face local
- `GET /health/gemma4` retourne l'etat du provider Gemma 4 local

Traduction:

- `POST /translate` utilise le provider par defaut defini par `PROVIDER`
- `POST /translate/gemini` force Gemini
- `POST /translate/local_llm` force le modele local Hugging Face
- `POST /translate/gemma4` force Gemma 4 local

Exemple:

```bash
curl -X POST http://127.0.0.1:8000/translate/gemini \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, how are you?",
    "source_lang": "en",
    "target_lang": "mg"
  }'
```

## Providers Disponibles

### 1. `gemini_api`

Utilise l'API Gemini. C'est aujourd'hui l'option la plus simple pour obtenir de bonnes traductions sans charger un gros modele local.

Points forts:

- bonne qualite de traduction
- aucun GPU local necessaire
- demarrage rapide

Points d'attention:

- depend d'une cle API
- depend d'un quota externe

### 2. `local_llm`

Utilise un modele Hugging Face seq2seq local, configure par:

- `HF_MODEL_NAME`
- `HF_MODEL_FAMILY`
- `HF_DEVICE`

Modeles deja testes dans ce projet:

- `facebook/nllb-200-distilled-600M`
- `facebook/nllb-200-distilled-1.3B`
- `facebook/m2m100_1.2B`

### 3. `gemma4`

Utilise `google/gemma-4-E2B-it` localement.

Points utiles:

- bon candidat pour `Colab T4`
- charge localement, sans API externe
- fonctionne mieux avec GPU

Point important:

- `local_llm` et `gemma4` se dechargent mutuellement pour eviter de saturer la VRAM dans Colab gratuit

## Variables D'Environnement Principales

Variables generales:

- `APP_NAME`
- `HOST`
- `PORT`
- `PROVIDER`
- `MODEL_CACHE_DIR`
- `LOAD_MODEL_ON_STARTUP`

Providers supportes par `PROVIDER`:

- `gemini_api`
- `hf_seq2seq`
- `local_llm`
- `local_nllb`
- `local_m2m100`
- `gemma4`

Variables Gemini:

- `GEMINI_API_KEY`
- `GEMINI_MODEL_NAME`
- `GEMINI_TEMPERATURE`
- `GEMINI_THINKING_BUDGET`
- `GEMINI_TIMEOUT_SECONDS`
- `GEMINI_MAX_RETRIES`
- `GEMINI_RETRY_DEFAULT_DELAY_SECONDS`

Variables Hugging Face local:

- `HF_MODEL_NAME`
- `HF_MODEL_FAMILY` avec `auto`, `nllb`, `m2m100`
- `HF_DEVICE` avec `auto`, `cpu`, `cuda`
- `TRANSLATION_MAX_LENGTH`

Variables Gemma 4:

- `GEMMA4_MODEL_NAME`
- `GEMMA4_DEVICE` avec `auto`, `cpu`, `cuda`
- `GEMMA4_MAX_NEW_TOKENS`

Des exemples sont fournis dans:

- `.env.example`
- `.env.colab.example`

## Demarrage Local

### Option 1. Python

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Le service sera disponible sur:

```bash
http://127.0.0.1:8000
```

### Option 2. Docker

```bash
docker compose up --build
```

## Demarrage Rapide Selon Le Provider

### Gemini

```bash
export PROVIDER=gemini_api
export GEMINI_API_KEY="votre_cle_gemini"
export GEMINI_MODEL_NAME=gemini-2.5-flash-lite
uvicorn app.main:app --reload
```

### Hugging Face local avec NLLB

```bash
export PROVIDER=hf_seq2seq
export HF_MODEL_NAME=facebook/nllb-200-distilled-1.3B
export HF_MODEL_FAMILY=nllb
export HF_DEVICE=auto
uvicorn app.main:app --reload
```

### Hugging Face local avec M2M100

```bash
export PROVIDER=hf_seq2seq
export HF_MODEL_NAME=facebook/m2m100_1.2B
export HF_MODEL_FAMILY=m2m100
export HF_DEVICE=auto
uvicorn app.main:app --reload
```

### Gemma 4 local

```bash
export PROVIDER=gemma4
export GEMMA4_MODEL_NAME=google/gemma-4-E2B-it
export GEMMA4_DEVICE=auto
export GEMMA4_MAX_NEW_TOKENS=256
uvicorn app.main:app --reload
```

## Google Colab

Si votre machine locale est faible, le mode recommande est:

- lancer le backend dans Colab
- exposer le port avec `ngrok`
- appeler ensuite l'URL publique depuis votre PC ou votre application

Fichiers utiles:

- `run_colab.py`
- `requirements-colab.txt`
- `notebooks/colab_backend_launcher.ipynb`
- `.env.colab.example`

### Workflow recommande

1. pousser le projet sur GitHub
2. ouvrir le notebook Colab
3. laisser le notebook faire `git clone` ou `git pull`
4. installer `requirements-colab.txt`
5. renseigner les variables d'environnement
6. lancer `python run_colab.py`
7. tester l'URL publique `ngrok`

### Workflow minimal

```bash
pip install -r requirements-colab.txt
export NGROK_AUTHTOKEN="votre_token_ngrok"
export ENABLE_NGROK=true
export PROVIDER=gemini_api
export GEMINI_API_KEY="votre_cle_gemini"
python run_colab.py
```

### Notes importantes pour Colab

- ne pas utiliser Docker dans Colab pour ce projet
- preferer `T4 GPU` pour `local_llm` et `gemma4`
- `Gemini` n'a pas besoin de GPU car l'inference se fait cote Google
- le premier chargement d'un modele local peut etre long
- si la session Colab s'arrete, le backend s'arrete aussi
- si vous voulez garder les modeles entre sessions, pointez `MODEL_CACHE_DIR` vers Google Drive
- le runtime `TPU v5e-1` n'est pas branche a `torch_xla` dans ce projet

## Workflow GitHub

Le flux recommande est:

1. modifier le code sur votre machine
2. faire `git add`, `git commit`, `git push`
3. dans Colab, relancer seulement la cellule de synchronisation
4. relancer le backend si du code Python a change

Exemple minimal:

```bash
git add .
git commit -m "Update translation backend"
git push
```

## Banc De Test Qualite

Le projet inclut un benchmark simple pour comparer les providers et reperer les traductions faibles.

Fichiers:

- `eval/cases.json`
- `eval/run_quality_benchmark.py`
- `eval/reports/latest_report.json`

Exemples:

Contre le provider par defaut:

```bash
python3 eval/run_quality_benchmark.py --base-url http://127.0.0.1:8000
```

Contre Gemini:

```bash
python3 eval/run_quality_benchmark.py \
  --base-url http://127.0.0.1:8000 \
  --translate-path /translate/gemini
```

Contre le modele Hugging Face local:

```bash
python3 eval/run_quality_benchmark.py \
  --base-url http://127.0.0.1:8000 \
  --translate-path /translate/local_llm
```

Contre Gemma 4:

```bash
python3 eval/run_quality_benchmark.py \
  --base-url http://127.0.0.1:8000 \
  --translate-path /translate/gemma4
```

Contre un backend Colab expose par `ngrok`:

```bash
python3 eval/run_quality_benchmark.py --base-url https://votre-url-ngrok
```

Le benchmark aide a reperer:

- les traductions vides
- les traductions trop courtes
- les pertes de phrases
- certains concepts medicaux critiques manquants

Il ne remplace pas une validation humaine.

## Structure Rapide Du Projet

- `app/main.py` contient l'API FastAPI
- `app/config.py` contient la configuration centralisee
- `app/services/providers/` contient les providers
- `run_colab.py` lance le backend dans Colab avec `ngrok`
- `eval/` contient les cas de test et le benchmark

## Limites Actuelles

- le service cible uniquement `en/fr -> mg`
- la detection automatique `en` vs `fr` reste heuristique
- la qualite depend du modele choisi
- les quotas API Gemini peuvent limiter les tests longs
- les modeles locaux peuvent etre lents ou lourds selon le runtime

## Evolutions Possibles

- ajouter une authentification pour proteger l'API
- ajouter d'autres providers
- ajouter du batch
- ajouter une file de taches pour de gros volumes
