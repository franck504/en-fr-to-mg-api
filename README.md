# 🇲🇬 Service de Traduction Anglais/Français vers Malgache (MG)

API haute performance pour traduire des textes de l'anglais ou du français vers le malgache (MG), propulsée **exclusivement par des modèles d'IA open source locaux** (NLLB-200, M2M100, Gemma 4).

L'infrastructure cible est **Google Colab et Kaggle** (GPU T4/P100 gratuit). L'API implémente nativement les protocoles des plus grands standards d'IA (**OpenAI**, **Anthropic**, **Ollama**), ce qui permet de l'intégrer directement dans n'importe quelle application cliente ou framework (LangChain, LlamaIndex, etc.) sans modifier votre code client.

---

## 🚀 Fonctionnalités Clés

- **100% Modèles Locaux & Gratuit** :
  - **Meta NLLB-200** (`facebook/nllb-200-distilled-600M` ou `1.3B`)
  - **Meta M2M100** (`facebook/m2m100_1.2B`)
  - **Google Gemma 4** (`google/gemma-4-E2B-it`)
  - Aucun appel vers une API cloud payante ou propriétaire.
- **Détection Automatique de Langue** : Détecte automatiquement si la source est en anglais ou en français.
- **Compatibilité Multi-Standard** :
  - Format **OpenAI** : `/v1/chat/completions`, `/v1/models`
  - Format **Anthropic** : `/v1/messages`
  - Format **Ollama** : `/api/chat`, `/api/generate`, `/api/tags`
  - Format **Natif REST** : `/translate`, `/translate/local_llm`, `/translate/gemma4`, `/health`
- **Infrastructure Colab & Kaggle** :
  - Déploiement en 1 clic via Jupyter Notebook avec GPU gratuit.
  - Tunnel **ngrok** intégré pour obtenir une URL HTTPS publique accessible partout.
- **Gestion Mémoire GPU** : Déchargement automatique et nettoyage du cache CUDA (`empty_cache`) entre les familles de modèles pour éviter les dépassements de mémoire (OOM).

---

## 📁 Structure du Projet

```text
en-fr-to-mg-api/
├── app/
│   ├── main.py                  # Point d'entrée FastAPI & middleware CORS
│   ├── config.py                # Configuration Pydantic & variables d'environnement
│   ├── schemas.py               # Schémas natifs de validation Pydantic
│   ├── schemas_compat.py        # Schémas de compatibilité OpenAI, Anthropic, Ollama
│   ├── routes.py                # Endpoints natifs (/translate, /health, /)
│   ├── routes_openai.py         # Endpoints compatibles OpenAI (/v1/chat/completions, /v1/models)
│   ├── routes_anthropic.py      # Endpoints compatibles Anthropic (/v1/messages)
│   ├── routes_ollama.py         # Endpoints compatibles Ollama (/api/chat, /api/generate, /api/tags)
│   └── services/
│       ├── dispatcher.py        # Gestionnaire de mémoire GPU & résolution de modèle
│       ├── translator.py        # Factories des providers locaux & service de traduction
│       ├── language_utils.py    # Détection heuristique de langue source (FR / EN)
│       └── providers/
│           ├── base.py          # Classe abstraite TranslationProvider & TranslationResult
│           ├── hf_seq2seq.py    # Provider Hugging Face (NLLB, M2M100)
│           └── gemma4_local.py  # Provider Gemma 4
├── notebooks/
│   ├── colab_backend_launcher.ipynb   # Notebook dédié Google Colab (Secrets via google.colab.userdata)
│   └── kaggle_backend_launcher.ipynb  # Notebook dédié Kaggle (Secrets via kaggle_secrets)
├── examples/
│   └── client_demo.py           # Script de test des 4 protocoles en une seule commande
├── eval/
│   ├── cases.json               # 24 cas de test (général et médical/dermatologie)
│   └── run_quality_benchmark.py # Script de validation de qualité et mesure de latence
├── run_colab.py                 # Script serveur FastAPI & tunnel ngrok (Colab/Kaggle)
├── requirements.txt             # Dépendances Python pour serveur local
├── requirements-colab.txt       # Dépendances légères pour Colab / Kaggle
└── .env.example                 # Variables d'environnement d'exemple
```

---

## ⚡ Déploiement Cloud Gratuit (Colab ou Kaggle)

Les deux notebooks utilisent le **gestionnaire de Secrets natif** de chaque plateforme pour protéger votre token ngrok sans jamais l'écrire en clair dans le code.

### Option A : Déploiement sur Google Colab
1. Ouvrez [`notebooks/colab_backend_launcher.ipynb`](notebooks/colab_backend_launcher.ipynb) dans [Google Colab](https://colab.research.google.com).
2. Activez le GPU : *Exécution* > *Changer le type d'exécution* > **GPU T4**.
3. Ajoutez votre token dans les Secrets : cliquez sur l'icône **Clé 🔑 (Secrets)** à gauche, ajoutez `NGROK_AUTHTOKEN` avec votre token gratuit depuis [dashboard.ngrok.com](https://dashboard.ngrok.com), et cochez *Accès au notebook*.
4. Exécutez les cellules (`Exécution > Tout exécuter`) : l'URL publique s'affiche !

### Option B : Déploiement sur Kaggle
1. Créez un nouveau Notebook sur [Kaggle](https://www.kaggle.com) et importez [`notebooks/kaggle_backend_launcher.ipynb`](notebooks/kaggle_backend_launcher.ipynb).
2. Dans le panneau de droite (*Session options*) :
   - Cochez **Internet ON**.
   - Dans *Accelerator*, choisissez **GPU T4 x2** ou **GPU P100**.
3. Dans le menu supérieur : **Add-ons** > **Secrets** > ajoutez `NGROK_AUTHTOKEN` avec votre token.
4. Exécutez le notebook : l'URL publique s'affiche !

---

## 💻 Exécution en Local (avec GPU ou CPU)

### Prérequis
- Python 3.10 ou supérieur
- PyTorch avec support CUDA (si GPU disponible)

```bash
# 1. Cloner le projet
git clone https://github.com/franck504/en-fr-to-mg-api.git
cd en-fr-to-mg-api

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Configurer l'environnement
cp .env.example .env

# 4. Lancer le serveur
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 🔌 Utilisation : Exemples Multi-Standards

Remplacez `http://localhost:8000` par votre URL ngrok si déployé sur Colab/Kaggle.

### 1. Via le SDK Officiel OpenAI (Python)
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
)

response = client.chat.completions.create(
    model="nllb",  # ou "gemma4", "default"
    messages=[
        {"role": "user", "content": "Hello, how are you today?"}
    ],
)
print(response.choices[0].message.content)
# -> Traduction en Malgache
```

### 2. Via le SDK Officiel Anthropic (Python)
```python
from anthropic import Anthropic

client = Anthropic(
    base_url="http://localhost:8000",
    api_key="not-needed",
)

response = client.messages.create(
    model="nllb",
    max_tokens=256,
    messages=[
        {"role": "user", "content": "Bonjour, comment allez-vous aujourd'hui ?"}
    ],
)
print(response.content[0].text)
# -> Traduction en Malgache
```

### 3. Via le SDK Ollama (Python)
```python
import ollama

client = ollama.Client(host="http://localhost:8000")

# Chat
response = client.chat(
    model="nllb",
    messages=[{"role": "user", "content": "Please apply this cream twice daily."}],
)
print(response["message"]["content"])

# Generate
gen = client.generate(model="nllb", prompt="Good morning my friend")
print(gen["response"])
```

### 4. Via LangChain
```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
    model="nllb",
)
result = llm.invoke("Unfortunately, you have eczema. Keep your skin moisturized.")
print(result.content)
```

### 5. Via l'API REST Native (cURL)
```bash
curl -X POST http://localhost:8000/translate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The patient shows moderate skin inflammation.",
    "source_lang": "auto",
    "target_lang": "mg"
  }'
```

---

## 🧪 Tests & Benchmark Qualité

Un banc d'évaluation couvrant 24 cas de test (généraux et médicaux/dermatologie) est inclus :

```bash
python eval/run_quality_benchmark.py --base-url http://localhost:8000 --translate-path /translate
```

Le script vérifie la latence, la conservation des phrases et la présence des termes médicaux essentiels en malgache.
