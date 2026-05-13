# Service de Traduction Anglais/Français vers Malgache

Ce projet est une API haute performance conçue pour traduire des textes de l'anglais ou du français vers le malgache (MG). Elle supporte plusieurs moteurs de traduction, allant d'APIs cloud (Gemini) à des modèles open source locaux (NLLB, Gemma, M2M100).

## Fonctionnalités

- **Traduction Multi-Source** : Supporte l'anglais et le français.
- **Détection Automatique** : Capacité à détecter la langue source du texte.
- **Providers Flexibles** :
  - **Gemini API** : Utilise les derniers modèles de Google (Gemini 2.5 Flash Lite).
  - **Modèles Locaux** : Supporte NLLB-200, Gemma 4, et les modèles Seq2Seq via Hugging Face.
- **Architecture Modulaire** : Facile d'ajouter de nouveaux providers ou modèles.
- **Optimisé pour Docker** : Prêt pour un déploiement conteneurisé.

## Structure du Projet

```text
app/
├── main.py          # Point d'entrée et configuration FastAPI
├── routes.py        # Définition des points d'accès API
├── schemas.py       # Modèles de données Pydantic
├── config.py        # Gestion de la configuration et des variables d'env
└── services/
    ├── translator.py    # Service d'orchestration de la traduction
    ├── language_utils.py # Utilitaires de détection de langue
    └── providers/       # Implémentations spécifiques (Gemini, Local, etc.)
```

## Installation

### Prérequis
- Python 3.10 ou supérieur
- (Optionnel) Docker et Docker Compose
- Une clé API Google Gemini (si vous utilisez le provider `gemini_api`)

### Installation Locale

1. **Cloner le dépôt** :
   ```bash
   git clone https://github.com/votre-repo/en-fr-to-mg-api.git
   cd en-fr-to-mg-api
   ```

2. **Installer les dépendances** :
   ```bash
   pip install -r requirements.txt
   ```

3. **Configurer l'environnement** :
   Copiez le fichier d'exemple et remplissez vos clés :
   ```bash
   cp .env.example .env
   ```

4. **Lancer l'application** :
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

## Utilisation de l'API

### Traduction par défaut
**Endpoint** : `POST /translate`
```json
{
  "text": "Hello world",
  "source_lang": "en",
  "target_lang": "mg"
}
```

### Traduction via Gemini
**Endpoint** : `POST /translate/gemini`

### Vérification de l'état
**Endpoint** : `GET /health` - Renvoie l'état de tous les providers configurés.

## Configuration Docker

Pour lancer le service avec Docker Compose :
```bash
docker-compose up --build
```
Le service sera disponible sur le port 8000.

## Maintenance

Le code a été refactorisé pour séparer clairement la logique de routage, les modèles de données et les implémentations des providers. Les commentaires sont disponibles en français pour faciliter la compréhension du fonctionnement interne.
