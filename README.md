# MCP Podcast Summarizer

Ce projet implémente un système d'analyse de podcasts utilisant le protocole MCP (Model Context Protocol) pour faciliter l'interaction entre le client et le serveur d'analyse.

## Table des matières
1. [Installation et utilisation](#1-installation-et-utilisation)
2. [Architecture du projet](#2-architecture-du-projet)
3. [Fonctionnalités MCP](#3-fonctionnalités-mcp)
4. [Sécurité](#4-sécurité)

## 1. Installation et utilisation

### Installation

1. Clonez le dépôt :
   ```bash
   git clone https://github.com/souhailelaissaoui/mcp-podcast-summarizer.git
   cd mcp-podcast-summarizer
   ```

2. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

3. Configurez les variables d'environnement :
   - Renommez le fichier `.env.example` en `.env` (si nécessaire)
   - Mettez à jour les clés API dans le fichier `.env` :
     ```
     ANTHROPIC_API_KEY=VOTRE_CLE_API_ANTHROPIC
     SERVER_MCP_API_KEY=VOTRE_CLE_API_MCP_SERVEUR
     ```
   - Assurez-vous que la clé dans `servers_config.json` correspond à votre clé serveur :
     ```json
     "env": {
       "CLIENT_MCP_API_KEY": "VOTRE_CLE_API_MCP_CLIENT"
     }
     ```

### Utilisation

1. Lancez l'application :
   ```bash
   python main.py
   ```

2. Une fois l'interface de chat lancée, vous pouvez demander une transcription et un résumé :
   ```
   summarize https://www.youtube.com/watch?v=JkZ32SbDrlw&t=3s
   ```

3. Pour quitter l'application, tapez `exit` ou `quit`.

## 2. Architecture du projet

### Structure des Fichiers

```
├── .env                  # Variables d'environnement (API keys)
├── .gitignore            # Fichiers à ignorer dans Git
├── README.md             # Documentation du projet
├── main.py               # Point d'entrée et client MCP
├── requirements.txt      # Dépendances Python
├── security.py           # Gestion de l'authentification
├── server.py             # Serveur MCP et définition des outils
├── servers_config.json   # Configuration des serveurs MCP
├── temp/                 # Dossier temporaire pour les fichiers audio
└── transcription.py      # Fonctions de transcription avec Whisper
```

### Description des Composants

#### 1. `main.py`
Point d'entrée principal de l'application. Ce fichier contient :
- La classe `Configuration` pour gérer les variables d'environnement
- La classe `Server` pour gérer les connexions aux serveurs MCP
- La classe `Tool` pour représenter les outils disponibles
- La classe `LLMClient` pour communiquer avec Claude (Anthropic)
- La classe `ChatSession` pour orchestrer l'interaction entre l'utilisateur, le LLM et les outils

#### 2. `server.py`
Implémentation du serveur MCP avec FastMCP. Ce fichier définit :
- L'outil `transcribe_audio` qui permet de télécharger et transcrire des vidéos YouTube
- Un décorateur `handle_errors` pour la gestion des erreurs

#### 3. `transcription.py`
Contient les fonctions de traitement audio :
- `download_audio` : télécharge l'audio d'une vidéo YouTube avec yt_dlp
- `transcribe_audio_file` : transcrit un fichier audio avec le modèle Whisper

#### 4. `security.py`
Gère l'authentification et la sécurité :
- Décorateur `require_api_key` pour protéger l'accès aux outils
- Validation des clés API entre client et serveur

#### 5. `servers_config.json`
Configuration des serveurs MCP, incluant :
- Commandes pour démarrer les serveurs
- Arguments de ligne de commande
- Variables d'environnement (dont les clés API)

### Flux de Données

1. L'utilisateur lance `main.py` qui initialise le client MCP
2. Le client se connecte au serveur MCP défini dans `servers_config.json`
3. Le serveur expose l'outil `transcribe_audio` via le protocole MCP
4. L'utilisateur envoie une requête qui est analysée par Claude
5. Si nécessaire, Claude appelle l'outil `transcribe_audio` via MCP
6. L'outil télécharge la vidéo et utilise Whisper pour la transcription
7. Le résultat est renvoyé à Claude qui génère une réponse pour l'utilisateur

## 3. Fonctionnalités MCP

### 3.1 Utilisation du SDK MCP Officiel (Python)

Le projet utilise le SDK MCP officiel en Python à travers plusieurs composants :

```python
# Dans server.py
from mcp.server.fastmcp import FastMCP

# Dans main.py
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
```

### 3.2 Exposition d'Outils Fonctionnels

Le système expose un outil fonctionnel pour l'analyse de podcasts :

```python
@mcp.tool()
@handle_errors
@require_api_key
def transcribe_audio(url):
    """
    Télécharge l'audio d'une vidéo YouTube et le transcrit.
    """
    # Implémentation de la transcription avec Whisper
```

Cet outil exécute deux fonctions spécifiques :
- Téléchargement audio via `yt_dlp`
- Transcription via le modèle Whisper

### 3.3 Implémentation des Mécanismes MCP

### Liste des Outils (list_tools)
```python
# Dans la classe Server
async def list_tools(self) -> list[Any]:
    if not self.session:
        raise RuntimeError(f"Server {self.name} not initialized")
    
    tools_response = await self.session.list_tools()
    tools = []
    
    for item in tools_response:
        if isinstance(item, tuple) and item[0] == "tools":
            tools.extend(
                Tool(tool.name, tool.description, tool.inputSchema)
                for tool in item[1]
            )
    
    return tools
```

### Appel des Outils (call_tool)
```python
# Dans la classe Server
async def execute_tool(self, tool_name, arguments, retries=2, delay=1.0, timeout=120.0):
    if not self.session:
        raise RuntimeError(f"Server {self.name} not initialized")
        
    # Création de la tâche d'exécution d'outil
    tool_task = asyncio.create_task(self.session.call_tool(tool_name, arguments))
    
    # Attente de la réponse avec gestion du timeout
    result = await asyncio.wait_for(tool_task, timeout=timeout)
    
    return result
```

### 3.4 Gestion des Erreurs et Exceptions

Le système implémente plusieurs mécanismes de gestion d'erreurs :

```python
# Décorateur d'erreur générique
def handle_errors(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            # Vérification si le résultat contient déjà une erreur
            if isinstance(result, dict) and not result.get("success", True):
                return {
                    "success": False,
                    "error": result.get("error", "Unknown error"),
                    "error_type": "tool_error",
                    "human_readable": f"The tool '{func.__name__}' encountered an error: {result.get('error')}."
                }
            return result
        except Exception as e:
            # Capture et formatage des exceptions
            return {
                "success": False,
                "error": str(e),
                "error_type": "exception",
                "human_readable": f"An exception occurred while running '{func.__name__}': {str(e)}."
            }
    return wrapper
```

Gestion des timeouts et tentatives multiples :
```python
# Gestion des timeouts
try:
    result = await asyncio.wait_for(tool_task, timeout=timeout)
except asyncio.TimeoutError:
    tool_task.cancel()
    await asyncio.shield(self.session.cancel_tool())
    return {"success": False, "error": f"Tool execution timed out after {timeout} seconds"}

# Gestion des tentatives
attempt = 0
while attempt < retries:
    try:
        # Exécution de l'outil
    except Exception as e:
        attempt += 1
        if attempt < retries:
            await asyncio.sleep(delay)
```

## 4. Sécurité

### 4.1 Mesures de Sécurité de Base

Le système implémente un mécanisme d'authentification par clé API :

```python
# Décorateur d'authentification
def require_api_key(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Si SERVER_API_KEY n'est pas défini, ignorer l'authentification
        if not SERVER_API_KEY:
            return func(*args, **kwargs)
        
        # Récupération de la clé API du client
        client_api_key = os.environ.get('CLIENT_MCP_API_KEY', '')
        
        # Comparaison de la clé API du client avec celle du serveur
        if client_api_key and client_api_key == SERVER_API_KEY:
            return func(*args, **kwargs)
        
        # Si l'authentification échoue
        logging.warning(f"Unauthorized access attempt to {func.__name__}")
        return {
            "success": False,
            "error": "Unauthorized access. Invalid or missing API key.",
            "error_type": "authentication_error",
            "human_readable": "Authentication failed. Please provide a valid SERVER_MCP_API_KEY."
        }
    
    return wrapper
```

Cette implémentation sécurise les points suivants :
- Authentification par clé API
- Prévention des appels non autorisés
- Messages d'erreur explicites sans révéler d'informations sensibles
