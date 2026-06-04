# RAG Challenge

API RAG con FastAPI + OpenAI + ChromaDB. Responde preguntas sobre un documento usando Retrieval-Augmented Generation via `POST /ask`.

## Estructura del proyecto

```
challenge_RAG/
├── app/
│   ├── main.py                       # Entry point FastAPI
│   ├── api/
│   │   └── routes.py                 # POST /ask endpoint
│   ├── core/
│   │   ├── config.py                 # Variables de entorno y constantes
│   │   └── prompt.py                 # Templates del system/user prompt
│   ├── services/
│   │   ├── embeddings.py             # Embeddings via OpenAI
│   │   ├── vector_store.py           # Queries a ChromaDB
│   │   └── llm.py                    # Chat LLM + deteccion de idioma
│   └── ingestion/
│       └── ingest.py                 # Carga, split y persistencia del documento
├── documento/
│   └── documento.docx                # Documento fuente
├── requirements.txt
├── .env.example
├── Dockerfile
└── RAG_Collection.postman_collection.json
```

## Requisitos previos

- Docker
- OpenAI API Key

## Instalacion y ejecucion

```bash
git clone <repo_url>
cd challenge_RAG
cp .env.example .env    # completar con tu OPENAI_API_KEY
```

> ⚠️ La API key debe estar sin comillas: `OPENAI_API_KEY=sk-proj-tukey`

```bash
docker build -t challenge-rag .
docker run --env-file .env -p 8000:8000 challenge-rag
```

La ingesta del documento y el startup de la API ocurren automaticamente dentro del contenedor.

## Sin Docker (alternativa)

```bash
python -m venv .venv

# Linux / Mac
source .venv/bin/activate

# PowerShell
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
python -m app.ingestion.ingest
uvicorn app.main:app --reload
```

## Swagger UI

`http://localhost:8000/docs`

## Probar desde la terminal

**Linux / Mac / Git Bash:**

```bash
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d '{"user_name":"John Doe","question":"COLOCA_TU_PREGUNTA"}'
```

**PowerShell:**

```powershell
Invoke-RestMethod -Uri http://localhost:8000/ask -Method Post -Body '{"user_name":"John Doe","question":"COLOCA_TU_PREGUNTA"}' -ContentType "application/json"
```

### Preguntas de ejemplo (reemplazar `COLOCA_TU_PREGUNTA`)

| Idioma | Preguntas |
|---|---|
| Espanol | `¿Quien es Zara?` |
| English | `What did Emma decide to do?`, `What is the name of the magical flower?` |
| Portugues | `Qual é o nome da flor mágica?` |
| Fuera de contexto | `Who is Batman?`, `¿Quien es Batman?`, `Quem é Batman?` |

## Decisiones técnicas

- **chunk_size=500 / overlap=50**: el documento tiene 5 párrafos temáticamente distintos, este tamaño garantiza que cada historia quede en un único chunk.
- **temperature=0**: asegura determinismo, el LLM siempre genera la misma respuesta ante la misma pregunta.
- **text-embedding-3-small**: mejor relación costo/performance para embeddings en OpenAI.
- **n_results=1**: se recupera solo el chunk más relevante para evitar ruido en el contexto.
- **system/user message separation**: las reglas del prompt van en `system` para que el LLM les dé máxima prioridad sobre el contenido del contexto.


