#Nexus RAG System - Quickstart

This is a production-grade, no-framework RAG (Retrieval-Augmented Generation) system built entirely from scratch with FastAPI, Postgres (Neon + pgvector), and Streamlit.

## 1. Prerequisites
- Python 3.11+ (or Docker)
- A Gemini API Key (or Groq API Key)
- A Postgres database with the pgvector extension, e.g. a free Neon project. Use its **direct** connection string (host without `-pooler`).

## 2. Environment Setup

Create a `.env` file in the root directory:
```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key_here
# Optional: GROQ_API_KEY=your_groq_api_key_here
ADMIN_API_KEY=your_admin_api_key_here
DATABASE_URL=postgresql://user:password@ep-xxxx.region.aws.neon.tech/neondb?sslmode=require
JINA_API_KEY=your_jina_api_key
```
See `.env.example` for every variable.

Create the schema (once per database, and again after pulling new migrations):
```bash
alembic upgrade head
```

## 3. Quick Run (Docker: API + worker)
Ingestion runs as two processes: the API (validates requests, serves queries) and a worker (fetches, parses, chunks and embeds documents). They are two separate images so the API stays free of document-parsing dependencies. Both need the same `.env`.
```bash
docker build -f Dockerfile.api -t nexus-rag-api .
docker build -f Dockerfile.worker -t nexus-rag-worker .
docker run --env-file .env nexus-rag-api alembic upgrade head   # release step: migrate first

docker run --env-file .env -p 8000:8000 nexus-rag-api
docker run --env-file .env nexus-rag-worker   # in a second terminal
```
The images deliberately contain no `.env`, so pass it at run time. Docker's `--env-file` keeps quotes literally, so write values unquoted. Uploading a document without a running worker leaves its job at `status: "queued"` indefinitely — start the worker before testing ingestion.

Issue a workspace key (there is no open sign-up):
```bash
curl -X POST http://localhost:8000/admin/keys -H "RAG-API-KEY: <your ADMIN_API_KEY>"
```
- API available at: [http://localhost:8000](http://localhost:8000)
- API docs at: [http://localhost:8000/docs](http://localhost:8000/docs)

You can run the Streamlit UI via Docker:
```bash
docker build -t nexus-rag-ui -f Dockerfile.ui .
docker run -p 8080:8080 -e API_BASE_URL=http://host.docker.internal:8000 nexus-rag-ui
```
- UI available at: [http://localhost:8080](http://localhost:8080)

## 4. Local Development Run

If you prefer to run it locally without Docker:

### Install dependencies
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### Start the API backend
```powershell
uvicorn src.api.main:app --reload --port 8000
```
Wait for `RAG Pipeline API ready` in the console.

### Start the worker
In a separate terminal — this is what actually processes uploads and URLs; the API only queues them:
```powershell
python -m src.jobs.worker
```
Or with Procrastinate's own CLI, which supports more options (`--concurrency`, `--queues`, ...):
```powershell
procrastinate --app=src.jobs.worker.app worker --queues=ingest
```

### Start the Streamlit UI
In a separate terminal:
```powershell
streamlit run scripts/chat_ui.py
```

## 5. Usage Guide
1. Open the UI at `http://localhost:8080` (or `http://localhost:8501` if running locally without Docker).
2. Go to the **Ingest Document** tab.
3. Paste a URL or select a local PDF/Markdown file and click **Ingest**.
4. Go to the **Chat** tab and ask a question about the document you just uploaded. The system will retrieve the relevant sections and stream an answer with citations.
