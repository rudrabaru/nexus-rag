#Nexus RAG System - Quickstart

This is a production-grade, no-framework RAG (Retrieval-Augmented Generation) system built entirely from scratch with FastAPI, Qdrant Cloud, and Streamlit.

## 1. Prerequisites
- Python 3.12+ (or Docker)
- A Gemini API Key (or Groq API Key)

## 2. Environment Setup

Create a `.env` file in the root directory:
```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key_here
# Optional: GROQ_API_KEY=your_groq_api_key_here
```

## 3. Quick Run (Docker API)
You can run the API backend via Docker:
```bash
docker build -t nexus-rag-api .
docker run -p 8000:8000 nexus-rag-api
```
- API available at: [http://localhost:8000](http://localhost:8000)
- API docs at: [http://localhost:8000/docs](http://localhost:8000/docs)

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

### Start the Streamlit UI
In a separate terminal:
```powershell
streamlit run scripts/chat_ui.py
```

## 5. Usage Guide
1. Open the UI at `http://localhost:7860` (or `http://localhost:8501` if running locally without Docker).
2. Go to the **Ingest Document** tab.
3. Paste a URL or select a local PDF/Markdown file and click **Ingest**.
4. Go to the **Chat** tab and ask a question about the document you just uploaded. The system will retrieve the relevant sections and stream an answer with citations.
