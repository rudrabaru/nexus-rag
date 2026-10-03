# Nexus RAG

A fast, accurate, and secure Retrieval-Augmented Generation (RAG) system. It reads your documents, understands the context, and answers questions reliably without making things up.

**Live Deployment:**
- 🖥️ **API Backend** (Render Free Tier): `https://nexus-rag-backend-hjxa.onrender.com/docs/`
- 💬 **Streamlit UI** (Streamlit Cloud): `https://nexus-rag-2026.streamlit.app`

## Overview
Nexus RAG takes your files (PDFs, URLs, text) and turns them into a searchable knowledge base. It's designed to be fast by processing data in memory, secure by keeping user workspaces completely separated, and smart enough to handle complex follow-up questions just like a real conversation.

## Local Quickstart

**1. Prerequisites & Tech Stack**
- **Language**: Python 3.10+
- **Frameworks**: FastAPI, Streamlit
- **Database**: Postgres (Neon) with pgvector: vectors (HNSW), keyword search (full-text), documents, jobs, keys and metrics in one store; schema managed by Alembic
- **Job queue**: [Procrastinate](https://procrastinate.readthedocs.io/) (Postgres-backed). The API validates a request and queues it; a **fetch worker** reads web pages through reader APIs, and a **parse worker** parses, chunks and embeds. The parse worker ships as its own image (`Dockerfile.worker`) with the dependencies the API doesn't need
- **Parsing**: [Docling](https://docling.org/) for PDF and DOCX (layout, tables, OCR of scanned pages)
- **Web pages**: hosted reader APIs only (keyless Jina Reader, optional Firecrawl). No process we host contacts the target site
- **APIs**: Gemini / Groq (generation), Voyage AI (embeddings; local Ollama optional)
- **Reranking**: [FlashRank](https://github.com/PrithivirajDamodaran/FlashRank), a small ONNX cross-encoder on CPU inside the API (default); Jina's or Voyage's hosted rerankers as options

**2. Environment Setup**
Create a `.env` file in the root directory and populate it with your API keys:
```env
# Required API Keys
GEMINI_API_KEY="your_gemini_key"
GROQ_API_KEY="your_groq_key"
VOYAGE_API_KEY="pa-..."   # embeddings (EMBEDDING_PROVIDER=voyage, the default)
JINA_API_KEY="your_jina_key"       # only for RERANKER=jina

# Postgres (Neon) — the DIRECT endpoint, not the "-pooler" one
DATABASE_URL="postgresql://user:password@ep-xxxx.region.aws.neon.tech/neondb?sslmode=require"

# Authorises POST /admin/keys and /admin/keys/revoke
ADMIN_API_KEY="your-admin-key"

# Optional Settings
WORKER_CONCURRENCY=2
ENABLE_QUERY_GENERALISATION=false
RETRIEVAL_STRATEGY=hybrid   # dense | sparse | hybrid
RERANKER=flashrank          # flashrank | jina | voyage | none (applied when a query asks for reranking)
```
`.env.example` lists every variable. The API and the worker each refuse to start and name each missing or invalid one.

**3. Run the Backend (FastAPI)**
```bash
git clone https://github.com/rudrabaru/nexus-rag.git
cd nexus-rag
python -m venv venv
# Windows: venv\Scripts\activate
# Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
alembic upgrade head          # create or update the schema (once per database, and after pulling migrations)
uvicorn src.api.main:app --reload
```
The API checks the schema revision at startup and refuses to run against a database that has not been migrated.
*The API will be available at `http://localhost:8000/docs`*

**4. Run the Workers**
In two more terminals, with the same `.env`. The API only queues ingestion jobs; these run them:
```bash
python -m src.jobs.workers ingest   # parse worker: uploads and fetched pages -> chunks and vectors
python -m src.jobs.workers fetch    # fetch worker: web pages and sitemaps, through reader APIs
# add --drain to either to run what is queued and exit (how workers run on a laptop)
```
Without the parse worker, every job stays at `status: "queued"`; without the fetch worker, URL jobs do. The first document the parse worker handles downloads Docling's models (~0.5 GB) once.

**5. Run the Frontend (Streamlit)**
In a new terminal window, activate the virtual environment and run:
```bash
streamlit run scripts/chat_ui.py
```
*The UI will automatically open in your default browser.*

## Key Features

### Document Processing
- **Reads Multiple Formats:** Processes web pages, sitemaps, PDFs, DOCX, Markdown and TXT files.
- **Keeps Document Structure:** PDFs and Word files are parsed with a layout model (Docling), so headings, reading order and tables survive, and scanned pages are read with OCR, all locally.
- **Polite Web Reading:** Give it a page or an XML sitemap (up to 50 pages). Pages are read through hosted reader APIs that respect `robots.txt`, paced per site, within a daily page quota, and every fetch is logged.

### Text Splitting
- **Noise Removal:** Automatically detects and removes useless website menus, footers, and legal boilerplate so the AI focuses only on the real content.
- **Smart Chunking:** Instead of blindly chopping text every 500 words, it respects your document's natural structure (headings, paragraphs, code blocks, and tables) so no context is ever lost.

### Search Engine
- **Hybrid Search:** Combines meaning-based search (Dense Vectors) with exact keyword matching (Sparse Text) so it never misses a relevant detail.
- **Configurable retrieval:** Every search knob (strategy, result count, fusion constant and weights, reranker, rerank pool size) is one `RetrievalConfig`. Chat runs the default configuration; the evaluator runs any configuration through the same code, so what is measured is what is served.
- **Reranking:** Re-sorts a pool of candidates with a cross-encoder that reads the question and each passage together. FlashRank runs locally for $0; Jina's and Voyage's hosted rerankers are options (Voyage's free tier allows 3 reranks a minute). It is a per-query toggle, because on the earlier prototype benchmark a reranker traded Recall@1 (0.974 → 0.816) for Recall@5 (0.974 → 1.000). If a reranker or the query embedding fails, the answer says so in its retrieval record instead of silently degrading.
- **Private Workspaces:** Every search is scoped to the workspace of your API key, and a request with no workspace returns nothing without touching the database. Keys are stored only as hashes and can be revoked individually.

### Chat & Memory
- **Follow-up Questions:** Remembers the context of your conversation so you can ask natural follow-up questions without repeating yourself.
- **No Hallucinations:** The AI is strictly programmed to answer *only* using the documents you provided. If the answer isn't in the text, it will tell you.
- **Clear Citations:** Every answer includes exact citations so you can verify where the AI found the information.
- **Real-Time Streaming:** Responses stream onto your screen instantly, just like ChatGPT.

### Testing & Monitoring
- **Performance Tracking:** Built-in logs track exactly how long each step (searching, ranking, generating) takes.
- **Evaluation with evidence:** `python -m src.evaluation run spec.json` runs several retrieval configurations over the same questions, stores every per-question result in Postgres, and reports whether each one is significantly better or worse than a baseline (or that the question set is too small to tell). Optionally it also generates answers and has a pinned judge model score their faithfulness, with answers and verdicts cached so identical work is never paid for twice.
- **Synthetic test sets:** `python -m src.testsets generate --tenant T` has an LLM write questions from your ingested chunks (easy, paraphrased and hard indirect ones), each with the exact chunk it came from as ground truth. You review them (`review`), freeze the accepted ones as a dataset (`finalize`), and an experiment can then score retrieval by exact chunk (`"relevance": "chunk"`) instead of by document. Reports label such sets *SYNTHETIC*, and `verify` tells you when re-chunking has made a set's ground truth stale.

## Pipeline Architecture

For deep-dive documentation on how each step works under the hood, check out the `docs/phases/` directory:

1. [Phase 1: Ingestion](docs/phases/phase1_ingestion.md) - Reading and extracting data.
2. [Phase 2: Processing](docs/phases/phase2_processing.md) - Cleaning out noise.
3. [Phase 3: Chunking](docs/phases/phase3_chunking.md) - Splitting text smartly.
4. [Phase 4: Embedding](docs/phases/phase4_embedding.md) - Converting text to searchable numbers.
5. [Phase 5: Retrieval](docs/phases/phase5_retrieval.md) - Finding the best answers.
6. [Phase 6: Evaluation](docs/phases/phase6_evaluation.md) - Grading the system's accuracy.
7. [Phase 7: Generation](docs/phases/phase7_generation.md) - Writing the final response.
8. [Phase 8: Conversational RAG](docs/phases/phase8_conversational_rag.md) - Handling chat memory.
9. [Phase 9: Production Tradeoffs](docs/phases/phase9_production_tradeoffs.md) - Why we built it this way.

## The User Workflow (How to use Nexus RAG)

This flow illustrates how your private workspace is kept secure.

```mermaid
sequenceDiagram
    actor Admin
    actor User
    participant Auth as Auth Store
    participant Ingest as Ingestion API
    participant Queue as Job Queue (Postgres)
    participant Worker
    participant Query as Query API

    %% Key provisioning (admin only; there is no open sign-up)
    Admin->>Auth: POST /admin/keys (admin key)
    Auth-->>Admin: Returns API Key & Workspace ID
    Admin-->>User: Shares the API Key
    Note right of User: Keep your API Key. No passwords<br/>are saved in the database.

    %% Ingestion: the API only validates and queues
    User->>Ingest: Upload a PDF or URL
    Ingest->>Auth: Verify API Key
    Auth-->>Ingest: Validated Workspace ID
    Ingest->>Queue: Defer job (URL: fetch queue, file: ingest queue)
    Ingest-->>User: job_id, status: queued
    Worker->>Queue: Claim the job
    Worker-->>Queue: (fetch worker) read pages via reader API, store them, defer ingest
    Worker-->>Queue: (parse worker) parse, chunk, embed, commit
    User->>Ingest: GET /ingest/{job_id}
    Ingest-->>User: status: complete
    Note right of User: Data is securely locked<br/>to your Workspace

    %% Querying
    User->>Query: Ask a question
    Query->>Auth: Verify API Key
    Auth-->>Query: Validated Workspace ID
    Query-->>User: Streams Answer
```

## The Ingestion Pipeline (Internal Flow)

This flowchart visualizes how your files are processed and saved. The API only validates and queues. The fetch worker (same slim image) talks only to reader APIs. The parse worker (its own heavy image) never contacts a website: it reads everything from Postgres.

```mermaid
graph TD
    subgraph API["API: validates & queues, nothing fetched or parsed"]
        A1[PDF / DOCX / TXT / MD upload] --> V1{Type, size, quota}
        A2[Page or sitemap URL] --> V2{https, public, domain lists,<br/>daily page quota}
    end

    V2 --> FQ[(fetch queue)]
    V1 --> IQ[(ingest queue)]

    subgraph Fetch["Fetch worker: slim, reader APIs only"]
        FQ --> R[Jina Reader, keyless<br/>robots.txt respected<br/>Firecrawl fallback]
        R --> FP[(fetched_pages + fetch_log)]
    end

    FP --> IQ

    subgraph Parse["Parse worker: heavy, compute only"]
        IQ --> P{Source}
        P -->|Upload| DL[Docling in a child process<br/>PyMuPDF text fallback]
        P -->|Fetched pages| MD[Markdown from Postgres]
        DL --> E[Clean up noise]
        MD --> E
        E --> F[Split by headings]
        F --> G[Embed into the active index<br/>Voyage, paced]
    end

    G --> H[(Postgres chunks table<br/>vector + keyword index + text<br/>one row per index, one transaction)]

    style H fill:#bbf,stroke:#333,stroke-width:2px
    style FQ fill:#bbf,stroke:#333,stroke-width:2px
    style IQ fill:#bbf,stroke:#333,stroke-width:2px
    style FP fill:#bbf,stroke:#333,stroke-width:2px
```

## The Query Pipeline (Internal Flow)

This flowchart visualizes how the system finds the perfect answer.

```mermaid
graph TD
    A[Your Question] --> B{Is it a follow-up?}
    
    %% Rewriting
    B -->|Yes| C[Rewrite question using chat history]
    B -->|No| D[Final Search Query]
    C --> D
    
    %% Hybrid Retrieval
    D --> E{Search concurrently}
    E -->|Meaning Search| F[(pgvector HNSW)]
    E -->|Keyword Search| G[(Postgres full-text)]
    
    F --> H[Combine ranks: weighted RRF]
    G --> H
    
    %% Refinement
    H --> I{Reranking requested?}
    I -->|Yes| J[Cross-encoder re-sorts the candidate pool]
    I -->|No| K[Top Results]
    J --> K
    
    %% Generation
    K --> L[Prepare prompt with citations]
    L --> M[AI generates the answer]
    
    M -->|Real-Time| N((Streams to your screen))
```

## Startup (Stateless Host, Durable Database)

The API keeps nothing on local disk, so a host that wipes its filesystem on restart (Render, Hugging Face Spaces) loses nothing and needs no recovery step.

```mermaid
graph LR
    A[Server start] --> B{Config complete?}
    B -->|No| X[Refuse to start<br/>name each missing setting]
    B -->|Yes| C{Schema at code revision?}
    C -->|No| Y[/health returns 503<br/>run: alembic upgrade head/]
    C -->|Yes| D[Serve traffic]

    style X fill:#f96,stroke:#333,stroke-width:2px
    style Y fill:#f96,stroke:#333,stroke-width:2px
```

Both workers follow the same rule on their own startup (config, then schema), and each refuses to start if it holds a task from another queue.

## Setup & Hosting Notes

**Storage:**
Everything durable lives in one Postgres database (Neon free tier): document text and vectors, the keyword index, jobs, the job queue itself, API keys and cost history. Deleting a document removes its chunks, vectors and keyword entries in the same transaction. Neon suspends an idle database after about five minutes, so the first request after a pause can take a few seconds longer.

**Ingestion needs running workers:**
`POST /ingest` only validates and queues. The parse worker (`Dockerfile.worker`) must run for anything to be indexed, and the fetch worker (the API image, run with `python -m src.jobs.workers fetch`) for URLs. A deployment that only runs the API accepts ingestions that never progress past `status: "queued"`.

**Embedding indexes:**
Each embedding model has its own index (`provider:model`), and the API searches and ingests into the index of `EMBEDDING_PROVIDER` (default `voyage:voyage-4`). Vectors of two models are never mixed: switching the provider points the API at a different index, which starts empty until documents are ingested with it.

**Workspace Access:**
There is no open sign-up. An administrator issues your workspace key with `POST /admin/keys` and can revoke it with `POST /admin/keys/revoke` (both send the `X-Admin-Key` header). Paste the key into the **"API Key"** box in the sidebar of the chat interface to unlock your workspace and the documents you previously uploaded. Chat history lives only in the browser session and is not restored after a refresh.
