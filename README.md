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
- **Job queue**: [Procrastinate](https://procrastinate.readthedocs.io/) (Postgres-backed). The API validates a request and queues it; a separate **worker** process does the actual fetching, parsing, chunking and embedding, so it ships as its own image (`Dockerfile.worker`) with the document-parsing dependencies the API doesn't need
- **APIs**: Groq / Gemini (Text Generation), Jina AI (Embeddings & Reranking)

**2. Environment Setup**
Create a `.env` file in the root directory and populate it with your API keys:
```env
# Required API Keys
GEMINI_API_KEY="your_gemini_key"
GROQ_API_KEY="your_groq_key"
JINA_API_KEY="your_jina_key"

# Postgres (Neon) — the DIRECT endpoint, not the "-pooler" one
DATABASE_URL="postgresql://user:password@ep-xxxx.region.aws.neon.tech/neondb?sslmode=require"

# Authorises POST /admin/keys and /admin/keys/revoke
ADMIN_API_KEY="your-admin-key"

# Optional Settings
WORKER_CONCURRENCY=2
ENABLE_QUERY_GENERALISATION=false
ENABLE_RERANKER=false
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

**4. Run the Worker**
In a second terminal, with the same `.env` — the API only queues an ingestion job; this is what actually runs it:
```bash
python -m src.jobs.worker
```
Without a running worker, an uploaded document's job stays at `status: "queued"` forever.

**5. Run the Frontend (Streamlit)**
In a new terminal window, activate the virtual environment and run:
```bash
streamlit run scripts/chat_ui.py
```
*The UI will automatically open in your default browser.*

## Key Features

### Document Processing
- **Reads Multiple Formats:** Easily processes website URLs, PDFs, DOCX, Markdown, and TXT files.
- **Image Reading:** Built-in OCR (Optical Character Recognition) can extract and read text directly from images and scanned documents.
- **Web Crawling:** You can drop in a URL or an XML Sitemap, and the system will automatically crawl and read the website for you.

### Text Splitting
- **Noise Removal:** Automatically detects and removes useless website menus, footers, and legal boilerplate so the AI focuses only on the real content.
- **Smart Chunking:** Instead of blindly chopping text every 500 words, it respects your document's natural structure (headings, paragraphs, code blocks, and tables) so no context is ever lost.

### Search Engine
- **Hybrid Search:** Combines meaning-based search (Dense Vectors) with exact keyword matching (Sparse Text) so it never misses a relevant detail.
- **Reranking:** Re-evaluates search results on the fly to ensure the most useful information is placed at the very top. Exposed as a runtime toggle — empirical ablation on our benchmark showed the off-the-shelf reranker reduces Recall@1 (0.974 → 0.816), so it is recommended only for latency-tolerant, non-interactive workloads where deeper cross-attention is more valuable than pinpoint top-1 precision.
- **Private Workspaces:** Every search is scoped to the workspace of your API key, and a request with no workspace returns nothing without touching the database. Keys are stored only as hashes and can be revoked individually.

### Chat & Memory
- **Follow-up Questions:** Remembers the context of your conversation so you can ask natural follow-up questions without repeating yourself.
- **No Hallucinations:** The AI is strictly programmed to answer *only* using the documents you provided. If the answer isn't in the text, it will tell you.
- **Clear Citations:** Every answer includes exact citations so you can verify where the AI found the information.
- **Real-Time Streaming:** Responses stream onto your screen instantly, just like ChatGPT.

### Testing & Monitoring
- **Performance Tracking:** Built-in logs track exactly how long each step (searching, ranking, generating) takes.
- **Automated Grading:** The system can automatically grade itself on how well it retrieved information and how accurate its final answers are.

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
    Ingest->>Queue: Defer ingestion job
    Ingest-->>User: job_id, status: queued
    Worker->>Queue: Claim the job
    Worker-->>Queue: Fetch, parse, chunk, embed, commit
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

This flowchart visualizes how your files are processed and saved. The API (left) only validates and queues; the worker (right) — a separate process, a separate Docker image — does everything else.

```mermaid
graph TD
    subgraph API["API — validates & queues, no document parsers installed"]
        A1[PDF / DOCX Upload] --> V{Validate: type, size, quota}
        A2[URL] --> V
        V --> Q[Register job + defer to queue]
    end

    Q --> QT[(Postgres job queue<br/>Procrastinate)]
    QT --> W

    subgraph Worker["Worker — claims jobs, does the parsing"]
        W(Format Routing) -->|Local File| C1[Extract Text]
        W -->|Web Link| C2[Read Website]

        C1 --> D[Convert to Markdown]
        C2 --> D

        D --> E[Clean up Noise & Menus]
        E --> F[Split text smartly by Headings]

        F --> G[Generate Searchable Vectors]
    end

    G --> H[(Postgres chunks table<br/>vector + keyword index + text<br/>one row, one transaction)]

    style H fill:#bbf,stroke:#333,stroke-width:2px
    style QT fill:#bbf,stroke:#333,stroke-width:2px
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
    
    F --> H[Combine Results]
    G --> H
    
    %% Refinement
    H --> I{Reranking Enabled?}
    I -->|Yes| J[Re-sort to find the best match]
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

The worker follows the same rule on its own startup (config, then schema), independently of the API.

## Setup & Hosting Notes

**Storage:**
Everything durable lives in one Postgres database (Neon free tier): document text and vectors, the keyword index, jobs, the job queue itself, API keys and cost history. Deleting a document removes its chunks, vectors and keyword entries in the same transaction. Neon suspends an idle database after about five minutes, so the first request after a pause can take a few seconds longer.

**Ingestion needs a running worker:**
`POST /ingest` only validates and queues; a separate worker process (`Dockerfile.worker`, or `python -m src.jobs.worker` locally) does the actual fetching, parsing, chunking and embedding. A deployment that only runs the API image will accept uploads that never progress past `status: "queued"`.

**Moving from the old Qdrant + SQLite storage:**
`python -m scripts.migrate_legacy --dry-run` reads both old stores and prints what it would copy. Without `--dry-run` it writes everything in one transaction and then verifies it. Existing API keys keep working.

**Workspace Access:**
There is no open sign-up. An administrator issues your workspace key with `POST /admin/keys` and can revoke it with `POST /admin/keys/revoke` (both send the `RAG-API-KEY` header). Paste the key into the **"API Key"** box in the sidebar of the chat interface to unlock your workspace and the documents you previously uploaded. Chat history lives only in the browser session and is not restored after a refresh.
