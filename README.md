# KhojAI — Grounded PDF Intelligence & Collaboration Platform

KhojAI is a secure, collaborative, Retrieval-Augmented Generation (RAG) platform. It allows users to upload PDF documents, automatically generate structured layouts and summaries using GPU-backed parsing, engage in grounded Q&A with real-time text streaming, and share interactive workspaces with guest collaborators (who can comment and chat without needing an account).

---

## 🚀 Key Features

* **Gpu-Accelerated PDF Ingestion**: Offloads heavy document parsing tasks to serverless containers (using **Docling** on Modal with A10G GPUs) to extract structural components (headings, tables, and page numbers) efficiently.
* **Intelligent Page Header Cleanup**: Automatically filters out running page headers, footers, and page numbers, correctly re-parenting structural subtrees to keep the context clean for the LLM.
* **Semantic Vector & Keyword Hybrid Search**: Combines dense semantic vector search (`BAAI/bge-base-en-v1.5`) with sparse keyword matching (BM25 for headings) in **Qdrant**, fused using Reciprocal Rank Fusion (RRF) for highly precise context retrieval.
* **K-Means Summarization**: Clusters document chunk embeddings using K-Means to identify the most representative context segments, generating a grounded 2-3 line summary of the entire document without exceeding LLM context windows.
* **Grounded Q&A with Citations**: Answers questions using *only* the retrieved context, dynamically injecting inline citations (e.g., `[p. X]`) representing source page numbers.
* **Real-time SSE Streaming**: Employs Server-Sent Events (SSE) to stream answers token-by-token directly to the browser.
* **Guest Collaboration**: Users can generate a secure share link (`/shared/{token}`) allowing guests to access the document summary, chat with the AI, and leave comments with names attributed to their session (no signup required).

---

## 🛠️ Technology Stack

* **Backend**:
  * [FastAPI](https://fastapi.tiangolo.com/) (API layer & SSE stream controller)
  * [LangGraph](https://www.langchain.com/langgraph) (Orchestrating stateful ingestion and retrieval workflows)
  * [Qdrant](https://qdrant.tech/) (Hybrid dense/sparse vector database)
  * [Supabase](https://supabase.com/) (Postgres for metadata/checkpointers, Storage for PDF files)
  * [Modal](https://modal.com/) (Serverless GPU runner for layout analysis & embedding models)
  * [Mistral AI / ChatMistralAI](https://mistral.ai/) (LLM generation)
* **Frontend**:
  * Vanilla HTML5 & CSS3 (Custom responsive layout system)
  * Vanilla ES Modules (Modular client logic & packet-resilient SSE stream parser)

---

## 📐 System Architecture

### Ingestion Pipeline
Processes the PDF through serverless layout parsers, generates dense/sparse index mappings, clusters chunks for summarization, and stores files/metadata.

```mermaid
graph TD
    START([Start Ingestion]) --> IDGen[pdf_id_generation]
    IDGen --> Parse[document_parsing on Modal GPU]
    Parse --> Chunk[chunks_creation & header cleanup]
    Chunk --> Embed[embeddings_generation on Modal]
    Embed --> VecStore[vector_storing in Qdrant]
    VecStore --> KMeans[get_summarised_chunks via K-Means]
    KMeans --> PromFormatter[generate_summarize_prompt]
    PromFormatter --> LLMSummary[generating_summary_response via Mistral]
    LLMSummary --> DBStore[file_upload to Supabase]
    DBStore --> END([End Ingestion])
```

### Retrieval & RAG Pipeline
Resolves user queries using hybrid dense/sparse search, fuses ranks via RRF, compiles prompts, and feeds checkpointed chat histories to stream AI responses.

```mermaid
graph TD
    START([Start Retrieval]) --> EmbedGen[embeddings_generation on Modal]
    EmbedGen --> SearchQdrant[retrieving_chunks via Qdrant Hybrid Search]
    SearchQdrant --> FormatPrompt[prompt_formatting with citations]
    FormatPrompt --> StreamResponse[generating_response via Mistral]
    StreamResponse --> END([End Retrieval])
```

---

## 📁 Repository Structure

```text
├── backend/
│   ├── api/
│   │   └── routes/          # API route controllers (auth, pdf, query, comments)
│   ├── core/                # Core initializers (db, qdrant, security, checkpointer)
│   ├── models/              # Pydantic schemas (request / response validation)
│   ├── pipeline/            # LangGraph state graphs (ingest & retrieval workflows)
│   ├── services/            # Core business logic (chunking, db, storage, vector indexing)
│   ├── main.py              # FastAPI app definition & lifespan registration
│   ├── dependencies.py      # Dependency injection & JWT user resolution
│   ├── requirements.txt     # Python dependencies
│   └── test_api.py          # API Smoke/Integration tests
├── frontend/
│   ├── css/                 # Styling (styles.css)
│   ├── js/                  # Client logic (api.js, panel-ui.js, viewer.js, shared.js)
│   ├── index.html           # Login / Signup landing page
│   ├── dashboard.html       # Personal workspace library grid
│   ├── viewer.html          # PDF Viewer & interactive chat/comments panel for owners
│   └── shared.html          # Collaborative guest workspace
```

---

## ⚙️ Environment Configuration

Create a `.env` file in the `backend/` directory with the following variables:

```ini
# FastAPI Configuration
SECRET_KEY=your_super_secret_jwt_key
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=60

# Supabase Configurations
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_KEY=your-supabase-anon-key
SUPABASE_DB_URI=postgresql://postgres.your-project-id:password@aws-0-us-east-1.pooler.supabase.com:6543/postgres?sslmode=require
BUCKET_NAME=pdfs

# Qdrant Configurations
QDRANT_API_KEY=your_qdrant_cloud_api_key

# Model APIs
MISTRAL_API_KEY=your_mistral_api_key
GROQ_API_KEY=your_groq_api_key
```

---

## 🔑 Model & Deployment Credentials Setup

1. **Supabase & Postgres Setup**:
   * Create a Supabase project.
   * Run your DB migrations (for `users`, `pdfs`, `comments`, and `chat_sessions` tables).
   * Create a Storage Bucket named `pdfs`.
2. **Qdrant Setup**:
   * Create a Qdrant Cloud cluster or run a local instance.
   * Provide the `QDRANT_API_KEY` (and optionally configure the URL if changed in `qdrant.py`).
3. **LLM Keys**:
   * Provide `MISTRAL_API_KEY` for LLM generations.
4. **Modal Deployment (Critical)**:
   * The PDF parser (Docling) and embedding model run serverless via **Modal**.
   * To deploy them under a new account:
     1. Install the Modal CLI locally: `pip install modal`
     2. Authenticate the CLI: `modal setup`
     3. Deploy the serverless app defined in [`modals.py`](file:///C:/Users/hp/OneDrive/Desktop/SpotDraftAssignment/backend/core/modals.py) to their Modal space:
        ```bash
        modal deploy backend/core/modals.py
        ```
     4. Once deployed, the local app will automatically resolve and connect to the remote parsing app under their account.

---

## 🏃 Getting Started

### Prerequisites
* Python 3.11+

### 1. Setup Virtual Environment & Dependencies
```bash
# Navigate to backend
cd backend

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 2. Run the Server
```bash
python main.py
```
The application will run locally at `http://127.0.0.1:8000`.

---

## 🧪 Running API Smoke Tests

Ensure the local server is running, then run the integration test suite:

```bash
cd backend
python test_api.py
```
This runs synthetic auth registration, upload verification, token generation, document sharing, commenting, and SSE stream querying sequences.
