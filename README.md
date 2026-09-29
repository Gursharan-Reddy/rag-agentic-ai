# Agentic AI RAG Chatbot

An enterprise-grade, stateful Retrieval-Augmented Generation (RAG) chatbot built with **Python, LangGraph, Pinecone, Groq LLM, HuggingFace embeddings, and Streamlit**. The system strictly grounds its responses in the *Agentic AI For Executives* eBook and refuses out-of-context questions.

---

## Project Architecture & Structure

```text
rag-agentic-ai/
│
├── data/
│   └── Ebook-Agentic-AI.pdf        # eBook source document
│
├── src/
│   ├── __init__.py
│   ├── ingestion.py                # PDF loading, splitting, and Pinecone vector indexing
│   ├── graph.py                    # LangGraph stateful workflow (Retrieve -> Generate)
│   └── config.py                   # Environment variable validation and configuration constants
│
├── app.py                          # FastAPI backend REST application
├── streamlit_app.py                # Interactive Streamlit frontend
├── tests_sample_queries.py         # Automated evaluation script for benchmark queries
├── requirements.txt                # Python project dependencies
├── .env.example                    # Template for required environment variables
└── README.md                       # Project documentation
```

---

## Technical Stack

| Layer | Technology |
|---|---|
| Orchestration & Workflow | `LangGraph` & `LangChain` |
| Vector Database | `Pinecone` (Serverless index, cosine similarity) |
| Embeddings | `HuggingFace Embeddings` (`all-MiniLM-L6-v2`, 384 dimensions) |
| LLM Inference | `Groq` (`openai/gpt-oss-20b`) |
| Backend API | `FastAPI` & `Uvicorn` |
| Frontend UI | `Streamlit` |
| Document Processing | `PyPDFLoader` & `RecursiveCharacterTextSplitter` |

---

## Setup and Installation

### 1. Clone the Repository and Create a Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the root directory using this template:

```env
GROQ_API_KEY=your_groq_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=agentic-ai-index
```

### 4. Place the Source Document

Make sure the PDF is in the `data/` directory:

```text
data/Ebook-Agentic-AI.pdf
```

---

## Document Ingestion Pipeline

Parse the PDF, split the text into optimized chunks, and vectorize/upsert them into your Pinecone index:

```bash
python -m src.ingestion
```

---

## Testing and Verification

Run the automated script to execute benchmark queries that verify document grounding and out-of-context refusal behavior:

```bash
python tests_sample_queries.py
```

---

## Running the Application

### Option A: Streamlit Web UI (Recommended)

Launch the chat interface with contextual chunk inspection:

```bash
streamlit run streamlit_app.py
```

Open in your browser: `http://localhost:8501`

### Option B: FastAPI Backend REST Server

Launch the API server:

```bash
uvicorn app:app --reload
```

- Swagger docs: `http://127.0.0.1:8000/docs`
- Chat endpoint: `POST http://127.0.0.1:8000/chat`

Example request body:

```json
{
  "query": "What is Agentic AI according to the eBook?"
}
```