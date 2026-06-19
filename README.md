# auron - Local Code-RAG Web Assistant

This is a 100% pure Python web application that allows you to analyze any GitHub repository locally using an AST-based chunking pipeline, ChromaDB vector store, and Ollama local LLMs.

## Tech Stack
- Frontend/Backend: Streamlit
- Vector Database: ChromaDB
- Local LLMs: `qwen2.5-coder` (Generation), `nomic-embed-text` (Embeddings)

## Setup Instructions
1. Clone this repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run Ollama and pull models:
   - `ollama pull qwen2.5-coder`
   - `ollama pull nomic-embed-text`
4. Start the application: `streamlit run app.py`
