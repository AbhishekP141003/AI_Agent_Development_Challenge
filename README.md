# Company Knowledge Base AI Agent

## Overview
This AI Agent allows employees to upload company documents (PDF/TXT) and query information using natural language.
It uses GPT-based retrieval augmented generation (RAG) to provide accurate answers.

## Features
- Upload multiple documents
- Automatic text splitting and vector storage
- Embedding-based search on queries
- GPT-powered answers using retrieved chunks
- Clean Streamlit UI

## Tech Stack
- Streamlit (UI)
- LangChain (RAG pipeline)
- ChromaDB (Vector DB)
- OpenAI GPT (LLM)
- Python

## How It Works (Architecture)
1. User uploads documents  
2. Documents are split into chunks  
3. Chunks stored in vector DB (Chroma)  
4. User asks question  
5. Most relevant chunks retrieved  
6. GPT generates final answer  

