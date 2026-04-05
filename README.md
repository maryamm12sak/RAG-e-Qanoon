# 🇵🇰 RAG-e-Qanoon: Pakistan Legal AI ⚖️
Hugging Face link:https://huggingface.co/spaces/MuhammadHamza33/RAG-e-Qanoon

An end-to-end, AI-powered legal assistant designed to answer questions regarding Pakistani law in fluent Urdu. This project utilizes an advanced Retrieval-Augmented Generation (RAG) pipeline to ensure all answers are strictly grounded in official legal documents (PDFs), preventing hallucinations and providing reliable legal information.

## ✨ Features

* **Modern Urdu Interface:** A beautiful, responsive frontend built with Streamlit, fully customized with CSS to support Urdu Nastaliq fonts and right-to-left (RTL) text formatting.
* **Persistent Chat Memory:** Chat histories are seamlessly saved and retrieved using MongoDB Atlas, allowing users to revisit past conversations.
* **Advanced RAG Pipeline:** * **Hybrid Search:** Combines Dense Vector Search (Meaning) via Pinecone and Sparse Keyword Search (BM25).
  * **Reciprocal Rank Fusion (RRF):** Intelligently merges results from both search methods.
  * **Cross-Encoder Re-ranking:** Re-evaluates the top results for maximum contextual accuracy before sending them to the LLM.
* **LLM-as-a-Judge Evaluation:** Built-in automated evaluation pipeline that calculates **Faithfulness** (factual grounding) and **Relevancy** (question alignment) for rigorous Ablation Studies.

## 🏗️ Architecture & Tech Stack

* **Frontend:** Streamlit, Custom CSS/HTML
* **Database (Memory):** MongoDB Atlas (`pymongo`)
* **Vector Store:** Pinecone Serverless
* **LLM (Generation):** Hugging Face Inference API (`Qwen/Qwen2.5-7B-Instruct` / `CohereForAI/aya-23-8B`)
* **Embeddings & Re-ranking:** `BAAI/bge-m3` & `BAAI/bge-reranker-v2-m3` (Running Locally via `sentence-transformers`)
* **Document Processing:** `pypdf`, `langchain-text-splitters`

## 📁 Project Structure

```text
RAG-e-Qanoon/
│
├── Frontend/
│   ├── main.py                 # Streamlit UI & MongoDB connection logic
│   └── pages/                  
│       └── 1_📊_Evaluation.py  # Dashboard for LLM Judge Ablation Study results
│
├── rag_backend/
│   └── rag_pipeline.py         # The core RAG engine, Chunking, Pinecone, & LLM logic
│
├── scrapper/
│   └── data/
│       └── raw/                # Place your downloaded Urdu Legal PDFs here
│
├── .env                        # Environment variables (API Keys & URIs)
└── README.md
