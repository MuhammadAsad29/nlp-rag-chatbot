# 🤖 NLP & RAG Intelligent Assistant

An end-to-end, production-grade **Retrieval-Augmented Generation (RAG)** chatbot web application built with **Google Gemini API**, **Sentence Transformers**, **FAISS Vector Database**, and **Gradio**.

This project provides grounded, accurate answers on Natural Language Processing (NLP) concepts and RAG architecture by retrieving semantic context chunks before generating responses.

---

## 🌟 Key Features

- **Semantic Vector Search**: Uses `sentence-transformers/all-MiniLM-L6-v2` and `FAISS` Cosine Similarity indexing to retrieve relevant context chunks.
- **Strict Factual Grounding**: Powered by Google Gemini (`gemini-3.5-flash-lite` / `gemini-3.1-flash-lite`) with automated rate-limit fallback handling to prevent hallucinations.
- **Interactive Gradio Web GUI**:
  - Live chat workspace with copy buttons and example queries.
  - **Real-Time RAG Context Inspector**: Visualizes exact vector context chunks pulled from FAISS for every query.
  - **Dynamic Hyperparameter Sliders**: Adjust Top-K context retrieval (1–10) and LLM Temperature (0.0–1.0) on the fly.
- **Modern Google GenAI SDK**: Built using the latest unified `google-genai` SDK.

---

## 📁 Repository Structure

```text
├── app.py                 # Main entry point (Gradio Web GUI)
├── requirements.txt       # Dependencies
├── data/
│   ├── Diverse_NLP_QA.txt # Grounded knowledge dataset
│   └── topics.txt         # NLP & RAG topic list
└── rag/
    ├── chatbot.py         # RAG synthesis engine & Gemini API logic
    ├── embedder.py        # SentenceTransformer embedding logic
    ├── retriever.py        # FAISS Cosine Similarity search index
    └── QA_generator.py    # Automated dataset generation pipeline
```

---

## 🚀 Quickstart Guide

### 1. Clone the Repository
```bash
git clone https://github.com/MuhammadAsad29/nlp-rag-chatbot.git
cd nlp-rag-chatbot
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Set Up API Key
Create a `.env` file in the project root directory:
```env
GEMINI_API_KEY=your_google_gemini_api_key_here
```

### 4. Run the Web Application
```bash
python app.py
```
Open your browser and navigate to **`http://127.0.0.1:7860`**.

---

## 🛠️ Tech Stack

- **Frontend / GUI**: Gradio 6.0+
- **LLM Engine**: Google Gemini API (`google-genai` SDK)
- **Vector Search Index**: FAISS (`faiss-cpu`)
- **Embedding Model**: SentenceTransformers (`all-MiniLM-L6-v2`)
- **Language**: Python 3.11+
