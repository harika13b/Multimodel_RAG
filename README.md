# Multimodal RAG with Citations

This is a simple Multimodal RAG application built using Streamlit, FAISS, CLIP, Sentence Transformers, and Groq.

The app allows users to:

- Upload PDFs and Images
- Ask questions based on uploaded files
- Retrieve relevant content using semantic search
- Generate AI answers with citations

---

# Features

- PDF text extraction
- Image embedding using CLIP
- Semantic search with FAISS
- AI-generated answers using Groq
- Source citations
- Streamlit user interface

---

# Tech Stack

- Python
- Streamlit
- FAISS
- Sentence Transformers
- CLIP
- Groq API
- PyMuPDF
- Torch

---

# Installation

## Clone the repository

```bash
git clone https://github.com/your-username/multimodal-rag.git

cd multimodal-rag
```

---

## Install dependencies

```bash
pip install -r requirements.txt
```

---

## Add Groq API Key

Replace:

```python
os.environ["GROQ_API_KEY"] = "YOUR_GROQ_API_KEY"
```

with your actual API key.

---

# Run the Project

```bash
streamlit run app.py
```

---

# How It Works

1. Upload PDFs or Images
2. Text and image embeddings are created
3. FAISS stores embeddings
4. User asks a question
5. Relevant chunks are retrieved
6. Groq generates the final answer with citations

---

# Requirements

```text
streamlit
pymupdf
numpy
faiss-cpu
sentence-transformers
transformers
pillow
groq
torch
```

---

# Future Improvements

- OCR support
- Better citations
- Chat history
- ChromaDB integration
- Audio/video support

---

# Author

Harika  
CSE (AI & ML) Student
Aspiring AI Engineer
