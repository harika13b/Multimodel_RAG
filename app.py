import streamlit as st
import os
import fitz
import numpy as np
import faiss
import re
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from groq import Groq
import torch


os.environ["GROQ_API_KEY"] = "YOUR_GROQ_API_KEY"
client = Groq()

#REMOVE EXTRA SPACES AND NEWLINES(clean text)
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Return taxt in chunks
def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i+chunk_size])
    return chunks

# Extract text from PDF and return as list of dicts with text, source, and type
def extract_text_from_pdf(file):
    file.seek(0)
    file_bytes = file.read()

    if not file_bytes:
        return []

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    chunks = []

    for page_num, page in enumerate(doc):
        text = clean_text(page.get_text())
        split_texts = chunk_text(text)

        for chunk in split_texts:
            chunks.append({
                "text": chunk,
                "source": f"{file.name} - Page {page_num + 1}",
                "type": "text"
            })

    return chunks

# Load CLIP model and processor once and cache them for reuse
@st.cache_resource
def load_clip():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

# Get image embedding using CLIP
def get_image_embedding(image_file):
    model, processor = load_clip()
    image = Image.open(image_file).convert("RGB")

    inputs = processor(images=image, return_tensors="pt")


    with torch.no_grad():
        features = model.get_image_features(**inputs)

    return features[0].numpy()


def generate_answer(query, context):
    prompt = f"""
Answer the question based ONLY on the context below.

Use citations like [1], [2].

Context:
{context}

Question:
{query}
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


st.title(" Multimodal RAG with Citations")

uploaded_files = st.file_uploader(
    "Upload PDF or Images",
    type=["pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=True
)

if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.data = []


if uploaded_files:
    st.write("Processing files...")

    all_data = []
    text_model = SentenceTransformer('all-MiniLM-L6-v2')

    embeddings = []

    for file in uploaded_files:
        file.seek(0)

        if file.size == 0:
            st.warning(f"{file.name} is empty. Skipping...")
            continue

        #  PDF
        if file.type == "application/pdf":
            chunks = extract_text_from_pdf(file)

            for chunk in chunks:
                emb = text_model.encode(chunk["text"])
                embeddings.append(emb)
                all_data.append(chunk)

        #  IMAGE
        elif file.type.startswith("image"):
            emb = get_image_embedding(file)

            all_data.append({
                "text": "Image content (semantic match via CLIP)",
                "source": file.name,
                "type": "image"
            })

            embeddings.append(emb)

    #  FAISS INDEX
    if embeddings:
        dimension = len(embeddings[0])
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings).astype("float32"))

        st.session_state.index = index
        st.session_state.data = all_data
        st.session_state.text_model = text_model

        st.success("Files processed! Ask your question below.")


query = st.text_input("Ask a question:")

if query and st.session_state.index:

    query_emb = st.session_state.text_model.encode([query]) # convert query to vector
    distances, indices = st.session_state.index.search(query_emb, 3) # search for top 3 matches

    retrieved = [st.session_state.data[i] for i in indices[0]] 

    context = "\n".join([
        f"[{i+1}] ({item['source']}) {item['text']}"
        for i, item in enumerate(retrieved)
    ])

    answer = generate_answer(query, context)

    
    st.subheader(" Answer")
    st.write(answer)

    st.subheader(" Sources")

    for i, item in enumerate(retrieved):
        preview = item["text"][:300]

        st.markdown(f"""
### [{i+1}] {item['source']}

> {preview}...

---
""")
