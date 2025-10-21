#!/usr/bin/env python3
"""
persona_rag.py

RAG pipeline using PDFs, FAISS, SentenceTransformers, and LLaMA.
"""

import os
import sys
import re
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

# -------------------------
# 1. Extract PDF chunks
# -------------------------
def clean_text(text):
    """Remove unusual characters, multiple spaces, non-ASCII symbols."""
    text = re.sub(r'\s+', ' ', text)       # collapse whitespace
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # remove non-ASCII
    return text.strip()

def extract_pdf_chunks(folder='inputDocs', chunk_size=300):
    """Extract text from PDFs and split into chunks of ≤ chunk_size words."""
    chunks = []
    if not os.path.isdir(folder):
        print(f"Folder '{folder}' not found. Create it and add PDFs.")
        return chunks

    for filename in os.listdir(folder):
        if filename.lower().endswith('.pdf'):
            path = os.path.join(folder, filename)
            doc = fitz.open(path)
            for page_num, page in enumerate(doc, start=1):
                text = clean_text(page.get_text())
                if not text:
                    continue
                # Try to get a page title (first line)
                lines = page.get_text("text").splitlines()
                title = lines[0].strip() if lines else "No Title"
                title = re.sub(r'\s+', ' ', title)  # clean extra spaces
                title = title[:60]  # truncate if very long

                words = text.split()
                for i in range(0, len(words), chunk_size):
                    chunk_text = " ".join(words[i:i+chunk_size])
                    chunks.append({
                        'document': filename,
                        'page_number': page_num,
                        'text': chunk_text
                    })

    return chunks

# -------------------------
# 2. Build FAISS index
# -------------------------
def build_faiss_index(chunks, embed_model):
    """Create FAISS index from chunk embeddings."""
    if not chunks:
        return None, None, embed_model

    texts = [c['text'] for c in chunks]
    embeddings = embed_model.encode(texts, convert_to_numpy=True)
    embeddings = np.atleast_2d(embeddings)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings.astype('float32'))

    print(f"FAISS index built for {len(chunks)} chunks.")
    return index, embeddings, embed_model

# -------------------------
# 3. Retrieve top K chunks
# -------------------------
def retrieve_top_k(query, chunks, index, embed_model, k=2):
    """Retrieve top K relevant chunks for the query."""
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    q_emb = q_emb / np.linalg.norm(q_emb)
    D, I = index.search(q_emb.astype('float32'), k)
    results = []
    for idx in I[0]:
        if idx < len(chunks):
            results.append(chunks[idx])
    return results

# -------------------------
# 4. Generate answer using LLaMA
# -------------------------
def generate_answer(query, top_chunks, model_path):
    """Call LLaMA with top chunks to generate answer."""
    context = "\n---\n".join([
        f"Document: {c['document']}, Page {c['page_number']}\n{c['text']}"
        for c in top_chunks
    ])
    prompt = f"""
You are an expert. Answer the question based ONLY on the context below.

[CONTEXT]
{context}

[QUESTION]
{query}
"""
    llm = Llama(model_path=model_path, n_ctx=4096, n_gpu_layers=-1)
    completion = llm.create_completion(prompt=prompt, max_tokens=512)
    return completion['choices'][0]['text'].strip()

# -------------------------
# 5. Human-readable display
# -------------------------
def display_chunks(chunks):
    """Print chunks nicely for verification."""
    for idx, chunk in enumerate(chunks, start=1):
        print(f"\nCHUNK {idx}")
        print(f"Document: {chunk['document']}")
        print(f"Page: {chunk['page_number']}")
        print("Text:")
        print(chunk['text'])
        print("-" * 50)

# -------------------------
# 6. Main execution
# -------------------------
def main():
    if len(sys.argv) < 4:
        sys.exit("Usage: python persona_rag.py <persona> <job> <model_path>")

    persona = sys.argv[1]
    job = " ".join(sys.argv[2:-1])
    model_path = sys.argv[-1]

    print(f"Persona: {persona}, Job: {job}")

    # Extract and prepare chunks
    chunks = extract_pdf_chunks()
    if not chunks:
        print("No PDF chunks found. Exiting.")
        return

    # Build embeddings and FAISS index
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    index, embeddings, embed_model = build_faiss_index(chunks, embed_model)

    # Prepare query
    query = f"As a {persona}, summarize: {job}"

    # Retrieve top 1–2 chunks
    top_chunks = retrieve_top_k(query, chunks, index, embed_model, k=2)

    # Display retrieved chunks in human-readable form
    print("\n[TOP CHUNKS]")
    display_chunks(top_chunks)

    # Generate final answer
    answer = generate_answer(query, top_chunks, model_path)
    answer = answer.lstrip("] \n")  
    print("\n[FINAL ANSWER]:")
    print(answer)

    # Show source list
    sources = "\n".join([f"{c['document']} - Page {c['page_number']}" for c in top_chunks])
    print("\n[SOURCES USED]:")
    print(sources)

if __name__ == '__main__':
    main()
