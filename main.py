
import os
import sys
import re
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

class RAG_Pipeline:
    def __init__(self, embed_model_name="all-MiniLM-L6-v2", llm_model_path=None, context_folder='inputDocs', chunk_size=300, n_gpu_layers=-1):
        """Initializes the RAG Pipeline with models and configuration."""
        self.context_folder = context_folder
        self.chunk_size = chunk_size
        self.llm_model_path = llm_model_path
        self.n_gpu_layers = n_gpu_layers
        
        # State variables to be populated
        self.chunks = []
        self.faiss_index = None
        
        # Load embedding model
        print(f"Loading SentenceTransformer: {embed_model_name}...")
        self.embed_model = SentenceTransformer(embed_model_name)
        
        # Load LLama model (can be done lazily, but here it's done upon request)
        if self.llm_model_path:
             print(f"Loading LLaMA model from: {self.llm_model_path}...")
             self.llm = Llama(model_path=self.llm_model_path, n_ctx=4096, n_gpu_layers=self.n_gpu_layers, verbose=False)
        else:
             self.llm = None
             print("LLaMA model path not provided. LLM generation will be skipped.")

    # -------------------------
    # 1. Extract PDF chunks
    # -------------------------
    def _clean_text(self, text):
        """Remove unusual characters, multiple spaces, non-ASCII symbols."""
        text = re.sub(r'\s+', ' ', text)       # collapse whitespace
        text = re.sub(r'[^\x00-\x7F]+', '', text)  # remove non-ASCII
        return text.strip()

    def extract_and_chunk(self):
        """Extract text from PDFs and split into chunks."""
        self.chunks = []
        if not os.path.isdir(self.context_folder):
            print(f"Folder '{self.context_folder}' not found. Create it and add PDFs.")
            return

        for filename in os.listdir(self.context_folder):
            if filename.lower().endswith('.pdf'):
                path = os.path.join(self.context_folder, filename)
                try:
                    doc = fitz.open(path)
                    print(f"Processing {filename}...")
                    for page_num, page in enumerate(doc, start=1):
                        text = self._clean_text(page.get_text())
                        if not text:
                            continue

                        words = text.split()
                        for i in range(0, len(words), self.chunk_size):
                            chunk_text = " ".join(words[i:i+self.chunk_size])
                            self.chunks.append({
                                'document': filename,
                                'page_number': page_num,
                                'text': chunk_text
                            })
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    continue
        
        print(f"Extracted {len(self.chunks)} chunks from PDFs.")
        if not self.chunks:
             print("No chunks to process.")
        
    # -------------------------
    # 2. Build FAISS index
    # -------------------------
    def build_index(self):
        """Create FAISS index from chunk embeddings."""
        if not self.chunks:
            print("Cannot build index: No chunks available.")
            return

        texts = [c['text'] for c in self.chunks]
        
        # Generate embeddings
        embeddings = self.embed_model.encode(texts, convert_to_numpy=True)
        embeddings = np.atleast_2d(embeddings)
        
        # Normalize embeddings for Inner Product (IP) similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        d = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(d)
        self.faiss_index.add(embeddings.astype('float32'))
        
        # Store embeddings (optional, but useful for debugging)
        self.embeddings = embeddings

        print(f"FAISS index built with dimension {d}.")

    # -------------------------
    # 3. Retrieve top K chunks
    # -------------------------
    def retrieve_top_k(self, query, k=2):
        """Retrieve top K relevant chunks for the query."""
        if self.faiss_index is None:
            print("FAISS index not built. Cannot retrieve chunks.")
            return []

        q_emb = self.embed_model.encode([query], convert_to_numpy=True)
        q_emb = q_emb / np.linalg.norm(q_emb) # Normalize query embedding
        
        D, I = self.faiss_index.search(q_emb.astype('float32'), k)
        
        results = []
        for idx in I[0]:
            if idx < len(self.chunks):
                results.append(self.chunks[idx])
        return results

    # -------------------------
    # 4. Generate answer using LLaMA
    # -------------------------
    def generate_answer(self, query, top_chunks):
        """Call LLaMA with top chunks to generate answer."""
        if self.llm is None:
            return "LLaMA model not loaded. Cannot generate answer."
            
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
        # Call LLama to generate completion
        completion = self.llm.create_completion(prompt=prompt, max_tokens=512)
        answer = completion['choices'][0]['text'].strip()
        return answer.lstrip("] \n")

    # -------------------------
    # 5. Main execution wrapper
    # -------------------------
    def run_rag_query(self, query, k=2):
        """Executes the full RAG pipeline for a given query."""
        if not self.chunks:
             self.extract_and_chunk()
             self.build_index()

        if self.faiss_index is None:
            return "RAG setup failed. Check logs for errors."

        # 1. Retrieve
        top_chunks = self.retrieve_top_k(query, k=k)

        # 2. Display retrieved chunks
        print("\n" + "="*20 + " [RETRIEVED CHUNKS] " + "="*20)
        self._display_chunks(top_chunks)

        # 3. Generate answer
        answer = self.generate_answer(query, top_chunks)

        # 4. Show results
        print("\n" + "="*20 + " [FINAL ANSWER] " + "="*20)
        print(answer)

        sources = "\n".join([f"{c['document']} - Page {c['page_number']}" for c in top_chunks])
        print("\n" + "="*20 + " [SOURCES USED] " + "="*20)
        print(sources)
        
        return answer, top_chunks

    # -------------------------
    # 6. Human-readable display (Helper)
    # -------------------------
    def _display_chunks(self, chunks):
        """Print chunks nicely for verification."""
        for idx, chunk in enumerate(chunks, start=1):
            print(f"\nCHUNK {idx}")
            print(f"Document: {chunk['document']}")
            print(f"Page: {chunk['page_number']}")
            print("Text:")
            print(chunk['text'][:200] + '...') # Truncate for cleaner output
            print("-" * 50)


# -------------------------
# Main execution for command line
# -------------------------
def main():
    if len(sys.argv) < 4:
        sys.exit("Usage: python rag_pipeline_class.py <persona> <job_summary> <model_path>")

    persona = sys.argv[1]
    # The 'job' summary/query can contain spaces, so join all arguments from index 2 up to the last one (model_path)
    job_summary = " ".join(sys.argv[2:-1])
    model_path = sys.argv[-1]
    
    query = f"As a {persona}, summarize: {job_summary}"

    print(f"Persona: {persona}, Job Summary: {job_summary}")
    print(f"Final Query: {query}")
    print("-" * 60)

    try:
        # Initialize the pipeline
        pipeline = RAG_Pipeline(
            llm_model_path=model_path,
            context_folder='inputDocs',
            chunk_size=300,
            n_gpu_layers=-1 # Use all GPU layers if available
        )
        
        # Run the full RAG process
        pipeline.run_rag_query(query, k=2)

    except Exception as e:
        print(f"\nAn error occurred during pipeline execution: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
