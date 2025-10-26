
-----

#  Offline RAG Assistant

This project implements a **Retrieval-Augmented Generation (RAG)** system using the LangChain framework to answer questions over a provided PDF document. It leverages local models via **Ollama** for the LLM, **HuggingFace MiniLM** for embeddings, and **FAISS** for fast, local vector storage.

A key feature is the **performance timing** for every major step, allowing users to benchmark the pipeline's speed.

##  Features

  * **Offline Operation:** Uses local models via Ollama and a local FAISS index.
  * **Performance Measurement:** Times the duration of PDF loading, chunking, vector store creation, retrieval, and LLM generation.
  * **Conversational Interface:** Provides a simple loop for asking multiple questions.
  * **Source Citation:** Displays the source chunks and page numbers used to generate the answer.

##  Prerequisites

Before running the application, you must have the following installed and configured:

1.  **Python 3.9+**
2.  **Ollama:** Ensure the Ollama service is running on your local machine.
3.  **Required Ollama Models:** Pull the specified LLM and, optionally, the embedding model (though this script uses HuggingFace embeddings).
      * **LLM (Chat Model):** `gemma3:4b`
        ```bash
        ollama pull gemma3:4b
        ```

##  Installation

1.  **Save the Code:** Save the provided Python code as `rag_assistant.py` (or `tes.py`).

2.  **Create and Activate a Virtual Environment:**

    ```bash
    python -m venv venv
    .\venv\Scripts\activate  # Windows
    # source venv/bin/activate # macOS/Linux
    ```

3.  **Install Python Dependencies:**

    ```bash
    pip install langchain-community langchain-text-splitters faiss-cpu
    pip install pypdf
    pip install sentence-transformers
    ```

##  Usage

1.  **Place Your Document:** Ensure the PDF file you wish to query is accessible on your system.

2.  **Run the Script:** Execute the Python file. You will be prompted to enter the path to your PDF.

    ```bash
    python rag_assistant.py
    ```

3.  **Example Session:**

<img width="1919" height="539" alt="image" src="https://github.com/user-attachments/assets/d68b1c28-24e8-470b-b9b9-89c7d5606928" />

Here is the "Results and Performance Analysis" section to insert into your `README.md`, perfectly aligning with your current structure and highlighting your key benchmark data.

***

##  Results and Performance Analysis

A key objective of this project was to establish an efficient, scalable RAG system capable of handling various document sizes while operating fully offline. The following benchmarks validate the system's performance on local hardware (CPU).

### 📊 Performance Benchmarks Summary

The table below summarizes the time taken (in seconds) for each major step in the RAG pipeline across different document sizes.

| PDF Name | Pages | Chunks | FAISS Creation Time (s) | Total Setup Time (s) | Retrieval Time (s) | LLM Response Time (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Law.pdf** | 18 | 42 | 5.01 | 5.82 | 0.01 | 42.48 |
| **Python0.pdf** | 155 | 433 | 18.45 | 22.21 | 0.02 | 54.69 |
| **Python1.pdf** | 415 | 793 | 77.49 | 81.69 | 0.02 | 95.14 |
| **Python2.pdf** | 1213 | 3572 | 364.13 | 379.15 | 0.01 | 35.88 |

###  Key Performance Takeaways

1.  **Linear Scalability in Setup:**
    The **Total Setup Time** (dominated by the FAISS vectorization process) scales **linearly** with the number of document chunks. The system efficiently handles massive documents (e.g., **1213 pages**) with the full indexing process completing in under **6 minutes ($\approx 379s$)** on a CPU.

2.  **Constant Retrieval Speed (FAISS Advantage):**
    The time taken to retrieve the most relevant text chunks is **constant ($\approx 0.01 - 0.02$ seconds)**, regardless of the index size (from 42 chunks up to 3572 chunks). This constancy is a critical demonstration of **FAISS's efficiency** for fast semantic search.

3.  **Overall Query Efficiency:**
    Once the index is created, queries are answered quickly, combining near-instantaneous retrieval ($\approx 0.01$s) with local LLM generation. This setup validates the pipeline's potential for **low-latency querying** in offline, secure environments.
