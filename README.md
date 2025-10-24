
-----

#  Offline RAG Assistant (Timed Version)

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

  <img width="1919" height="747" alt="image" src="https://github.com/user-attachments/assets/decff57a-06f8-46b6-a559-260f3f2c9a58" />
