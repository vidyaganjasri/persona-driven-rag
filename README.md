

# Persona-Driven Document Intelligence

**A Retrieval-Augmented Generation (RAG) system utilizing LLaMA and FAISS for role-aware Q\&A from PDFs.**

## Overview & Core Technology

This project builds a sophisticated **Retrieval-Augmented Generation (RAG) pipeline** that provides context-specific answers from your documents, delivered with the tone and expertise of a defined **persona**.

By running entirely locally with open-source models (LLaMA via `llama-cpp-python`), it offers a powerful, private, and cost-effective solution for querying document knowledge bases.

| Component | Role | Why It's Used |
| :--- | :--- | :--- |
| **Python** | Orchestration | Manages the entire pipeline flow. |
| **PyMuPDF** | Document Processing | Reads and extracts clean text from PDFs (the `fitz` library in the code). |
| **SentenceTransformers** | Embedding | Converts text chunks into high-density vector representations. |
| **FAISS** | Retrieval Index | Provides **fast, scalable similarity search** to find the most relevant context chunks. |
| **LLaMA** (`llama-cpp-python`) | Generation | Generates the final, context-aware, **persona-specific** answer. |

-----

##  Features

  * **Persona Customization:** Injects a specified role (e.g., "Astronomy Professor") into the prompt to tailor the answer's tone and depth.
  * **Context-Grounded Generation:** Retrieves only the most relevant document chunks to prevent hallucination (RAG).
  * **Local Execution:** Runs entirely on your machine using a quantized LLaMA model (`.gguf`), ensuring data privacy.
  * **Source Citation:** Returns the answer along with the source document file and page numbers.
  * **Modular Design:** Easy to adjust parameters like chunk size and retrieval depth (`k`).

### Demo Output (Successful Execution)

The following output was generated when asking about a scientific discovery as an "Astronomy Professor":

| Item | Value |
| :--- | :--- |
| **Query** | `"Henrietta Leavitt's key discovery"` |
| **Persona** | `"Astronomy Professor"` |
| **Answer** | Henrietta Leavitt's key discovery was that the length of time between a star's brightest and dimmest points, known as the "cycling time," is related to the star's overall brightness. This discovery allowed astronomers to determine the distances to far-off stars and hence, the size of our own galaxy. Leavitt's observation was a true surprise, as it came after years of carefully comparing thousands of photos of stars, looking for patterns in the darkness. |
| **Sources Used** | `what_is_science.pdf - Page 3`, `what_is_science.pdf - Page 2` |

-----

##  Installation & Setup

### 1\. Repository Setup

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/<your-username>/persona-driven-rag.git
    cd persona-driven-rag
    ```

2.  **Create and activate a virtual environment (Recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate     # Windows
    ```

### 2\. Install Dependencies

Install all necessary libraries using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 3\. Project Structure

Ensure your file structure matches this layout before running the script:

```
persona-driven-rag/
├── inputDocs/
│   └── your_document.pdf  # Add all your source PDFs here
├── models/
│   └── llama-2-7b-chat.Q4_K_M.gguf # Place your LLaMA model here
├── persona_rag.py         # The main script (or main.py if renamed)
└── requirements.txt
```

-----

##  Usage

Run the script from your terminal with the three required command-line arguments:

### General Syntax

```bash
python persona_rag.py "<Persona>" "<Question>" "<Model_Path>"
```

### Example Execution

```bash
python persona_rag.py "Astronomy Professor" "Henrietta Leavitt's key discovery" "models/llama-2-7b-chat.Q4_K_M.gguf"
```

-----

## How It Works (The RAG Flow)

The pipeline executes in a clear, five-step Retrieval-Augmented Generation process:

1.  **Ingestion:** The **PDF Documents** are split into **Text Chunks** (using `PyMuPDF`).
2.  **Embedding:** The chunks are converted into **Vector Embeddings** (using `SentenceTransformers`).
3.  **Indexing:** The vectors are stored in a **FAISS Index** for fast search.
4.  **Retrieval:** The user's **Query & Persona** is also embedded, and the **FAISS Index** quickly returns the **Top K Chunks** (the most relevant context).
5.  **Generation:** The **LLaMA Model** is prompted using the question, the persona, and **ONLY** the retrieved chunks to synthesize the final **Generated Answer**.

-----

##  Requirements.txt

For your reference, here are the contents of the `requirements.txt` file used for installation:

```text
# Requirements for Persona-Driven Document Intelligence RAG Pipeline
PyMuPDF
sentence-transformers
faiss-cpu
llama-cpp-python
numpy
```

-----

