# 🎓 University Academic Knowledge Assistant

A production-oriented **Retrieval-Augmented Generation (RAG)** application for answering university and academic questions from pre-indexed Excel documents.

The application uses:

* **Streamlit** — web application interface
* **FAISS** — vector similarity search
* **Sentence Transformers** — document and query embeddings
* **Groq** — LLM inference
* **GPT-OSS 120B** — `openai/gpt-oss-120b`
* **Metadata-based source traceability**
* **Google Colab** — document ingestion and vector database creation
* **GitHub + Streamlit Cloud** — application deployment

---

## 🚀 Project Architecture

```text
                 UNIVERSITY EXCEL DOCUMENTS
                           │
                           ▼
                    Google Drive
                           │
                           ▼
                    Google Colab
                           │
                           ▼
                 Excel File Processing
                           │
                           ▼
                  Cleaning & Chunking
                           │
                           ▼
             Sentence Transformer Embeddings
                           │
                           ▼
                 L2 Normalization
                           │
                           ▼
                  FAISS IndexFlatIP
                           │
             Cosine Similarity Search
                           │
                           ▼
              ┌─────────────────────────┐
              │    RAG Database         │
              │                         │
              │ academic_knowledge.faiss│
              │ metadata.json           │
              │ config.json             │
              └─────────────────────────┘
                           │
                           │ Upload only database
                           ▼
                        GitHub
                           │
                           ▼
                    Streamlit Cloud
                           │
                     Student Query
                           │
                           ▼
             Sentence Transformer Encoder
                           │
                           ▼
                    FAISS Top-K
                           │
                           ▼
              Relevant Knowledge Chunks
                           │
                           ▼
                  Metadata + Context
                           │
                           ▼
                Groq GPT-OSS 120B
                           │
                           ▼
                  Generated Answer
                           │
                           ▼
             Source Metadata / Traceability
```

---

# 📁 Repository Structure

```text
university-academic-rag/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── rag_database/
    ├── academic_knowledge.faiss
    ├── metadata.json
    ├── config.json
    └── README.md
```

---

# 🔐 Important: Original Excel Files Are NOT Uploaded

The original Excel documents are used only during the indexing stage.

The workflow is:

```text
Excel Documents
      │
      ▼
Google Colab
      │
      ▼
Extract data
      │
      ▼
Create chunks
      │
      ▼
Generate embeddings
      │
      ▼
Create FAISS index
      │
      ▼
Save metadata
      │
      ▼
Upload database to GitHub
```

The original files do **not** need to be uploaded to GitHub.

---

# 🧠 Why metadata.json is required

FAISS stores vectors, but it does not by itself store the original text associated with every vector.

Therefore, the application needs:

```text
academic_knowledge.faiss
        +
metadata.json
```

For example:

```json
{
  "records": [
    {
      "text": "Students must complete the registration process...",
      "metadata": {
        "file_name": "Academic_Policies.xlsx",
        "sheet_name": "Registration",
        "row_start": 10,
        "row_end": 17,
        "chunk_number": 2,
        "total_chunks_in_sheet": 8
      }
    }
  ]
}
```

This allows the application to retrieve the text and also tell the user where the information came from.

---

# 📊 Metadata / Source Traceability

The application displays metadata for every retrieved source.

Example:

```text
Source 1

File:
Academic_Policies.xlsx

Sheet:
Registration

Excel Rows:
10 – 17

Chunk:
2

Vector ID:
15

Similarity:
0.8234
```

This provides source traceability for the generated answer.

---

# 🔎 Similarity Search

The vector database uses:

```text
FAISS IndexFlatIP
```

with normalized embeddings.

The indexing process performs:

```python
faiss.normalize_L2(embeddings)
```

and the query embedding is also normalized.

Therefore:

```text
Inner Product = Cosine Similarity
```

The application can then use a similarity threshold to remove weakly related chunks.

Example:

```text
Top-K = 5
Minimum similarity = 0.20
```

Only retrieved chunks with a score greater than or equal to the threshold are passed to the LLM.

---

# 🤖 Language Model

The application uses Groq with:

```text
openai/gpt-oss-120b
```

GPT-OSS 120B is used to generate the final response from the retrieved university context.

The application also provides a reasoning-effort setting:

```text
Low
Medium
High
```

The model should only use the retrieved university information when answering academic-policy questions.

---

# 🔑 Groq API Key

The Groq API key should **never be hardcoded** into `app.py`.

The application expects:

```text
GROQ_API_KEY
```

---

# ☁️ Streamlit Cloud Secrets

After deploying the repository to Streamlit Cloud:

1. Open your Streamlit application.
2. Open the application settings.
3. Find **Secrets**.
4. Add:

```toml
GROQ_API_KEY = "your-groq-api-key"
```

5. Save the secret.
6. Restart/redeploy the application.

Do not put the API key inside:

```text
app.py
```

or:

```text
README.md
```

or:

```text
GitHub repository
```

---

# 🧩 Application Features

## 1. Academic Question Answering

Students can ask questions such as:

```text
What are the requirements for course registration?
```

or:

```text
What is the procedure for academic withdrawal?
```

or:

```text
How many courses can a student register for?
```

The system retrieves relevant information from the indexed university knowledge base.

---

## 2. Adjustable Retrieval

The sidebar allows the user to select:

```text
Number of sources
```

from 1 to 10.

This controls the number of FAISS chunks retrieved.

---

## 3. Similarity Threshold

The application provides:

```text
Minimum similarity
```

This prevents very weak matches from being sent to the LLM.

---

## 4. Technicality Level

The user can choose:

```text
Beginner
Intermediate
Advanced
```

This controls how technical the generated response should be.

---

## 5. Response Length

Available options:

```text
Short
Medium
Detailed
```

---

## 6. GPT-OSS Reasoning Effort

Available options:

```text
Low
Medium
High
```

---

## 7. Source Traceability

Every generated answer can display its retrieved sources.

The application shows:

* File name
* Sheet name
* Excel row range
* Chunk number
* Vector ID
* Similarity score
* Retrieved content

---

# 🗂️ Database Files

## academic_knowledge.faiss

Contains the vector index.

It is used for fast similarity search.

---

## metadata.json

Contains the text chunks and their metadata.

Example metadata:

```text
file_name
relative_path
sheet_name
row_start
row_end
chunk_number
total_chunks_in_sheet
columns
source_google_drive
indexed_at
embedding_model
chunk_id
vector_id
```

---

## config.json

Contains information about how the vector database was created.

Example:

```json
{
  "version": "1.0",
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "dimension": 384,
  "index_type": "IndexFlatIP",
  "similarity_metric": "cosine",
  "vectors_normalized": true
}
```

---

# 🧪 Embedding Model

The application uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The embedding dimension is:

```text
384
```

The same embedding model must be used during:

```text
Indexing
```

and:

```text
Query retrieval
```

Do not create the FAISS database with one embedding model and run the application with another.

---

# 🔄 Complete RAG Pipeline

The complete runtime pipeline is:

```text
Student Question
       │
       ▼
Sentence Transformer
       │
       ▼
Normalized Query Embedding
       │
       ▼
FAISS IndexFlatIP
       │
       ▼
Cosine Similarity
       │
       ▼
Top-K Documents
       │
       ▼
Similarity Threshold
       │
       ▼
Retrieved Chunks
       │
       ▼
Metadata + Context
       │
       ▼
Groq GPT-OSS 120B
       │
       ▼
Final Answer
       │
       ▼
Source Traceability
```

---

# 🛠️ Creating the Vector Database

The vector database should be created in Google Colab.

The indexing pipeline should:

1. Download/read Excel documents.
2. Read all relevant Excel sheets.
3. Clean the data.
4. Divide rows into chunks.
5. Create embeddings.
6. Normalize embeddings.
7. Create FAISS `IndexFlatIP`.
8. Store metadata.
9. Save the database.
10. Test retrieval.
11. Create the database ZIP file.

The original Excel documents are not required in the deployment repository.

---

# 📦 Database Output

After indexing, you should have:

```text
rag_database/
│
├── academic_knowledge.faiss
├── metadata.json
├── config.json
└── README.md
```

Upload this folder to your GitHub repository.

---

# 💻 Running Locally

Install the dependencies:

```bash
pip install -r requirements.txt
```

Then run:

```bash
streamlit run app.py
```

---

# ☁️ Deploying on Streamlit Cloud

## Step 1 — Create GitHub Repository

Create:

```text
university-academic-rag
```

---

## Step 2 — Upload Files

Upload:

```text
app.py
requirements.txt
README.md
.gitignore
```

Then create:

```text
rag_database
```

and upload:

```text
academic_knowledge.faiss
metadata.json
config.json
README.md
```

---

# Step 3 — Do NOT Upload Excel Files

Do not upload:

```text
.xlsx
.xls
.xlsm
```

files to the GitHub repository.

They are source documents used during the indexing stage only.

---

# Step 4 — Deploy

In Streamlit Cloud:

```text
Create App
    ↓
Select GitHub repository
    ↓
Select branch
    ↓
Select app.py
    ↓
Deploy
```

---

# Step 5 — Add Secret

Add:

```toml
GROQ_API_KEY = "your-api-key"
```

to Streamlit Cloud Secrets.

---

# 🔒 Security Recommendations

Do not commit:

```text
.env
*.xlsx
*.xls
*.xlsm
*.csv
```

or API keys.

For sensitive university information, use a **private GitHub repository** and carefully control who can access the Streamlit application and its database.

Also remember that `metadata.json` contains retrieved text chunks. Therefore, even though the original Excel files are not uploaded, the metadata database may still contain sensitive information.

---

# 🧹 Updating the Knowledge Base

When university documents change:

```text
New Excel Documents
        ↓
Google Drive
        ↓
Run indexing again in Colab
        ↓
Create new FAISS database
        ↓
Create new metadata
        ↓
Replace rag_database
        ↓
Push changes to GitHub
        ↓
Streamlit Cloud redeploys
```

Do not manually edit the FAISS file.

---

# ⚠️ Important Compatibility Rule

The following must remain consistent between indexing and deployment:

```text
Embedding model
Embedding dimension
Normalization
FAISS index type
Similarity metric
Metadata ordering
```

For this project:

```text
Embedding model:
sentence-transformers/all-MiniLM-L6-v2

Dimension:
384

FAISS:
IndexFlatIP

Similarity:
Cosine similarity

Normalization:
L2 normalization
```

---

# 📌 Project Summary

This project implements an academic RAG system that combines:

```text
Excel Knowledge
      +
Semantic Embeddings
      +
FAISS Vector Search
      +
Metadata Traceability
      +
Groq GPT-OSS 120B
      +
Streamlit
```

The system retrieves relevant academic information before generating an answer, reducing the need for the LLM to rely on unsupported information.

---

# 👩‍💻 Technology Stack

| Component            | Technology            |
| -------------------- | --------------------- |
| Frontend             | Streamlit             |
| LLM                  | Groq GPT-OSS 120B     |
| LLM Model ID         | `openai/gpt-oss-120b` |
| Embeddings           | Sentence Transformers |
| Embedding Model      | `all-MiniLM-L6-v2`    |
| Vector Database      | FAISS                 |
| Similarity           | Cosine Similarity     |
| Index                | `IndexFlatIP`         |
| Source Data          | Excel                 |
| Indexing Environment | Google Colab          |
| Deployment           | Streamlit Cloud       |
| Source Control       | GitHub                |

---

# 📄 License

This project can be adapted for educational, research, and university knowledge-assistant applications.

Before deploying real university data, verify the institution's privacy, security, and data-governance requirements.
