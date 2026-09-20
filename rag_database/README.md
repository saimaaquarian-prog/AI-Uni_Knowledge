# University Academic Knowledge Assistant

Precomputed RAG database.

## Vector Database

FAISS IndexFlatIP.

## Embedding Model

sentence-transformers/all-MiniLM-L6-v2

## Embedding Dimension

384

## Similarity

Cosine similarity.

Embeddings are normalized before being
stored in FAISS.

## Total Vectors

73

## Files

- academic_knowledge.faiss
- metadata.json
- config.json
- README.md

## Metadata

Every chunk stores:

- vector ID
- chunk ID
- file name
- relative path
- sheet name
- Excel row start
- Excel row end
- chunk number
- column names
- source Google Drive folder
- indexing timestamp
- embedding model

## Source

https://drive.google.com/drive/folders/1-3niStlwtNfaywiaUBADg4KCKkq_fe2g?usp=drive_link

## Architecture

Google Drive
-> Excel
-> Chunking
-> Embeddings
-> FAISS
-> Metadata
-> Streamlit
-> Groq

The Streamlit application should load
the existing FAISS database instead of
creating embeddings on every application run.

When source documents change, rebuild
the database and replace these files.