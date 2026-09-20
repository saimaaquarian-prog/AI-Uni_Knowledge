import json
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer

APP_TITLE = "University Academic Knowledge Assistant"
MODEL_NAME = "openai/gpt-oss-120b"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

DATABASE_DIR = Path("rag_database")
FAISS_FILE = DATABASE_DIR / "academic_knowledge.faiss"
METADATA_FILE = DATABASE_DIR / "metadata.json"
CONFIG_FILE = DATABASE_DIR / "config.json"

st.set_page_config(
page_title=APP_TITLE,
page_icon="🎓",
layout="wide",
initial_sidebar_state="expanded",
)

st.markdown(
""" <style>
.main-title {
font-size: 2.3rem;
font-weight: 700;
margin-bottom: 0.2rem;
}

```
.subtitle {
    font-size: 1.05rem;
    opacity: 0.75;
    margin-bottom: 1.5rem;
}
</style>
""",
unsafe_allow_html=True,
```

)

st.markdown(
f'<div class="main-title">🎓 {APP_TITLE}</div>',
unsafe_allow_html=True,
)

st.markdown(
""" <div class="subtitle">
Ask questions about university academic information
using a retrieval-augmented generation system. </div>
""",
unsafe_allow_html=True,
)

# ============================================================

# CHECK DATABASE FILES

# ============================================================

missing_files = []

if not FAISS_FILE.exists():
missing_files.append(str(FAISS_FILE))

if not METADATA_FILE.exists():
missing_files.append(str(METADATA_FILE))

if missing_files:
st.error("RAG database files are missing.")

```
st.write("Missing files:")

for file in missing_files:
    st.code(file)

st.info(
    "Make sure the rag_database folder is uploaded "
    "to the GitHub repository."
)

st.stop()
```

# ============================================================

# LOAD FAISS

# ============================================================

@st.cache_resource(show_spinner=False)
def load_faiss_index():
return faiss.read_index(str(FAISS_FILE))

# ============================================================

# LOAD METADATA

# ============================================================

@st.cache_data(show_spinner=False)
def load_metadata():
with open(
METADATA_FILE,
"r",
encoding="utf-8",
) as file:
return json.load(file)

# ============================================================

# LOAD CONFIG

# ============================================================

@st.cache_data(show_spinner=False)
def load_config():

```
if not CONFIG_FILE.exists():
    return {}

with open(
    CONFIG_FILE,
    "r",
    encoding="utf-8",
) as file:
    return json.load(file)
```

# ============================================================

# LOAD EMBEDDING MODEL

# ============================================================

@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
return SentenceTransformer(
EMBEDDING_MODEL_NAME
)

# ============================================================

# LOAD DATABASE

# ============================================================

try:
index = load_faiss_index()
database = load_metadata()
database_config = load_config()
embedding_model = load_embedding_model()

except Exception as error:

```
st.error("Failed to load the RAG database.")
st.exception(error)
st.stop()
```

# ============================================================

# READ RECORDS

# ============================================================

records = database.get("records", [])

if not records:
st.error(
"metadata.json does not contain any records."
)
st.stop()

# ============================================================

# CHECK VECTOR / METADATA COUNT

# ============================================================

if index.ntotal != len(records):

```
st.warning(
    f"FAISS contains {index.ntotal} vectors, "
    f"but metadata.json contains {len(records)} records. "
    "The FAISS index and metadata should be generated "
    "together."
)
```

# ============================================================

# GROQ API

# ============================================================

try:
groq_api_key = st.secrets["GROQ_API_KEY"]

except Exception:
groq_api_key = None

if not groq_api_key:

```
st.error("GROQ_API_KEY is not configured.")

st.info(
    "Add GROQ_API_KEY to Streamlit Cloud Secrets."
)

st.stop()
```

client = Groq(
api_key=groq_api_key
)

# ============================================================

# SESSION STATE

# ============================================================

if "messages" not in st.session_state:
st.session_state.messages = []

# ============================================================

# SIDEBAR

# ============================================================

with st.sidebar:

```
st.header("⚙️ RAG Settings")

st.markdown("### Retrieval")

max_sources = min(
    10,
    index.ntotal,
)

top_k = st.slider(
    "Number of sources",
    min_value=1,
    max_value=max_sources,
    value=min(5, max_sources),
    step=1,
    help="Number of chunks retrieved from FAISS.",
)

similarity_threshold = st.slider(
    "Minimum similarity",
    min_value=0.0,
    max_value=1.0,
    value=0.20,
    step=0.05,
    help="Minimum cosine similarity required.",
)

st.markdown("### Answer Settings")

technicality = st.select_slider(
    "Technicality",
    options=[
        "Beginner",
        "Intermediate",
        "Advanced",
    ],
    value="Intermediate",
)

response_length = st.select_slider(
    "Response length",
    options=[
        "Short",
        "Medium",
        "Detailed",
    ],
    value="Medium",
)

reasoning_effort = st.select_slider(
    "Reasoning effort",
    options=[
        "low",
        "medium",
        "high",
    ],
    value="medium",
)

show_scores = st.checkbox(
    "Show similarity scores",
    value=True,
)

st.divider()

st.markdown("### Database")

st.caption(
    f"Vectors: {index.ntotal:,}"
)

st.caption(
    f"Embedding dimension: {index.d}"
)

st.caption(
    f"Embedding model: {EMBEDDING_MODEL_NAME}"
)

st.caption(
    f"LLM: {MODEL_NAME}"
)

if database_config:

    st.caption(
        "Database version: "
        + str(
            database_config.get(
                "version",
                "Unknown",
            )
        )
    )

    st.caption(
        "Similarity: "
        + str(
            database_config.get(
                "similarity_metric",
                "Cosine",
            )
        )
    )

st.divider()

if st.button(
    "🗑️ Clear conversation",
    use_container_width=True,
):

    st.session_state.messages = []
    st.rerun()
```

# ============================================================

# RETRIEVAL

# ============================================================

def retrieve_documents(
query,
number_of_results,
threshold,
):

```
if not query.strip():
    return []

query_embedding = embedding_model.encode(
    [query],
    normalize_embeddings=True,
    convert_to_numpy=True,
    show_progress_bar=False,
)

query_embedding = np.asarray(
    query_embedding,
    dtype=np.float32,
)

if query_embedding.ndim == 1:

    query_embedding = query_embedding.reshape(
        1,
        -1,
    )

scores, indices = index.search(
    query_embedding,
    number_of_results,
)

retrieved = []

for score, vector_id in zip(
    scores[0],
    indices[0],
):

    vector_id = int(vector_id)

    if vector_id < 0:
        continue

    if vector_id >= len(records):
        continue

    score = float(score)

    score = max(
        -1.0,
        min(1.0, score),
    )

    if score < threshold:
        continue

    record = records[vector_id]

    metadata = dict(
        record.get(
            "metadata",
            {},
        )
    )

    metadata["vector_id"] = vector_id

    retrieved.append(
        {
            "vector_id": vector_id,
            "score": score,
            "text": record.get(
                "text",
                "",
            ),
            "metadata": metadata,
        }
    )

return retrieved
```

# ============================================================

# BUILD CONTEXT

# ============================================================

def build_context(
retrieved_documents,
):

```
context_parts = []

for number, document in enumerate(
    retrieved_documents,
    start=1,
):

    metadata = document["metadata"]

    source = (
        f"Source {number}\n"
        f"File: {metadata.get('file_name', 'Unknown')}\n"
        f"Sheet: {metadata.get('sheet_name', 'Unknown')}\n"
        f"Excel rows: "
        f"{metadata.get('row_start', '?')}-"
        f"{metadata.get('row_end', '?')}\n"
        f"Chunk: {metadata.get('chunk_number', '?')}\n"
        f"Vector ID: {metadata.get('vector_id', '?')}\n"
        f"Cosine similarity: "
        f"{document['score']:.4f}\n\n"
        f"Content:\n"
        f"{document['text']}"
    )

    context_parts.append(source)

return "\n\n".join(context_parts)
```

# ============================================================

# SYSTEM PROMPT

# ============================================================

def create_system_prompt():

```
return f"""
```

You are a university academic knowledge assistant.

Answer student questions using ONLY the retrieved
university knowledge provided to you.

LLM:
{MODEL_NAME}

Technicality:
{technicality}

Response length:
{response_length}

Rules:

1. Use retrieved university information as the primary
   source of truth.

2. Do not invent university policies, requirements,
   dates, fees, procedures, programs, rules, or
   academic information.

3. If the retrieved information does not contain the
   answer, clearly say that the knowledge base does not
   contain enough information.

4. You may combine information from multiple retrieved
   sources.

5. Preserve important qualifications and exceptions.

6. If sources contain conflicting information, explain
   the difference.

7. Keep responses appropriate for university students.

8. Use headings, bullets, or numbered steps when useful.

9. Provide a concise Sources section when appropriate.

10. Do not reveal hidden reasoning or chain-of-thought.

Provide only the useful answer and concise explanations.
"""

# ============================================================

# GENERATE ANSWER

# ============================================================

def generate_answer(
question,
context,
):

```
system_prompt = create_system_prompt()

user_prompt = f"""
```

Answer the following student question using only the
retrieved university context.

STUDENT QUESTION:
{question}

## RETRIEVED UNIVERSITY CONTEXT:

{context}

---

If the context does not support the answer, clearly
state that the available knowledge base does not contain
enough information.

Do not use unrelated information.
"""

```
response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ],
    reasoning_effort=reasoning_effort,
    temperature=0.2,
    max_completion_tokens=2500,
)

return response.choices[0].message.content
```

# ============================================================

# DISPLAY SOURCES

# ============================================================

def display_sources(
sources,
show_similarity=True,
):

```
if not sources:
    return

st.markdown("### 📚 Sources")

for source_number, source in enumerate(
    sources,
    start=1,
):

    metadata = source["metadata"]

    file_name = metadata.get(
        "file_name",
        "Unknown",
    )

    sheet_name = metadata.get(
        "sheet_name",
        "Unknown",
    )

    row_start = metadata.get(
        "row_start",
        "?",
    )

    row_end = metadata.get(
        "row_end",
        "?",
    )

    chunk_number = metadata.get(
        "chunk_number",
        "?",
    )

    with st.expander(
        f"Source {source_number} — {file_name}"
    ):

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("**Source file**")
            st.write(file_name)

            st.markdown("**Sheet**")
            st.write(sheet_name)

            st.markdown("**Chunk**")
            st.write(chunk_number)

        with col2:

            st.markdown("**Excel rows**")
            st.write(
                f"{row_start} – {row_end}"
            )

            st.markdown("**Vector ID**")
            st.write(
                source.get(
                    "vector_id",
                    "Unknown",
                )
            )

            if show_similarity:

                st.markdown(
                    "**Cosine similarity**"
                )

                st.write(
                    f"{source['score']:.4f}"
                )

        st.markdown(
            "**Retrieved content**"
        )

        st.text(
            source["text"]
        )
```

# ============================================================

# CHAT HISTORY

# ============================================================

for message in st.session_state.messages:

```
with st.chat_message(
    message["role"]
):

    st.markdown(
        message["content"]
    )

    if (
        message["role"] == "assistant"
        and message.get("sources")
    ):

        display_sources(
            message["sources"],
            show_scores,
        )
```

# ============================================================

# USER QUESTION

# ============================================================

question = st.chat_input(
"Ask a question about the university..."
)

if question:

```
question = question.strip()

if not question:
    st.stop()

st.session_state.messages.append(
    {
        "role": "user",
        "content": question,
    }
)

with st.chat_message("user"):
    st.markdown(question)

with st.chat_message("assistant"):

    with st.spinner(
        "Searching the academic knowledge base..."
    ):

        retrieved_documents = retrieve_documents(
            question,
            top_k,
            similarity_threshold,
        )

    if not retrieved_documents:

        answer = (
            "I could not find sufficiently relevant "
            "information in the university knowledge "
            "base to answer this question reliably."
        )

        st.markdown(answer)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": [],
            }
        )

    else:

        context = build_context(
            retrieved_documents
        )

        with st.spinner(
            "Generating answer with GPT-OSS 120B..."
        ):

            try:

                answer = generate_answer(
                    question,
                    context,
                )

            except Exception as error:

                st.error(
                    "The Groq request failed."
                )

                st.exception(error)
                st.stop()

        st.markdown(answer)

        display_sources(
            retrieved_documents,
            show_scores,
        )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": retrieved_documents,
            }
        )
```
