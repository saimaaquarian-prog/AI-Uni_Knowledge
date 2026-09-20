
import json
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer


# ============================================================
# APPLICATION CONFIGURATION
# ============================================================

APP_TITLE = "University Academic Knowledge Assistant"

MODEL_NAME = "openai/gpt-oss-120b"

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

DATABASE_DIR = Path("rag_database")

FAISS_FILE = (
    DATABASE_DIR / "academic_knowledge.faiss"
)

METADATA_FILE = (
    DATABASE_DIR / "metadata.json"
)

CONFIG_FILE = (
    DATABASE_DIR / "config.json"
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1.05rem;
        opacity: 0.75;
        margin-bottom: 1.5rem;
    }

    .source-card {
        padding: 0.8rem;
        border-radius: 0.6rem;
        border: 1px solid rgba(128,128,128,0.25);
        margin-bottom: 0.5rem;
    }

    .metric-label {
        font-size: 0.8rem;
        opacity: 0.7;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    f'<div class="main-title">🎓 {APP_TITLE}</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    Ask questions about university academic information
    using a retrieval-augmented generation system.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# VALIDATE DATABASE
# ============================================================

missing_files = []

if not FAISS_FILE.exists():
    missing_files.append(str(FAISS_FILE))

if not METADATA_FILE.exists():
    missing_files.append(str(METADATA_FILE))


if missing_files:

    st.error(
        "RAG database files are missing."
    )

    st.write(
        "The following files could not be found:"
    )

    for file in missing_files:
        st.code(file)

    st.info(
        "Upload the precomputed database files "
        "inside the rag_database folder."
    )

    st.stop()


# ============================================================
# LOAD DATABASE
# ============================================================

@st.cache_resource(show_spinner=False)
def load_faiss_index():

    return faiss.read_index(
        str(FAISS_FILE)
    )


@st.cache_data(show_spinner=False)
def load_metadata():

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


@st.cache_data(show_spinner=False)
def load_config():

    if not CONFIG_FILE.exists():
        return {}

    with open(
        CONFIG_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource(
    show_spinner="Loading embedding model..."
)
def load_embedding_model():

    return SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )


try:

    index = load_faiss_index()

    database = load_metadata()

    database_config = load_config()

    embedding_model = load_embedding_model()

except Exception as error:

    st.error(
        "Failed to load the RAG database."
    )

    st.exception(error)

    st.stop()


# ============================================================
# PREPARE METADATA RECORDS
# ============================================================

records = database.get(
    "records",
    []
)


if not records:

    st.error(
        "metadata.json does not contain "
        "any RAG records."
    )

    st.stop()


# ============================================================
# GROQ CLIENT
# ============================================================

try:

    groq_api_key = st.secrets[
        "GROQ_API_KEY"
    ]

except Exception:

    groq_api_key = None


if not groq_api_key:

    st.warning(
        "GROQ_API_KEY is not configured."
    )

    st.info(
        "Add GROQ_API_KEY to your Streamlit "
        "Cloud Secrets."
    )

    st.stop()


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

    st.header("⚙️ RAG Settings")

    st.markdown(
        "### Retrieval"
    )

    top_k = st.slider(
        "Number of sources",
        min_value=1,
        max_value=min(
            10,
            len(records)
        ),
        value=min(
            5,
            len(records)
        ),
        step=1,
        help=(
            "Number of document chunks "
            "retrieved from FAISS."
        ),
    )


    similarity_threshold = st.slider(
        "Minimum similarity",
        min_value=0.0,
        max_value=1.0,
        value=0.20,
        step=0.05,
        help=(
            "Retrieved chunks below this "
            "similarity are ignored."
        ),
    )


    st.markdown(
        "### Answer Settings"
    )


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
        help=(
            "GPT-OSS 120B supports low, medium "
            "and high reasoning effort."
        ),
    )


    show_scores = st.checkbox(
        "Show similarity scores",
        value=True,
    )


    st.divider()


    st.markdown(
        "### Database"
    )


    st.caption(
        f"Vectors: {index.ntotal}"
    )

    st.caption(
        f"Embedding dimension: {index.d}"
    )

    st.caption(
        f"Model: {MODEL_NAME}"
    )


    if database_config:

        st.caption(
            "Database version: "
            + str(
                database_config.get(
                    "version",
                    "Unknown"
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


# ============================================================
# RETRIEVAL FUNCTION
# ============================================================

def retrieve_documents(
    query,
    number_of_results,
    threshold,
):

    # Create query embedding
    query_embedding = (
        embedding_model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
    )


    query_embedding = np.asarray(
        query_embedding,
        dtype="float32",
    )


    # Search FAISS
    scores, indices = index.search(
        query_embedding,
        number_of_results,
    )


    retrieved = []


    for score, vector_id in zip(
        scores[0],
        indices[0],
    ):

        vector_id = int(
            vector_id
        )

        score = float(
            score
        )


        if vector_id < 0:

            continue


        if vector_id >= len(records):

            continue


        if score < threshold:

            continue


        record = records[
            vector_id
        ]


        retrieved.append(
            {
                "vector_id":
                    vector_id,

                "score":
                    score,

                "text":
                    record.get(
                        "text",
                        ""
                    ),

                "metadata":
                    record.get(
                        "metadata",
                        {}
                    ),
            }
        )


    return retrieved


# ============================================================
# CONTEXT BUILDER
# ============================================================

def build_context(
    retrieved_documents
):

    context_parts = []


    for number, document in enumerate(
        retrieved_documents,
        start=1,
    ):

        metadata = document[
            "metadata"
        ]


        source = (
            f"Source {number}\n"
            f"File: "
            f"{metadata.get('file_name', 'Unknown')}\n"
            f"Sheet: "
            f"{metadata.get('sheet_name', 'Unknown')}\n"
            f"Excel rows: "
            f"{metadata.get('row_start', '?')}-"
            f"{metadata.get('row_end', '?')}\n"
            f"Chunk: "
            f"{metadata.get('chunk_number', '?')}\n"
            f"Similarity: "
            f"{document['score']:.4f}\n\n"
            f"Content:\n"
            f"{document['text']}"
        )


        context_parts.append(
            source
        )


    return "\n\n".join(
        context_parts
    )


# ============================================================
# SYSTEM PROMPT
# ============================================================

def create_system_prompt():

    return f"""
You are a university academic knowledge assistant.

Your task is to answer the student's question using
ONLY the retrieved university knowledge supplied
in the context.

You are powered by:
{MODEL_NAME}

Technicality level:
{technicality}

Requested response length:
{response_length}

IMPORTANT RAG RULES:

1. Use the retrieved context as the primary source
   of truth.

2. Do not invent university policies, requirements,
   dates, procedures, rules, programs, fees, or
   academic information.

3. If the answer is not supported by the retrieved
   context, clearly say that the available university
   knowledge base does not contain enough information.

4. You may explain or synthesize information from
   multiple retrieved sources.

5. Do not claim that a source says something when it
   does not.

6. Preserve important qualifications and exceptions.

7. If multiple sources contain different information,
   clearly identify the difference instead of silently
   choosing one.

8. Keep the answer appropriate for university students.

9. When useful, organize the answer using headings,
   bullets, numbered steps, or tables.

10. At the end, provide a concise "Sources" section.
    The application will separately display detailed
    source metadata.

Do not reveal hidden reasoning or internal chain-of-thought.
Provide only the useful answer and concise explanations.
"""


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    question,
    context,
):

    system_prompt = create_system_prompt()


    user_prompt = f"""
Answer the following student question using the
retrieved university context.

STUDENT QUESTION:
{question}

RETRIEVED UNIVERSITY CONTEXT:
------------------------------

{context}

------------------------------

If the context does not support the answer,
say so clearly.

Do not use information that is unrelated to
the retrieved university documents.
"""


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


    return response.choices[
        0
    ].message.content


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

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

            st.markdown(
                "### 📚 Sources"
            )


            for source_number, source in enumerate(
                message["sources"],
                start=1,
            ):

                metadata = source[
                    "metadata"
                ]


                title = (
                    f"Source {source_number}: "
                    f"{metadata.get('file_name', 'Unknown')}"
                )


                with st.expander(
                    title
                ):

                    col1, col2 = st.columns(
                        2
                    )


                    with col1:

                        st.markdown(
                            "**File**"
                        )

                        st.write(
                            metadata.get(
                                "file_name",
                                "Unknown"
                            )
                        )


                        st.markdown(
                            "**Sheet**"
                        )

                        st.write(
                            metadata.get(
                                "sheet_name",
                                "Unknown"
                            )
                        )


                        st.markdown(
                            "**Chunk**"
                        )

                        st.write(
                            metadata.get(
                                "chunk_number",
                                "Unknown"
                            )
                        )


                    with col2:

                        st.markdown(
                            "**Excel Rows**"
                        )

                        st.write(
                            f"{metadata.get('row_start', '?')}"
                            f" – "
                            f"{metadata.get('row_end', '?')}"
                        )


                        st.markdown(
                            "**Vector ID**"
                        )

                        st.write(
                            metadata.get(
                                "vector_id",
                                "Unknown"
                            )
                        )


                        if show_scores:

                            st.markdown(
                                "**Similarity**"
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


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask a question about the university..."
)


if question:

    question = question.strip()


    if not question:

        st.stop()


    # --------------------------------------------------------
    # DISPLAY USER QUESTION
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )


    with st.chat_message(
        "user"
    ):

        st.markdown(
            question
        )


    # --------------------------------------------------------
    # RETRIEVE SOURCES
    # --------------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

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
                "I could not find sufficiently "
                "relevant information in the "
                "university knowledge base to "
                "answer this question reliably."
            )


            st.markdown(
                answer
            )


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


            # ------------------------------------------------
            # GENERATE ANSWER
            # ------------------------------------------------

            with st.spinner(
                "Generating answer..."
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

                    st.exception(
                        error
                    )

                    st.stop()


            st.markdown(
                answer
            )


            # ------------------------------------------------
            # SOURCE TRACEABILITY
            # ------------------------------------------------

            st.markdown(
                "### 📚 Sources"
            )


            for source_number, source in enumerate(
                retrieved_documents,
                start=1,
            ):

                metadata = source[
                    "metadata"
                ]


                file_name = metadata.get(
                    "file_name",
                    "Unknown"
                )

                sheet_name = metadata.get(
                    "sheet_name",
                    "Unknown"
                )

                row_start = metadata.get(
                    "row_start",
                    "?"
                )

                row_end = metadata.get(
                    "row_end",
                    "?"
                )

                chunk_number = metadata.get(
                    "chunk_number",
                    "?"
                )


                title = (
                    f"Source {source_number} — "
                    f"{file_name}"
                )


                with st.expander(
                    title
                ):

                    col1, col2 = st.columns(
                        2
                    )


                    with col1:

                        st.markdown(
                            "**Source file**"
                        )

                        st.write(
                            file_name
                        )


                        st.markdown(
                            "**Sheet**"
                        )

                        st.write(
                            sheet_name
                        )


                        st.markdown(
                            "**Chunk**"
                        )

                        st.write(
                            chunk_number
                        )


                    with col2:

                        st.markdown(
                            "**Excel rows**"
                        )

                        st.write(
                            f"{row_start} – {row_end}"
                        )


                        if show_scores:

                            st.markdown(
                                "**Similarity score**"
                            )

                            st.write(
                                f"{source['score']:.4f}"
                            )


                        st.markdown(
                            "**Vector ID**"
                        )

                        st.write(
                            metadata.get(
                                "vector_id",
                                "Unknown"
                            )
                        )


                    st.markdown(
                        "**Retrieved content**"
                    )

                    st.text(
                        source["text"]
                    )


            # ------------------------------------------------
            # SAVE ASSISTANT MESSAGE
            # ------------------------------------------------

            st.session_state.messages.append(
                {
                    "role": "assistant",

                    "content":
                        answer,

                    "sources":
                        retrieved_documents,
                }
            )
```
