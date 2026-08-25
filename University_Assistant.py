import streamlit as st
import os
import requests
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Groq API
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant").strip()


def load_api_key():
    """Load the Groq key from an environment variable or local .env file."""
    env_key = os.environ.get("GROQ_API_KEY", "").strip()
    if env_key:
        return env_key

    if os.path.exists(".env"):
        with open(".env", "r", encoding="utf-8") as env_file:
            for line in env_file:
                if line.strip().startswith("GROQ_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")

    return ""


API_KEY = load_api_key()

template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""

pdfs_directory = 'chat-with-pdf/pdfs/'
os.makedirs(pdfs_directory, exist_ok=True)

# In-memory local search index
documents_store = []
vectorizer = None
document_matrix = None

def call_groq_api(url, body):
    if not API_KEY:
        raise RuntimeError("Missing Groq API key. Add GROQ_API_KEY to your .env file.")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }
    response = requests.post(url, headers=headers, json=body, timeout=60)

    try:
        result = response.json()
    except ValueError:
        result = {"error": {"message": response.text}}

    if not response.ok:
        message = result.get("error", {}).get("message", response.text)
        raise RuntimeError(f"Groq API error {response.status_code}: {message}")

    return result

# ---- Groq Answer Generator ----
def answer_question_with_groq(question, context):
    prompt = template.format(question=question, context=context)

    body = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
    }

    result = call_groq_api(GROQ_CHAT_URL, body)
    try:
        return result["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, AttributeError) as error:
        raise RuntimeError(f"Could not read Groq response: {result}") from error

# ---- Text Handling ----
def upload_pdf(file):
    with open(pdfs_directory + file.name, "wb") as f:
        f.write(file.getbuffer())

def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    return loader.load()

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_splitter.split_documents(documents)

# ---- Indexing ----
def index_documents(chunks):
    global documents_store, vectorizer, document_matrix

    documents_store = [chunk.page_content for chunk in chunks if chunk.page_content.strip()]
    vectorizer = TfidfVectorizer(stop_words="english")
    document_matrix = vectorizer.fit_transform(documents_store)


def has_indexed_documents():
    return bool(documents_store) and vectorizer is not None and document_matrix is not None

# ---- Retrieval ----
def retrieve_relevant_docs(query, top_k=3):
    if not has_indexed_documents():
        return []

    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, document_matrix)[0]
    top_indexes = similarities.argsort()[::-1][:top_k]

    return [
        documents_store[index]
        for index in top_indexes
        if similarities[index] > 0
    ]


def ask_assistant(question):
    relevant_contexts = retrieve_relevant_docs(question)
    if not relevant_contexts:
        return "I could not find relevant information in the uploaded PDF."

    context = "\n\n".join(relevant_contexts)
    return answer_question_with_groq(question, context)


def apply_page_styles():
    st.markdown(
        """
        <style>
            :root {
                --ink: #17202a;
                --muted: #64748b;
                --panel: #ffffff;
                --line: #d8e0ea;
                --brand: #0f766e;
                --brand-dark: #134e4a;
                --accent: #d97706;
                --wash: #f6f8fb;
            }

            .stApp {
                background: linear-gradient(135deg, #f7fbfa 0%, #eef4f8 46%, #fffaf1 100%);
                color: var(--ink);
            }

            [data-testid="stHeader"] {
                background: transparent;
            }

            [data-testid="stSidebar"] {
                background: rgba(255, 255, 255, 0.86);
                border-right: 1px solid rgba(216, 224, 234, 0.9);
            }

            [data-testid="stSidebar"] h1,
            [data-testid="stSidebar"] h2,
            [data-testid="stSidebar"] h3,
            [data-testid="stSidebar"] h4,
            [data-testid="stSidebar"] h5,
            [data-testid="stSidebar"] h6,
            [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
                color: #000000 !important;
            }

            .main .block-container {
                max-width: 1120px;
                padding-top: 2rem;
                padding-bottom: 6rem;
            }

            .hero {
                display: grid;
                grid-template-columns: 1fr auto;
                gap: 1.5rem;
                align-items: center;
                padding: 1.5rem;
                border: 1px solid rgba(216, 224, 234, 0.9);
                border-radius: 8px;
                background:
                    linear-gradient(90deg, rgba(255,255,255,0.95), rgba(255,255,255,0.72)),
                    repeating-linear-gradient(135deg, rgba(15,118,110,0.10) 0 1px, transparent 1px 16px);
                box-shadow: 0 18px 50px rgba(15, 23, 42, 0.08);
            }

            .hero h1 {
                margin: 0;
                font-size: clamp(2rem, 4vw, 3.4rem);
                line-height: 1.03;
                letter-spacing: 0;
                color: var(--brand-dark);
            }

            .hero p {
                margin: 0.7rem 0 0;
                color: var(--muted);
                font-size: 1.02rem;
                max-width: 46rem;
            }

            .hero-badge {
                width: 8.5rem;
                aspect-ratio: 1;
                display: grid;
                place-items: center;
                border-radius: 8px;
                background: linear-gradient(145deg, var(--brand), #1d4ed8);
                color: white;
                font-weight: 800;
                font-size: 2rem;
                box-shadow: inset 0 1px 0 rgba(255,255,255,0.24), 0 20px 40px rgba(15, 118, 110, 0.24);
            }

            .stat-row {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 0.85rem;
                margin: 1rem 0 1.2rem;
            }

            .stat {
                padding: 1rem;
                border-radius: 8px;
                background: rgba(255, 255, 255, 0.78);
                border: 1px solid rgba(216, 224, 234, 0.9);
            }

            .stat span {
                display: block;
                color: var(--muted);
                font-size: 0.78rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
            }

            .stat strong {
                display: block;
                margin-top: 0.3rem;
                font-size: 1.4rem;
                color: var(--ink);
            }

            .section-title {
                margin: 1.4rem 0 0.55rem;
                color: #000000;
                font-size: 1rem;
                font-weight: 800;
            }

            .stButton > button {
                width: 100%;
                min-height: 2.65rem;
                border-radius: 8px;
                border: 1px solid rgba(15, 118, 110, 0.22);
                background: rgba(255, 255, 255, 0.84);
                color: var(--brand-dark);
                font-weight: 700;
                transition: 140ms ease;
            }

            .stButton > button:hover {
                border-color: var(--brand);
                background: #f0fdfa;
                color: var(--brand-dark);
                transform: translateY(-1px);
            }

            .stButton > button,
            .stButton > button * {
                color: #000000 !important;
            }

            [data-testid="stChatMessage"] {
                border-radius: 8px;
                border: 1px solid rgba(216, 224, 234, 0.86);
                background: rgba(255,255,255,0.76);
                box-shadow: 0 10px 30px rgba(15, 23, 42, 0.05);
            }

            [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
            [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] * {
                color: #000000 !important;
            }

            [data-testid="stAlert"] * {
                color: #000000 !important;
            }

            [data-testid="stChatInput"] {
                background: rgba(255,255,255,0.88);
                border-top: 1px solid rgba(216, 224, 234, 0.9);
            }

            [data-testid="stChatInput"] textarea,
            [data-testid="stChatInput"] textarea::placeholder {
                color: #000000 !important;
                -webkit-text-fill-color: #000000 !important;
            }

            @media (max-width: 760px) {
                .hero {
                    grid-template-columns: 1fr;
                }

                .hero-badge {
                    width: 100%;
                    aspect-ratio: 4 / 1;
                }

                .stat-row {
                    grid-template-columns: 1fr;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(document_name, chunk_count, message_count):
    st.markdown(
        f"""
        <div class="hero">
            <div>
                <h1>University Assistant</h1>
                <p>Ask direct questions from your university PDF and get concise answers backed by retrieved document context.</p>
            </div>
            <div class="hero-badge">UA</div>
        </div>
        <div class="stat-row">
            <div class="stat"><span>Document</span><strong>{document_name}</strong></div>
            <div class="stat"><span>Indexed Chunks</span><strong>{chunk_count}</strong></div>
            <div class="stat"><span>Messages</span><strong>{message_count}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def initialize_state():
    defaults = {
        "messages": [],
        "document_name": "Not loaded",
        "chunk_count": 0,
        "last_uploaded_name": None,
        "pending_question": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def set_pending_question(question):
    st.session_state.pending_question = question


# ---- Streamlit App ----
st.set_page_config(
    page_title="University Assistant",
    page_icon="UA",
    layout="wide",
    initial_sidebar_state="expanded",
)

initialize_state()
apply_page_styles()

with st.sidebar:
    st.markdown("### Document")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf", label_visibility="collapsed")

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()

    st.markdown("### Sample Questions")
    sample_questions = [
        "What documents are required for admission?",
        "What is the admission process?",
        "What scholarships are available?",
        "Is hostel facility available?",
        "What is the fee structure?",
        "What programs are offered?",
    ]

    for sample_question in sample_questions:
        st.button(
            sample_question,
            key=f"sample_{sample_question}",
            on_click=set_pending_question,
            args=(sample_question,),
        )

if uploaded_file:
    is_new_document = st.session_state.last_uploaded_name != uploaded_file.name
    upload_pdf(uploaded_file)
    docs = load_pdf(os.path.join(pdfs_directory, uploaded_file.name))
    chunks = split_text(docs)
    try:
        index_documents(chunks)
        st.session_state.document_name = uploaded_file.name
        st.session_state.chunk_count = len(documents_store)

        if is_new_document:
            st.session_state.messages = []
            st.session_state.last_uploaded_name = uploaded_file.name
            st.toast("PDF processed and indexed.")
    except Exception as e:
        st.error(str(e))
        st.stop()

render_header(
    st.session_state.document_name,
    st.session_state.chunk_count,
    len(st.session_state.messages),
)

if not uploaded_file:
    st.info("Upload a university PDF from the sidebar to begin.")

st.markdown('<div class="section-title">Conversation</div>', unsafe_allow_html=True)

if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.write("Ready when your PDF is loaded.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

question = st.session_state.pending_question or st.chat_input("Ask a question about the uploaded PDF")
st.session_state.pending_question = None

if question:
    if not uploaded_file or not has_indexed_documents():
        st.warning("Upload and process a PDF first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Reading the document context..."):
            try:
                answer = ask_assistant(question)
            except Exception as e:
                answer = f"Error: {e}"
        st.write(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
