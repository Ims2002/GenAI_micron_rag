"""
Interfaz de chat en Streamlit para el Asistente Experto en Micron Technology.

Extensión opcional (bonus) del proyecto: reutiliza la misma base de conocimiento vectorial
(ChromaDB + Gemini Embeddings) y el mismo agente LangGraph + Gemini definidos en
Micron_RAG_Agent.ipynb, mostrados aquí en una interfaz visual de chat en vez del notebook.

Requisito previo: haber ejecutado el notebook al menos hasta la sección 5 (indexación en
ChromaDB), para que exista la colección persistida en ./chroma_db_micron.

Ejecutar con:
    streamlit run streamlit_app.py
"""

import os
import uuid

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

CHAT_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "gemini-embedding-001"
PERSIST_DIR = "./chroma_db_micron"
COLLECTION_NAME = "micron_knowledge_base"

# Mismo system prompt justificado que en la sección 7 del notebook (mantenido en un único sitio
# lógico del proyecto; si lo cambias aquí, actualiza también el notebook para que coincidan).
SYSTEM_PROMPT = """Eres un analista experto en los informes públicos de Micron Technology, Inc.
(NASDAQ: MU) presentados ante la SEC: sus informes anuales (Form 10-K) y sus proxy statements
(DEF 14A) de los últimos dos años fiscales.

INSTRUCCIONES:
1. Usa SIEMPRE la herramienta `buscar_en_informes_micron` para fundamentar tus respuestas antes
   de responder preguntas sobre Micron (finanzas, riesgos, negocio, gobierno corporativo,
   compensación de ejecutivos, junta directiva, etc.).
2. Responde ÚNICAMENTE con base en la información recuperada de los documentos indexados. No
   inventes cifras ni completes huecos con conocimiento general.
3. Si la información recuperada no es suficiente para responder con precisión, dilo
   explícitamente: "No tengo información suficiente en los documentos indexados para responder
   con precisión a esa pregunta."
4. Cuando cites datos, menciona el documento y año fiscal de origen (por ejemplo: "según el 10-K
   de FY2025...").
5. NO des recomendaciones de inversión ni opiniones sobre si comprar, vender o mantener la
   acción de Micron. Si te lo piden, aclara que solo puedes reportar lo que indican los
   documentos públicos, no ofrecer asesoría financiera.
6. Mantén coherencia con las preguntas anteriores de la conversación.
7. Si la pregunta no tiene relación con Micron o sus informes, indícalo brevemente y redirige
   al usuario al alcance del asistente.
8. Responde siempre en español, de forma clara y profesional.
"""

# ---------------------------------------------------------------------------
# Identidad visual: insignia SVG original tipo "chip de memoria" con el ticker
# bursátil de Micron (MU). No reproduce el logotipo corporativo real de Micron
# (que está registrado); es un diseño propio inspirado en un chip + ticker.
# ---------------------------------------------------------------------------
MU_LOGO_SVG = (
    '<svg width="56" height="56" viewBox="0 0 72 72" xmlns="http://www.w3.org/2000/svg" '
    'role="img" aria-label="Insignia MU">'
    '<defs><linearGradient id="muGrad" x1="0%" y1="0%" x2="100%" y2="100%">'
    '<stop offset="0%" stop-color="#0B3D91"/><stop offset="100%" stop-color="#00B4D8"/>'
    '</linearGradient></defs>'
    '<rect x="2" y="2" width="68" height="68" rx="16" fill="url(#muGrad)"/>'
    '<g stroke="#ffffff" stroke-width="2" opacity="0.55">'
    '<line x1="14" y1="6" x2="14" y2="14"/><line x1="26" y1="6" x2="26" y2="14"/>'
    '<line x1="46" y1="6" x2="46" y2="14"/><line x1="58" y1="6" x2="58" y2="14"/>'
    '<line x1="14" y1="58" x2="14" y2="66"/><line x1="26" y1="58" x2="26" y2="66"/>'
    '<line x1="46" y1="58" x2="46" y2="66"/><line x1="58" y1="58" x2="58" y2="66"/>'
    '<line x1="6" y1="14" x2="14" y2="14"/><line x1="6" y1="26" x2="14" y2="26"/>'
    '<line x1="6" y1="46" x2="14" y2="46"/><line x1="6" y1="58" x2="14" y2="58"/>'
    '<line x1="58" y1="14" x2="66" y2="14"/><line x1="58" y1="26" x2="66" y2="26"/>'
    '<line x1="58" y1="46" x2="66" y2="46"/><line x1="58" y1="58" x2="66" y2="58"/>'
    '</g>'
    '<rect x="14" y="14" width="44" height="44" rx="8" fill="#ffffff" opacity="0.08"/>'
    '<text x="36" y="44" font-family="Arial, Helvetica, sans-serif" font-size="22" '
    'font-weight="700" fill="#ffffff" text-anchor="middle">MU</text>'
    '</svg>'
)
SVG_ORIG_SIZE = 'width="56" height="56"'
SVG_SIDEBAR_SIZE = 'width="40" height="40"'

st.set_page_config(page_title="Asistente Micron (MU)", page_icon="🧠", layout="centered")

# ---------------------------------------------------------------------------
# Estilos personalizados
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f4f9fc 0%, #ffffff 55%);
    }

    .mu-header {
        display: flex;
        align-items: center;
        gap: 18px;
        padding-bottom: 16px;
        margin-bottom: 8px;
        border-bottom: 1px solid #dde6ee;
    }
    .mu-header-text h1 {
        margin: 0;
        font-size: 1.5rem;
        font-weight: 700;
        color: #0b3d91;
        line-height: 1.2;
    }
    .mu-header-text p {
        margin: 4px 0 0 0;
        color: #5b6b7c;
        font-size: 0.92rem;
    }

    .mu-pill-row {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin: 14px 0 22px 0;
    }
    .mu-pill {
        display: inline-block;
        background: #eaf6fb;
        color: #0b3d91;
        border: 1px solid #cfe8f5;
        border-radius: 999px;
        padding: 4px 12px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }

    [data-testid="stChatMessage"] {
        border-radius: 16px;
        padding: 2px 4px;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b3d91 0%, #082a66 100%);
    }
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] li,
    section[data-testid="stSidebar"] span {
        color: #eaf3fb !important;
    }
    section[data-testid="stSidebar"] .stButton > button {
        background-color: #00b4d8;
        color: #04263f;
        border: none;
        font-weight: 700;
        border-radius: 8px;
        width: 100%;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background-color: #48cae4;
        color: #04263f;
    }
    section[data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.2);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Cabecera visual (insignia SVG + título)
# ---------------------------------------------------------------------------
st.markdown(
    f'<div class="mu-header">{MU_LOGO_SVG}'
    f'<div class="mu-header-text">'
    f'<h1>Asistente Experto en Micron Technology</h1>'
    f'<p>RAG con Gemini + ChromaDB + LangGraph, sobre informes públicos de la SEC.</p>'
    f'</div></div>'
    f'<div class="mu-pill-row">'
    f'<span class="mu-pill">NASDAQ: MU</span>'
    f'<span class="mu-pill">Semiconductores</span>'
    f'<span class="mu-pill">10-K FY2024 · FY2025</span>'
    f'<span class="mu-pill">Proxy Statement 2024 · 2025</span>'
    f'<span class="mu-pill">No es asesoría de inversión</span>'
    f'</div>',
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Cargando base de conocimiento y agente...")
def cargar_agente():
    if not API_KEY:
        st.error(
            "No se encontró GEMINI_API_KEY. Crea un archivo .env con tu clave "
            "(ver .env.example) antes de continuar."
        )
        st.stop()

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=API_KEY)
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )

    if vectorstore._collection.count() == 0:
        st.error(
            "La base de conocimiento vectorial está vacía. Ejecuta primero "
            "Micron_RAG_Agent.ipynb hasta completar la sección 5 (indexación en ChromaDB)."
        )
        st.stop()

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    @tool
    def buscar_en_informes_micron(query: str) -> str:
        """Busca información relevante en los informes 10-K y proxy statements (DEF 14A) de
        Micron Technology. Úsala para cualquier pregunta sobre finanzas, riesgos, negocio,
        segmentos, gobierno corporativo, junta directiva o compensación de ejecutivos de
        Micron."""
        docs = retriever.invoke(query)
        if not docs:
            return "No se encontró información relevante en los documentos indexados."

        resultados = []
        for doc in docs:
            fuente = doc.metadata.get("source_name", "desconocida")
            tipo = doc.metadata.get("tipo", "")
            anio = doc.metadata.get("anio_fiscal", "")
            pagina = doc.metadata.get("page", "?")
            resultados.append(
                f"[Fuente: {fuente} | {tipo} FY{anio} | página {pagina}]\n{doc.page_content}"
            )
        return "\n\n---\n\n".join(resultados)

    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0.2, google_api_key=API_KEY)
    checkpointer = InMemorySaver()
    agente = create_agent(
        model=llm,
        tools=[buscar_en_informes_micron],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )
    return agente


agente_micron = cargar_agente()

if "thread_id" not in st.session_state:
    # Cada sesión de navegador tiene su propio thread_id -> memoria de conversación aislada por
    # usuario/pestaña, igual que la verificación multi-thread de la sección 12 del notebook.
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

AVATAR_USUARIO = "🧑‍💼"
AVATAR_ASISTENTE = "🧠"

with st.sidebar:
    st.markdown(
        f'<div style="display:flex; align-items:center; gap:10px; margin-bottom:6px;">'
        f'{MU_LOGO_SVG.replace(SVG_ORIG_SIZE, SVG_SIDEBAR_SIZE)}'
        f'<h2 style="margin:0; font-size:1.1rem;">Sobre este asistente</h2>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "Responde preguntas sobre **Micron Technology (NASDAQ: MU)** usando exclusivamente "
        "información de sus informes 10-K y proxy statements (DEF 14A) indexados en ChromaDB.\n\n"
        "- No inventa cifras: solo usa lo recuperado de los documentos.\n"
        "- Cita documento y año fiscal de origen.\n"
        "- No da recomendaciones de inversión."
    )
    st.divider()
    if st.button("🗑️ Nueva conversación"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

# Historial de la conversación actual
for msg in st.session_state.messages:
    avatar = AVATAR_USUARIO if msg["role"] == "user" else AVATAR_ASISTENTE
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

pregunta = st.chat_input("Escribe tu pregunta sobre Micron...")

if pregunta:
    st.session_state.messages.append({"role": "user", "content": pregunta})
    with st.chat_message("user", avatar=AVATAR_USUARIO):
        st.markdown(pregunta)

    with st.chat_message("assistant", avatar=AVATAR_ASISTENTE):
        with st.spinner("Consultando los informes de Micron..."):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            respuesta = agente_micron.invoke(
                {"messages": [HumanMessage(content=pregunta)]},
                config=config,
            )
            texto_respuesta = respuesta["messages"][-1].content
        st.markdown(texto_respuesta)

    st.session_state.messages.append({"role": "assistant", "content": texto_respuesta})
