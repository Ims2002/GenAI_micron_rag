# Asistente Experto en Micron Technology — RAG + Gemini + LangGraph

Proyecto Final del módulo de IA Generativa. Agente conversacional experto en los informes
públicos de **Micron Technology, Inc. (NASDAQ: MU)**, construido con Retrieval-Augmented
Generation (RAG), Google Gemini y LangGraph.

## 1. Dominio elegido

**Finanzas corporativas y gobierno corporativo de Micron Technology.** El agente responde
preguntas sobre el negocio, los resultados financieros, los factores de riesgo, la junta
directiva y la compensación de ejecutivos de Micron, a partir de documentos públicos
presentados ante la SEC (Securities and Exchange Commission de EE. UU.):

| Archivo esperado en `data/micron_docs/` | Tipo | Contenido | Fuente oficial (SEC EDGAR) |
|---|---|---|---|
| `micron_10K_FY2025.pdf` | Form 10-K | Negocio, riesgos y resultados FY2025 | https://www.sec.gov/Archives/edgar/data/723125/000072312525000028/mu-20250828.htm |
| `micron_10K_FY2024.pdf` | Form 10-K | Negocio, riesgos y resultados FY2024 | https://www.sec.gov/Archives/edgar/data/723125/000072312524000027/mu-20240829.htm |
| `micron_proxy_2025.pdf` | DEF 14A (proxy statement) | Gobierno corporativo y compensación (nov. 2025) | https://www.sec.gov/Archives/edgar/data/723125/000072312525000038/mu-20251125.htm |
| `micron_proxy_2024.pdf` | DEF 14A (proxy statement) | Gobierno corporativo y compensación (nov. 2024) | https://www.sec.gov/Archives/edgar/data/723125/000072312524000039/mu-20241126.htm |

Se eligieron 2 informes anuales + 2 proxy statements (en vez de un único documento) para: (a)
superar ampliamente el mínimo del proyecto (3 documentos / ~20 páginas), (b) cubrir dos
dimensiones complementarias del dominio (desempeño financiero y gobierno corporativo), y (c)
permitir preguntas comparativas interanuales que además sirven para demostrar la memoria de
conversación del agente.

**Cómo obtener los documentos:** el notebook no descarga los ficheros automáticamente (SEC EDGAR
bloquea descargas automatizadas sin las cabeceras adecuadas). Descárgalos manualmente:
abre cada enlace de la tabla anterior en tu navegador y guárdalo como PDF (Ctrl+P → "Guardar
como PDF") o como HTML (Ctrl+S), con el nombre indicado en la primera columna, dentro de
`data/micron_docs/`. El notebook admite ambos formatos.

## 2. Stack tecnológico

- **LLM y Embeddings:** Google Gemini (`langchain-google-genai`)
- **Base de conocimiento vectorial:** ChromaDB (`langchain-chroma`)
- **Framework de agente:** LangGraph, con LangChain como base (`create_agent`)
- **Memoria de conversación:** `langgraph.checkpoint.memory.InMemorySaver`
- **Entorno:** Jupyter Notebook

### ¿Por qué Gemini y no Claude?

Se evaluó usar la API de Claude (Anthropic) como LLM. Se descartó por dos motivos: (1) el
guideline del proyecto exige explícitamente Gemini tanto para el LLM como para los embeddings,
y la rúbrica de evaluación puntúa "Chroma + Gemini Embeddings" y "RAG con Gemini" como criterios
independientes — usar otro proveedor pone en riesgo esos dos criterios; y (2) Anthropic no ofrece
una API pública de embeddings, por lo que aun usando Claude como LLM habría sido necesario
recurrir a otro proveedor (Gemini, OpenAI o un modelo local) solo para los embeddings. Se optó
por mantener Gemini de extremo a extremo, tal como pide el guideline.

## 3. Instalación y ejecución

### Requisitos

- Python 3.10+
- Una API key de Google Gemini (gratuita): https://aistudio.google.com/app/apikey
- Los 4 documentos de Micron descargados en `data/micron_docs/` (ver sección 1)

### Pasos

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Configurar la API key
cp .env.example .env
# Edita .env y añade tu GEMINI_API_KEY

# 3. Abrir y ejecutar el notebook
jupyter notebook Micron_RAG_Agent.ipynb
```

Ejecuta las celdas en orden. La celda de creación de la base vectorial puede tardar varios
minutos la primera vez (llama a la API de embeddings de Gemini por cada chunk). Una vez creada,
queda persistida en `./chroma_db_micron/` y no es necesario repetir el proceso en ejecuciones
posteriores (puedes comentar esa celda si ya existe la carpeta).

**Nunca subas tu `.env` a un repositorio.** La API key debe vivir solo en esa variable de
entorno local, nunca escrita directamente en el código.

## 4. Justificación del System Prompt

**Rol:** analista experto en los informes públicos de Micron Technology (10-K y proxy
statements).

**Tono:** profesional, preciso y basado en evidencia, como un analista financiero que cita sus
fuentes.

**Limitaciones definidas explícitamente en el prompt:**

1. Responde únicamente con información recuperada de los documentos indexados; no inventa
   cifras ni completa huecos con conocimiento general del modelo.
2. Cuando no encuentra información suficiente, lo dice explícitamente en vez de especular.
3. No da recomendaciones de inversión ni opina sobre si comprar o vender la acción — se limita a
   reportar lo que dicen los documentos.
4. Cita el documento y año fiscal de origen de cada dato relevante.
5. Mantiene coherencia con el historial de la conversación para resolver preguntas de
   seguimiento.

**Por qué estas decisiones:** el dominio financiero es sensible a errores de precisión — una
cifra incorrecta puede confundirse con un dato real. Por eso el prompt prioriza la fidelidad a la
fuente sobre la fluidez, y prohíbe explícitamente la asesoría de inversión, tanto por alcance del
proyecto (informar, no aconsejar) como para evitar que el agente actúe como asesor financiero no
regulado.

El texto completo del system prompt está documentado y versionado en la sección 7 del notebook.

## 5. Extensión opcional (bonus): interfaz de chat con Streamlit

El proyecto incluye `streamlit_app.py`, una interfaz visual de chat que reutiliza la misma base
de conocimiento (ChromaDB) y el mismo agente LangGraph + Gemini definidos en el notebook (mismo
system prompt, misma herramienta de recuperación).

**Requisito previo:** haber ejecutado el notebook al menos hasta la sección 5 (indexación en
ChromaDB), para que exista la colección persistida en `./chroma_db_micron/`.

```bash
streamlit run streamlit_app.py
```

Se abre en `http://localhost:8501`. Cada pestaña/sesión del navegador obtiene su propio
`thread_id`, por lo que la memoria de conversación queda aislada por sesión (igual que la
verificación multi-thread de la sección 12 del notebook).

## 6. Estructura del proyecto

```
Proyecto RAG/
├── README.md                     (este archivo)
├── requirements.txt
├── .env.example
├── Micron_RAG_Agent.ipynb        (notebook principal)
├── streamlit_app.py              (interfaz de chat, extensión opcional)
├── guideline.txt                 (enunciado del proyecto)
└── data/
    └── micron_docs/              (coloca aquí los 4 documentos, ver sección 1)
```

## 7. Entregables cubiertos

- ✅ Base de conocimiento vectorial con ChromaDB + Gemini Embeddings (mínimo 3 docs / ~20 páginas)
- ✅ Agente LangGraph con RAG, Gemini y memoria de conversación
- ✅ Celda de interacción con 5+ preguntas de ejemplo documentadas, incluyendo una que depende de
  una respuesta anterior (demuestra memoria)
- ✅ (Opcional / bonus) Interfaz Streamlit — incluida (`streamlit_app.py`)
