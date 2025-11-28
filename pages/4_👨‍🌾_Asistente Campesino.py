"""
  Página de RAG con PDF - Generación Aumentada por Recuperación
Usa el archivo recomendaciones.pdf de la raíz del proyecto
Compatible con LangChain 0.2.x y OpenAI 1.x
Incluye Speech-to-Text (Whisper) y Text-to-Speech (OpenAI TTS)
"""
import streamlit as st
import os
import tempfile
import base64

st.set_page_config(page_title="RAG PDF", page_icon="📄", layout="wide")

st.title("👨‍🌾 Asistente Campesino")
st.markdown("Haz preguntas por texto o voz y te responderá con base en mi conocimiento.")

# ==========================
# CONFIGURACIÓN


RUTA_PDF = "recomendaciones.pdf"

# ============================================================
# CONFIGURACIÓN DE API KEY DESDE SECRETS


try:
    openai_api_key = st.secrets["settings"]["key"]
    ia_disponible = True
except:
    ia_disponible = False

if ia_disponible:
    os.environ["OPENAI_API_KEY"] = openai_api_key



with st.sidebar:
    st.header("⚙️ Configuración RAG")
    
    if ia_disponible:
        st.success("🔑 Credenciales cargadas correctamente")
    else:
        st.error("❌ Error: No se encontraron las credenciales necesarias")
        st.info("💡 Configura las credenciales en los secrets de la aplicación")
    
    # Selección de modelo
    model_name = st.selectbox(
        "🤖 Modelo:",
        ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo", "gpt-4-turbo"],
        index=0,
        help="Modelo usado para generar respuestas"
    )
    
    st.divider()
    
    # Configuración de voz
    st.subheader("🎙️ Configuración de Voz")
    
    enable_tts = st.checkbox("🔊 Habilitar respuesta por voz", value=False)
    
    if enable_tts:
        tts_voice = st.selectbox(
            "Voz:",
            ["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
            index=4,  # nova por defecto
            help="Selecciona la voz para las respuestas"
        )
        
        tts_speed = st.slider(
            "Velocidad:",
            min_value=0.5,
            max_value=2.0,
            value=1.0,
            step=0.1
        )
        
        #st.caption("⚠️ TTS tiene costo: $0.015/1K caracteres")
    
    st.divider()
    
    # Configuración avanzada
    with st.expander("🔧 Configuración avanzada"):
        chunk_size = st.slider("Tamaño de chunks:", 200, 1500, 500, 50)
        chunk_overlap = st.slider("Overlap de chunks:", 0, 200, 50, 10)
        k_results = st.slider("Documentos a recuperar:", 1, 10, 4)
    
    st.divider()

    with st.expander("ℹ️ ¿Cómo funciona?"):
        st.markdown("""
        **RAG (Retrieval-Augmented Generation):**
        
        1. **Indexación**: El documento se divide en fragmentos con embeddings vectoriales
        2. **Búsqueda**: Tu pregunta se busca entre los fragmentos más similares
        3. **Generación**: GPT genera una respuesta basada en los fragmentos
        
        **Funciones de voz:**
        - 🎤 **Speech-to-Text**: Whisper ($0.006/min)
        - 🔊 **Text-to-Speech**: OpenAI TTS ($0.015/1K chars)
        """)


# Verificar que existe el PDF
if not os.path.exists(RUTA_PDF):
    st.error(f"❌ No se encontró el archivo `{RUTA_PDF}` en la raíz del proyecto.")
    st.info("Asegúrate de que el archivo PDF esté en la misma carpeta que la aplicación.")
    st.stop()

# Verificar disponibilidad de IA
if not ia_disponible:
    st.error("❌ Asistente no disponible: credenciales no configuradas")
    st.info("💡 Contacta al administrador para configurar las credenciales del sistema")
    st.stop()


# =====
# IMPORTAR DEPENDENCIAS
# ============

try:
    from pypdf import PdfReader
    from langchain.text_splitter import CharacterTextSplitter
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain_community.vectorstores import FAISS
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
    from openai import OpenAI
except ImportError as e:
    st.error(f"❌ Error al importar dependencias: {str(e)}")
    st.info("""
    Asegúrate de tener instalados:
    - pypdf
    - langchain
    - langchain-openai
    - langchain-community
    - faiss-cpu
    - openai
    """)
    st.stop()

# Intentar importar audio_recorder_streamlit
try:
    from audio_recorder_streamlit import audio_recorder
    AUDIO_RECORDER_AVAILABLE = True
except ImportError:
    AUDIO_RECORDER_AVAILABLE = False

# Cliente OpenAI para STT y TTS
client = OpenAI(api_key=openai_api_key)

# ==========================
# FUNCIONES DE VOZ
# ================

def speech_to_text(audio_bytes):
    """Convierte audio a texto usando Whisper de OpenAI."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_file_path = tmp_file.name
        
        with open(tmp_file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="es"
            )
        
        os.unlink(tmp_file_path)
        return transcript.text
    except Exception as e:
        st.error(f"❌ Error en transcripción: {str(e)}")
        return None


def text_to_speech(text, voice="nova", speed=1.0):
    """Convierte texto a audio usando OpenAI TTS."""
    try:
        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text,
            speed=speed
        )
        return response.content
    except Exception as e:
        st.error(f"❌ Error en síntesis de voz: {str(e)}")
        return None


def display_audio_player(audio_bytes):
    """Muestra un reproductor de audio."""
    b64 = base64.b64encode(audio_bytes).decode()
    audio_html = f"""
        <audio controls>
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            Tu navegador no soporta el elemento de audio.
        </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)



# FUNCIONES RAG
# 

@st.cache_data
def extract_text_from_pdf(ruta):
    """Extrae texto del PDF."""
    try:
        pdf_reader = PdfReader(ruta)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        st.error(f"Error extrayendo texto: {e}")
        return None


@st.cache_data
def create_chunks(text, chunk_size=500, chunk_overlap=50):
    """Divide el texto en chunks."""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return text_splitter.split_text(text)


@st.cache_resource
def create_vector_store(_chunks, api_key, chunk_config):
    """Crea el vector store con FAISS."""
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vector_store = FAISS.from_texts(list(_chunks), embeddings)
    return vector_store


def get_qa_chain(vector_store, model_name, api_key, k=4):
    """Crea la cadena de QA."""
    llm = ChatOpenAI(
        model=model_name,
        temperature=0.1,
        openai_api_key=api_key
    )
    
    prompt_template = """Eres un asistente experto que responde preguntas basándose en el documento proporcionado.
Usa el siguiente contexto para responder la pregunta del usuario.
Si no encuentras la respuesta en el contexto, di claramente que no tienes suficiente información en el documento.
Responde siempre en español de manera clara, precisa y concisa.

Contexto:
{context}

Pregunta: {question}

Respuesta:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": k}),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )
    
    return qa_chain


# =======================================
# CARGAR Y PROCESAR PDF
# ===============

text = extract_text_from_pdf(RUTA_PDF)

if not text:
    st.error("❌ No se pudo extraer texto del PDF")
    st.stop()

chunks = create_chunks(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

try:
    chunk_config = f"{chunk_size}_{chunk_overlap}"
    vector_store = create_vector_store(tuple(chunks), openai_api_key, chunk_config)
    st.success(f"✅ Asistente listo: {len(chunks)} secciones indexadas | 🎤 Voz disponible")
except Exception as e:
    st.error(f"❌ Error creando embeddings: {str(e)}")
    st.info("Verifica que la API key sea válida y tenga créditos disponibles.")
    st.stop()


# INTERFAZ DE CONSULTA


if 'rag_chat_history' not in st.session_state:
    st.session_state.rag_chat_history = []

if 'voice_question' not in st.session_state:
    st.session_state.voice_question = ""

st.divider()

# Ejemplos de preguntas
st.subheader("💡 Ejemplos de preguntas:")
col1, col2 = st.columns(2)
    
with col1:
    examples1 = [
        "¿Qué significa un alto valor de Aluminio?",
        "¿Qué hacer si tengo un pH de agua bajo?",
        "¿Qué significa la métrica de Completitud en el Índice de Calidad de Datos?"
    ]
    for example in examples1:
        st.write(f"• {example}")
    
with col2:
    examples2 = [
        "¿Cómo mejoro la materia orgánica de mi finca?",
        "¿Qué diferencia hay entre asimetría positiva y negativa?",
        "¿Cómo se interpreta un valor muy alto de acidez KCl?"
    ]
    for example in examples2:
        st.write(f"• {example}")
    
st.divider()

def limpiar_historial_rag():
    st.session_state.rag_chat_history = []

# =================================================
# ENTRADA: TEXTO O VOZ
# ============================================================================

st.subheader("🔍 Haz tu consulta")

tab_texto, tab_voz = st.tabs(["⌨️ Escribir", "🎤 Grabar voz"])

with tab_texto:
    with st.form(key="rag_question_form", clear_on_submit=True):
        user_question = st.text_area(
            "Escribe tu pregunta:",
            placeholder="Ej: ¿Cuáles son las principales recomendaciones?",
            height=100
        )
        col1, col2, col3 = st.columns([1, 1, 3])
        with col1:
            ask_button = st.form_submit_button("🚀 Preguntar", type="primary")

with tab_voz:
    #st.caption("⚠️ Costo: $0.006 por minuto de audio")
    
    if AUDIO_RECORDER_AVAILABLE:
        st.info("🎤 Clic en el micrófono para grabar")
        
        audio_bytes = audio_recorder(
            text="",
            recording_color="#e74c3c",
            neutral_color="#3498db",
            icon_name="microphone",
            icon_size="2x",
            pause_threshold=2.0,
            sample_rate=16000
        )
        
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            
            if st.button("📝 Transcribir y preguntar", type="primary"):
                with st.spinner("🔄 Transcribiendo..."):
                    transcribed_text = speech_to_text(audio_bytes)
                    if transcribed_text:
                        st.success(f"📝 *{transcribed_text}*")
                        st.session_state.voice_question = transcribed_text
    else:
        st.warning("Instala: `pip install audio-recorder-streamlit`")
        
        st.markdown("**Alternativa: Subir audio**")
        uploaded_audio = st.file_uploader(
            "Archivo de audio (WAV, MP3, M4A):",
            type=["wav", "mp3", "m4a", "ogg", "webm"]
        )
        
        if uploaded_audio:
            st.audio(uploaded_audio)
            
            if st.button("📝 Transcribir y preguntar", type="primary"):
                with st.spinner("🔄 Transcribiendo..."):
                    audio_bytes = uploaded_audio.read()
                    suffix = f".{uploaded_audio.name.split('.')[-1]}"
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                        tmp_file.write(audio_bytes)
                        tmp_file_path = tmp_file.name
                    
                    try:
                        with open(tmp_file_path, "rb") as audio_file:
                            transcript = client.audio.transcriptions.create(
                                model="whisper-1",
                                file=audio_file,
                                language="es"
                            )
                        os.unlink(tmp_file_path)
                        
                        if transcript.text:
                            st.success(f"📝 *{transcript.text}*")
                            st.session_state.voice_question = transcript.text
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")

st.button("🗑️ Limpiar historial", on_click=limpiar_historial_rag)

# ============================================================================
# PROCESAR PREGUNTA
# ============================================================================

question_to_process = None

if ask_button and user_question.strip():
    question_to_process = user_question.strip()
elif st.session_state.voice_question:
    question_to_process = st.session_state.voice_question
    st.session_state.voice_question = ""

if question_to_process:
    with st.spinner("🔍 Buscando en el documento..."):
        try:
            qa_chain = get_qa_chain(
                vector_store,
                model_name,
                openai_api_key,
                k=k_results
            )
            
            result = qa_chain.invoke({"query": question_to_process})
            answer = result["result"]
            
            chat_entry = {
                "question": question_to_process,
                "answer": answer,
                "sources": [doc.page_content[:300] + "..." for doc in result.get("source_documents", [])],
                "audio": None
            }
            
            # Generar audio si TTS está habilitado
            if enable_tts:
                with st.spinner("🔊 Generando audio..."):
                    audio_response = text_to_speech(answer, voice=tts_voice, speed=tts_speed)
                    if audio_response:
                        chat_entry["audio"] = audio_response
            
            st.session_state.rag_chat_history.append(chat_entry)
            
            st.markdown("### 💬 Respuesta:")
            st.write(answer)
            
            if chat_entry["audio"]:
                st.markdown("**🔊 Escuchar:**")
                display_audio_player(chat_entry["audio"])
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            import traceback
            with st.expander("🔧 Detalles"):
                st.code(traceback.format_exc())

# ============================================================================
# HISTORIAL
# ============================================================================

if st.session_state.rag_chat_history:
    st.divider()
    st.subheader("💬 Historial")
    
    for i, chat in enumerate(reversed(st.session_state.rag_chat_history)):
        question_preview = chat['question'][:60] + "..." if len(chat['question']) > 60 else chat['question']
        
        with st.expander(f"❓ {question_preview}", expanded=(i == 0)):
            st.markdown("**Pregunta:**")
            st.write(chat['question'])
            
            st.markdown("**Respuesta:**")
            st.write(chat['answer'])
            
            if chat.get('audio'):
                st.markdown("**🔊 Audio:**")
                display_audio_player(chat['audio'])

# ============================================================================
### INFORMACIÓN ADICIONAL
# ============================================================================


