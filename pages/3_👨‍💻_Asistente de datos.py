"""
Página de Análisis con IA
"""
import streamlit as st
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Análisis IA", page_icon="🤖", layout="wide")
 
st.title("👨‍💻 Asistente de datos IA")
st.markdown("Hazme preguntas sobre los datos cargados, soy experto en python (pandas).")

# Verificar que hay datos cargados
if 'df' not in st.session_state or st.session_state.df is None:
    st.warning("⚠️ No hay datos cargados. Por favor ve a la página de Inicio para cargar datos.")
    st.stop()

# ============================================================================
# CONFIGURACIÓN DE API KEY DESDE SECRETS
# ============================================================================

# Verificar si existe la API key en secrets
try:
    openai_api_key = st.secrets["settings"]["key"]
    ia_disponible = True
except:
    ia_disponible = False

if ia_disponible:
    os.environ["OPENAI_API_KEY"] = openai_api_key

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.header("⚙️ Configuración IA")
    
    if ia_disponible:
        st.success("🔑 Credenciales cargadas correctamente")
    else:
        st.error("❌ Error: No se encontraron las credenciales necesarias")
        st.info("💡 Configura las credenciales en los secrets de la aplicación")
    
    # Selección de modelo
    model_name = st.selectbox(
        "🤖 Modelo OpenAI:",
        ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
        index=0
    )
    st.session_state.model_name = model_name
    
    st.divider()

    # Información adicional
    with st.expander("ℹ️ Consejos para mejores resultados"):
        st.markdown("""
        **Consejos para hacer preguntas efectivas:**
        
        1. **Sé específico**: En lugar de "muéstrame estadísticas", pregunta "¿Cuál es la media y desviación estándar de ph_agua_suelo?"
        
        2. **Usa nombres exactos de columnas**: Verifica los nombres de las columnas en la vista previa de datos.
        
        3. **Preguntas complejas**: El agente puede hacer análisis complejos como correlaciones, agrupaciones y filtros.
        
        4. **Iteración**: Puedes hacer preguntas de seguimiento basándote en respuestas anteriores.
        
        **Limitaciones:**
        - El agente trabaja con los datos en memoria, no puede guardar cambios permanentes.
        - Para análisis muy complejos, considera dividir la pregunta en pasos más pequeños.
        """)

# Verificar disponibilidad de IA
if not ia_disponible:
    st.error("❌ Análisis inteligente no disponible: credenciales no configuradas")
    st.info("💡 Contacta al administrador para configurar las credenciales del sistema")
    st.stop()

# Temperatura fija (no visible para el usuario)
temperature = 0.1

# Importar LangChain
try:
    from langchain.agents.agent_types import AgentType
    from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
    from langchain_openai import ChatOpenAI
except ImportError as e:
    st.error(f"❌ Error al importar dependencias: {str(e)}")
    st.info("Asegúrate de tener instalados: langchain, langchain-experimental, langchain-openai")
    st.stop()


def create_agent():
    """Crear agente de pandas"""
    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        openai_api_key=openai_api_key
    )
    return create_pandas_dataframe_agent(
        llm,
        st.session_state.df,
        verbose=False,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        allow_dangerous_code=True
    )


# Inicializar historial de chat
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Verificar si necesitamos recrear el agente
agent_config_key = f"{model_name}_{temperature}_{id(st.session_state.df)}"

if 'agent_config_key' not in st.session_state:
    st.session_state.agent_config_key = None

if st.session_state.get('agent') is None or st.session_state.agent_config_key != agent_config_key:
    try:
        with st.spinner("🔄 Inicializando agente IA..."):
            st.session_state.agent = create_agent()
            st.session_state.agent_config_key = agent_config_key
    except Exception as e:
        st.error(f"❌ Error al inicializar el agente: {str(e)}")
        st.info("Verifica que la API key de OpenAI sea válida y tenga créditos disponibles.")
        st.session_state.agent = None
        st.stop()

if st.session_state.agent is not None:
    st.success("🎯 Agente IA inicializado correctamente")
    
    # Ejemplos de preguntas
    st.subheader("💡 Ejemplos de preguntas que puedes hacer:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        examples1 = [
            "Muestra un resumen estadístico de los datos de materia orgánica",
            "¿Cuantos datos nulos tiene la conductividad eléctrica?",
            "¿Cual es la media de ph en los cultivos de café?"
        ]
        for example in examples1:
            st.write(f"• {example}")
    
    with col2:
        examples2 = [
            "¿Cuál es la correlación mayor entre las variables numéricas?",
            "¿Cuáles son los valores únicos de [columna]?",
            "¿Qué cultivos se dan en el muncipio de pasca?"
        ]
        for example in examples2:
            st.write(f"• {example}")
    
    st.divider()
    
    # Función para limpiar historial
    def limpiar_historial():
        st.session_state.chat_history = []
    
    st.subheader("🔍 Haz tu consulta")
    with st.form(key="question_form", clear_on_submit=True):
        user_question = st.text_area(
            "Escribe tu pregunta:",
            placeholder="Ej: ¿Cuál es la correlación entre las variables numéricas?",
            key="user_input_form"
        )
        
        col1, col2, col3 = st.columns([1, 1, 3])
        with col1:
            ask_button = st.form_submit_button("🚀 Preguntar", type="primary")
        with col2:
            pass
        with col3:
            pass
    
    # Botón de limpiar historial usando callback (sin rerun manual)
    st.button("🗑️ Limpiar historial", on_click=limpiar_historial)
    
    if ask_button and user_question:
        with st.spinner("🔄 El agente está analizando tus datos..."):
            try:
                response = st.session_state.agent.invoke({"input": user_question})
                
                st.session_state.chat_history.append({
                    "question": user_question,
                    "answer": response["output"]
                })
                
            except Exception as e:
                st.error(f"❌ Error al procesar la pregunta: {str(e)}")
                st.info("💡 Intenta reformular tu pregunta o verifica que la columna existe en el dataset.")
    
    # Mostrar historial de conversación
    if st.session_state.chat_history:
        st.divider()
        st.subheader("💬 Historial de conversación")
        
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            question_preview = chat['question'][:60] + "..." if len(chat['question']) > 60 else chat['question']
            with st.expander(f"❓ {question_preview}", expanded=(i==0)):
                st.markdown("**Pregunta:**")
                st.write(chat['question'])
                st.markdown("**Respuesta:**")
                st.write(chat['answer'])
                st.divider()

else:
    st.error("❌ No se pudo inicializar el agente. Verifica la configuración.")
