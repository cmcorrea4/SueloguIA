"""
Página Principal - Carga de los datos
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
from sodapy import Socrata

# Importar módulos locales
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import asignar_tipos_datos, DataCleaner

# Configuración de la página
st.set_page_config(
    page_title="Suelos Agrosavia",
    page_icon="📥",
    layout="wide"
)

# ========
# INICIALIZACIÓN DE SESSION STATE
# =====================================

if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_original' not in st.session_state:
    st.session_state.df_original = None
if 'data_source' not in st.session_state:
    st.session_state.data_source = None
if 'cleaning_report' not in st.session_state:
    st.session_state.cleaning_report = None
if 'variables_seleccionadas' not in st.session_state:
    st.session_state.variables_seleccionadas = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'agent_config_key' not in st.session_state:
    st.session_state.agent_config_key = None


def load_data_from_socrata(domain: str, dataset_id: str, limit: int, app_token: str = None) -> tuple:
    """Carga datos desde Socrata API"""
    try:
        client = Socrata(domain, app_token, timeout=30)
        results = client.get(dataset_id, limit=limit)
        df = pd.DataFrame.from_records(results)
        return df, None
    except Exception as e:
        return None, str(e)


# =============================================
# INTERFAZ PRINCIPAL
# ===========================================================================

st.title("📊 SueloGuIA")
st.markdown("**Herramienta para cálculo del índice de calidad de datos (ICD) y análisis con asistentes conversacionales**")

# Sidebar para información del dataset
with st.sidebar:
    st.header("📋 Información")
    
    if st.session_state.df is not None:
        st.subheader("📋 Dataset")
        st.info(f"📏 {st.session_state.df.shape[0]} filas × {st.session_state.df.shape[1]} columnas")
        
        if st.session_state.data_source:
            st.caption(f"🔗 Fuente: {st.session_state.data_source}")
        
        with st.expander("ℹ️ Tipos de datos"):
            numeric_cols = st.session_state.df.select_dtypes(include=[np.number]).columns
            text_cols = st.session_state.df.select_dtypes(include=['object']).columns
            
            st.write(f"**Numéricas:** {len(numeric_cols)}")
            st.write(f"**Texto:** {len(text_cols)}")
            
            null_total = st.session_state.df.isnull().sum().sum()
            st.write(f"**Valores nulos:** {null_total}")
        
        if st.button("🗑️ Limpiar todo", use_container_width=True):
            st.session_state.df = None
            st.session_state.df_original = None
            st.session_state.agent = None
            st.session_state.agent_config_key = None
            st.session_state.data_source = None
            st.session_state.cleaning_report = None
            st.session_state.variables_seleccionadas = []
            st.session_state.chat_history = []
            st.rerun()


# CARGAR DATOS


st.header("📁 Cargar Datos")

subtab1, subtab2 = st.tabs(["🌐 API Socrata", "📁 Archivo CSV/Excel"])

with subtab1:
    st.markdown("### 🌐 Configuración de Socrata")
    
    col1, col2 = st.columns(2)
    
    with col1:
        domain = st.text_input("🌍 Dominio:", value="www.datos.gov.co")
    
    with col2:
        dataset_id = st.text_input("🆔 Dataset ID:", value="ch4u-f3i5")
    
    col3, col4 = st.columns(2)
    
    with col3:
        limit = st.number_input("📊 Límite de registros:", min_value=100, max_value=50000, value=2000, step=500)
    
    with col4:
        app_token = st.text_input("🔑 App Token (opcional):", type="password")
    
    if st.button("🔄 Cargar desde API", use_container_width=True, type="primary"):
        if domain and dataset_id:
            with st.spinner("📥 Cargando datos desde API..."):
                df_raw, error = load_data_from_socrata(domain, dataset_id, limit, app_token if app_token else None)
                
                if df_raw is not None:
                    df_raw = asignar_tipos_datos(df_raw)
                    st.session_state.df_original = df_raw.copy()
                    st.session_state.df = df_raw.copy()
                    st.session_state.data_source = f"API: {domain}/{dataset_id}"
                    
                    st.success(f"✅ Datos cargados exitosamente desde API")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📏 Filas", st.session_state.df.shape[0])
                    with col2:
                        st.metric("📊 Columnas", st.session_state.df.shape[1])
                    with col3:
                        st.metric("💾 Tamaño", f"{st.session_state.df.memory_usage(deep=True).sum() / 1024:.1f} KB")
                    
                    with st.expander("👀 Vista previa de datos", expanded=True):
                        st.dataframe(st.session_state.df.head(10), use_container_width=True)
                else:
                    st.error(f"❌ Error al cargar datos: {error}")
        else:
            st.warning("⚠️ Por favor completa los campos requeridos")

with subtab2:
    uploaded_file = st.file_uploader(
        "Selecciona un archivo CSV o Excel:",
        type=['csv', 'xlsx', 'xls'],
        help="Formatos soportados: CSV, Excel (.xlsx, .xls)"
    )
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df_raw = pd.read_csv(uploaded_file, on_bad_lines='skip')
            else:
                df_raw = pd.read_excel(uploaded_file)
            
            df_raw = asignar_tipos_datos(df_raw)
            st.session_state.df_original = df_raw.copy()
            
            if st.checkbox("🧹 Aplicar limpieza automática", value=False, key="clean_file"):
                with st.spinner("🧹 Limpiando datos..."):
                    df_clean, report = DataCleaner.clean_dataframe(df_raw)
                    st.session_state.df = df_clean
                    st.session_state.cleaning_report = report
                    st.session_state.data_source = f"Archivo: {uploaded_file.name}"
                
                st.success("✅ Limpieza completada")
                with st.expander("📋 Ver reporte de limpieza", expanded=True):
                    for item in report:
                        st.write(item)
            else:
                st.session_state.df = df_raw.copy()
                st.session_state.data_source = f"Archivo: {uploaded_file.name} (sin limpiar)"
                st.info("📊 Usando datos originales (sin limpieza)")
            
            st.success(f"✅ Archivo cargado exitosamente: {uploaded_file.name}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📏 Filas", st.session_state.df.shape[0])
            with col2:
                st.metric("📊 Columnas", st.session_state.df.shape[1])
            with col3:
                st.metric("💾 Tamaño", f"{st.session_state.df.memory_usage(deep=True).sum() / 1024:.1f} KB")
            
            with st.expander("👀 Vista previa de datos", expanded=False):
                st.dataframe(st.session_state.df.head(10), use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Error al cargar el archivo: {str(e)}")

# Información inicial
if st.session_state.df is None:
    st.divider()
    st.info("""
    ### 🚀 Cómo usar:
    
    **Opción 1: API Socrata**
    1. 🌍 Ingresa dominio y Dataset ID
    2. 🔄 Carga desde API
    
    **Opción 2: Archivo CSV/Excel**
    1. 📁 Sube un archivo CSV o Excel
    2. 🧹 Opcional: Aplica limpieza automática
    
    **Luego navega a las otras páginas:**
    - 📊 **Estadísticos**: Consulta estadísticos y calidad de datos
    - 🤖 **Análisis IA**: Haz preguntas en lenguaje natural
    """)

# Footer
st.divider()
st.caption("📊 Agente datos suelos Agrosavia Powered by SUME | Índice de Calidad de Datos y asistencia IA")

