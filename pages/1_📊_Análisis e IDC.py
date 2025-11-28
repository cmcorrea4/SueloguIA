"""
Página de Estadísticos y Calidad de Datos
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import importlib

import utils          
import calidad_datos
import visualizaciones

importlib.reload(utils)
importlib.reload(calidad_datos)
importlib.reload(visualizaciones)

# Agregar path del proyecto
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import VARIABLES_ESTADISTICAS, preparar_dataframe_numerico
from calidad_datos import calcular_indice_calidad_datos, generar_recomendaciones
from visualizaciones import calcular_estadisticos, crear_histogramas, crear_boxplots, crear_matriz_correlacion

st.set_page_config(page_title="Estadísticos", page_icon="📊", layout="wide")

st.title("📊 Análisis Estadístico de Variables")

# Verificar que hay datos cargados
if 'df_original' not in st.session_state or st.session_state.df_original is None:
    st.warning("⚠️ No hay datos cargados. Por favor ve a la página de Inicio para cargar datos.")
    st.stop()

df = st.session_state.df_original

# Encontrar variables disponibles
vars_disponibles = [v for v in VARIABLES_ESTADISTICAS if v in df.columns]
vars_no_disponibles = [v for v in VARIABLES_ESTADISTICAS if v not in df.columns]

if not vars_disponibles:
    st.error("❌ No se encontraron las variables especificadas en el dataset")
    st.info("Las columnas disponibles en tu dataset son:")
    st.write(list(df.columns))
    st.stop()

st.success(f"✅ Se encontraron {len(vars_disponibles)} de {len(VARIABLES_ESTADISTICAS)} variables")

if vars_no_disponibles:
    with st.expander("⚠️ Variables no encontradas en el dataset"):
        for var in vars_no_disponibles:
            st.write(f"- {var}")

# ============================================================================
# CONFIGURACIÓN DE ANÁLISIS
# ============================================================================

st.subheader("🔧 Configuración de análisis")

# Inicializar el key del multiselect si no existe
if 'ms_variables' not in st.session_state:
    st.session_state.ms_variables = []

# Limpiar variables que ya no existen en el dataset actual
st.session_state.ms_variables = [
    v for v in st.session_state.ms_variables 
    if v in vars_disponibles
]

col1, col2 = st.columns([3, 1])

with col1:
    # Multiselect usando key directamente (Streamlit maneja el estado)
    variables_seleccionadas = st.multiselect(
        "Selecciona variables para analizar:",
        options=vars_disponibles,
        key="ms_variables",
        help="Selecciona las variables que deseas incluir en el análisis"
    )
    
    metodo_outliers = st.selectbox(
        "🎯 Método de detección de outliers para ICD:",
        options=['iqr', 'kmeans', 'svm', 'combinado'],
        format_func=lambda x: {
            'iqr': '📊 IQR (Cuartiles) - Tradicional',
            'kmeans': '🎯 K-means - Clustering',
            'svm': '🤖 SVM - One-Class',
            'combinado': '🔄 Combinado (suma de los 3)'
        }[x],
        help="Selecciona el método para calcular la dimensión de Precisión en el ICD"
    )

with col2:
    st.write("")
    st.write("")
    
    # Usar callback para seleccionar todas
    def seleccionar_todas():
        st.session_state.ms_variables = vars_disponibles.copy()
    
    def deseleccionar_todas():
        st.session_state.ms_variables = []
    
    st.button("✅ Seleccionar Todas", use_container_width=True, on_click=seleccionar_todas)
    st.button("❌ Deseleccionar", use_container_width=True, on_click=deseleccionar_todas)
    
    analizar_btn = st.button("📈 Generar Análisis", type="primary", use_container_width=True)

# ============================================================================
# ANÁLISIS
# ============================================================================

if analizar_btn and variables_seleccionadas:
    with st.spinner("📊 Generando análisis..."):
        stats_df = calcular_estadisticos(df, variables_seleccionadas)
        
        if stats_df is not None:
            st.divider()
            
            # SECCIÓN 1: Estadísticos descriptivos
            st.subheader("📋 Estadísticos Descriptivos")
            
            stats_display = stats_df.copy()
            numeric_columns = ['Media', 'Mediana', 'Desv. Std', 'Mínimo', 'Q1 (25%)', 'Q3 (75%)', 'Máximo', 'Rango', 'CV (%)', 'Asimetría', 'Curtosis']
            for col in numeric_columns:
                if col in stats_display.columns:
                    stats_display[col] = stats_display[col].round(3)
            
            st.info("💡 **Detección de Outliers por 3 métodos:** IQR (Cuartiles), K-means (Clustering), SVM (One-Class)")
            
            st.dataframe(stats_display, use_container_width=True, hide_index=True, height=400)
            
            # Análisis de outliers
            if 'Total Outliers' in stats_display.columns:
                total_outliers_sum = stats_display['Total Outliers'].sum()
                if total_outliers_sum > 0:
                    st.warning(f"⚠️ **Total de outliers detectados (suma de 3 métodos): {int(total_outliers_sum)}**")
                    
                    top_outliers = stats_display.nlargest(5, 'Total Outliers')[['Variable', 'Outliers IQR', 'Outliers K-means', 'Outliers SVM', 'Total Outliers']]
                    if len(top_outliers) > 0:
                        st.markdown("**🔍 Variables con más outliers detectados:**")
                        st.dataframe(top_outliers, use_container_width=True, hide_index=True)
                else:
                    st.success("✅ No se detectaron outliers significativos")
            
            csv = stats_display.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Descargar estadísticos como CSV",
                data=csv,
                file_name="estadisticos_variables.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            st.divider()
            
            # SECCIÓN 2: ÍNDICE DE CALIDAD DE DATOS
            st.subheader("🎯 Índice de Calidad de Datos (ICD)")
            
            with st.spinner("Calculando índice de calidad..."):
                resultado_icd = calcular_indice_calidad_datos(
                    df=df,
                    variables_numericas=variables_seleccionadas,
                    columnas_esperadas=VARIABLES_ESTADISTICAS,
                    metodo_outliers=metodo_outliers
                )
            
            # Métrica principal
            st.markdown("### 📊 Calidad General")
            col_metric1, col_metric2, col_metric3 = st.columns([2, 1, 1])
            
            with col_metric1:
                icd_total = resultado_icd['icd_total']
                nivel = resultado_icd['nivel_calidad']
                emoji = resultado_icd['emoji']
                
                st.markdown(f"""
                <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            border-radius: 10px;'>
                    <h1 style='margin: 0; font-size: 60px; color: white;'>{emoji} {icd_total:.1f}</h1>
                    <h3 style='margin: 10px 0 0 0; color: white;'>Calidad {nivel}</h3>
                    <p style='margin: 5px 0 0 0; opacity: 0.9; color: white;'>sobre 100 puntos</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_metric2:
                st.metric("Variables analizadas", len(variables_seleccionadas))
                completitud_pct = resultado_icd['detalles']['completitud']['pct_completo']
                st.metric("Completitud", f"{completitud_pct:.1f}%")
            
            with col_metric3:
                unicidad_pct = resultado_icd['detalles']['unicidad']['pct_registros_unicos']
                st.metric("Unicidad", f"{unicidad_pct:.1f}%")
                precision_pct = resultado_icd['detalles']['precision']['pct_datos_precisos']
                st.metric("Precisión", f"{precision_pct:.1f}%")
            
            st.markdown("---")
            
            # Desglose por dimensiones
            st.markdown("### 📈 Desglose por Dimensiones")
            
            col1, col2, col3 = st.columns(3)
            desglose = resultado_icd['desglose']
            
            with col1:
                st.metric("🔵 Completitud", f"{desglose['Completitud (25pts)']:.1f} / 25")
                st.metric("🟣 Unicidad", f"{desglose['Unicidad (15pts)']:.1f} / 15")
            
            with col2:
                st.metric("🟢 Consistencia", f"{desglose['Consistencia (15pts)']:.1f} / 15")
                st.metric("🟡 Precisión", f"{desglose['Precisión (20pts)']:.1f} / 20")
            
            with col3:
                st.metric("🟠 Variabilidad", f"{desglose['Variabilidad (15pts)']:.1f} / 15")
                st.metric("🔴 Integridad", f"{desglose['Integridad (10pts)']:.1f} / 10")
            
            st.markdown("---")
            
            # Métricas detalladas
            st.markdown("### 🔍 Métricas Detalladas")
            
            tab_comp, tab_uni, tab_prec, tab_var = st.tabs([
                "📊 Completitud", "🔄 Unicidad", "🎯 Precisión (Outliers)", "📉 Variabilidad"
            ])
            
            with tab_comp:
                detalles_comp = resultado_icd['detalles']['completitud']
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Valores totales", f"{detalles_comp['total_valores']:,}")
                col2.metric("Valores nulos", f"{detalles_comp['total_nulos']:,}")
                col3.metric("Completitud", f"{detalles_comp['pct_completo']:.1f}%")
                
                if detalles_comp['columnas_problematicas']:
                    st.warning("⚠️ **Columnas con >50% de valores nulos:**")
                    df_prob = pd.DataFrame([
                        {'Columna': col, '% Nulos': f"{pct:.1f}%"} 
                        for col, pct in detalles_comp['columnas_problematicas'].items()
                    ])
                    st.dataframe(df_prob, use_container_width=True, hide_index=True)
                else:
                    st.success("✅ Todas las columnas tienen menos del 50% de valores nulos")
            
            with tab_uni:
                detalles_uni = resultado_icd['detalles']['unicidad']
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total filas", f"{len(df):,}")
                col2.metric("Filas duplicadas", f"{detalles_uni['filas_duplicadas']:,}")
                col3.metric("Unicidad", f"{detalles_uni['pct_registros_unicos']:.1f}%")
                
                if detalles_uni['filas_duplicadas'] > 0:
                    st.warning(f"⚠️ Se detectaron **{detalles_uni['filas_duplicadas']}** filas duplicadas")
                else:
                    st.success("✅ No hay filas duplicadas")
            
            with tab_prec:
                detalles_prec = resultado_icd['detalles']['precision']
                metodo_usado = detalles_prec.get('metodo_usado', 'iqr')
                
                metodo_nombre = {
                    'iqr': '📊 IQR (Cuartiles)',
                    'kmeans': '🎯 K-means',
                    'svm': '🤖 SVM (One-Class)',
                    'combinado': '🔄 Combinado (3 métodos)'
                }.get(metodo_usado, metodo_usado)
                
                st.info(f"**Método usado para ICD:** {metodo_nombre}")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Datos numéricos", f"{detalles_prec['total_datos_numericos']:,}")
                col2.metric("Outliers detectados", f"{detalles_prec['total_outliers']:,}")
                col3.metric("Precisión", f"{detalles_prec['pct_datos_precisos']:.1f}%")
                
                if detalles_prec['outliers_por_columna']:
                    st.warning(f"⚠️ **Variables con outliers detectados:**")
                    
                    outliers_data = []
                    for col, info in detalles_prec['outliers_por_columna'].items():
                        row = {
                            'Variable': col,
                            'Outliers': info['cantidad'],
                            '% Outliers': f"{info['porcentaje']:.2f}%"
                        }
                        
                        if metodo_usado == 'combinado':
                            row.update({
                                'Outliers IQR': info.get('outliers_iqr', 0),
                                'Outliers K-means': info.get('outliers_kmeans', 0),
                                'Outliers SVM': info.get('outliers_svm', 0),
                            })
                        
                        outliers_data.append(row)
                    
                    df_outliers = pd.DataFrame(outliers_data)
                    st.dataframe(df_outliers, use_container_width=True, hide_index=True)
                    
                    # DataFrame completo con filas de outliers
                    st.markdown("---")
                    st.markdown("#### 📋 Filas Completas con Outliers")
                    
                    df_outliers_full = detalles_prec.get('df_outliers_completo', pd.DataFrame())
                    num_filas_outliers = detalles_prec.get('num_filas_con_outliers', 0)
                    
                    if not df_outliers_full.empty:
                        st.markdown(f"**Total de filas con al menos un outlier: {num_filas_outliers}**")
                        st.dataframe(df_outliers_full, use_container_width=True, height=400)
                        
                        csv_outliers = df_outliers_full.to_csv(index=True).encode('utf-8')
                        st.download_button(
                            label="📥 Descargar filas con outliers (CSV)",
                            data=csv_outliers,
                            file_name=f"outliers_completos_{metodo_usado}.csv",
                            mime="text/csv"
                        )
                else:
                    st.success("✅ No se detectaron outliers significativos")
            
            with tab_var:
                detalles_var = resultado_icd['detalles']['variabilidad']
                
                col1, col2 = st.columns(2)
                col1.metric("CV Promedio", f"{detalles_var['cv_promedio']:.1f}%")
                col2.metric("% Variables CV adecuado", f"{detalles_var['pct_variabilidad_adecuada']:.1f}%")
                
                if detalles_var['cv_por_columna']:
                    st.markdown("**📊 Coeficiente de Variación:**")
                    
                    cv_data = []
                    for col, cv in detalles_var['cv_por_columna'].items():
                        if abs(cv) < 10:
                            categoria, emoji_cv = "Baja", "🟢"
                        elif abs(cv) < 50:
                            categoria, emoji_cv = "Moderada", "🟡"
                        elif abs(cv) < 100:
                            categoria, emoji_cv = "Alta", "🟠"
                        else:
                            categoria, emoji_cv = "Muy Alta", "🔴"
                        
                        cv_data.append({'Variable': col, 'CV (%)': f"{cv:.2f}", 'Categoría': f"{emoji_cv} {categoria}"})
                    
                    df_cv = pd.DataFrame(cv_data)
                    st.dataframe(df_cv, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            
            # Recomendaciones
            st.markdown("### 💡 Recomendaciones")
            recomendaciones = generar_recomendaciones(resultado_icd)
            for rec in recomendaciones:
                st.markdown(rec)
            
            st.markdown("---")
            
            # Interpretación final
            st.markdown("### 📝 Interpretación Final")
            
            if icd_total >= 90:
                st.success(f"**{emoji} Excelente calidad ({icd_total:.1f}/100)** - Datos listos para análisis avanzados.")
            elif icd_total >= 75:
                st.info(f"**{emoji} Buena calidad ({icd_total:.1f}/100)** - Utilizables con limpieza menor.")
            elif icd_total >= 60:
                st.warning(f"**{emoji} Calidad aceptable ({icd_total:.1f}/100)** - Requiere limpieza antes de análisis.")
            elif icd_total >= 40:
                st.warning(f"**{emoji} Calidad baja ({icd_total:.1f}/100)** - Limpieza profunda requerida.")
            else:
                st.error(f"**{emoji} Calidad crítica ({icd_total:.1f}/100)** - Revisar proceso de captura.")
            
            st.divider()
            
            # SECCIÓN 3: Visualizaciones
            st.subheader("📊 Visualizaciones")
            
            viz_tab1, viz_tab2, viz_tab3 = st.tabs(["📊 Histogramas", "📦 Boxplots", "🔥 Correlaciones"])
            
            with viz_tab1:
                st.markdown("#### Distribución de Variables")
                fig_hist = crear_histogramas(df, variables_seleccionadas)
                if fig_hist:
                    st.plotly_chart(fig_hist, use_container_width=True)
                else:
                    st.warning("No se pudieron generar histogramas")
            
            with viz_tab2:
                st.markdown("#### Detección de Valores Atípicos")
                fig_box = crear_boxplots(df, variables_seleccionadas)
                if fig_box:
                    st.plotly_chart(fig_box, use_container_width=True)
                else:
                    st.warning("No se pudieron generar boxplots")
            
            with viz_tab3:
                st.markdown("#### Relaciones entre Variables")
                if len(variables_seleccionadas) >= 2:
                    fig_corr = crear_matriz_correlacion(df, variables_seleccionadas)
                    if fig_corr:
                        st.plotly_chart(fig_corr, use_container_width=True)
                        
                        df_numeric = preparar_dataframe_numerico(df, variables_seleccionadas)
                        
                        if len(df_numeric.columns) >= 2:
                            corr_matrix = df_numeric.corr()
                            
                            correlaciones = []
                            for i in range(len(corr_matrix.columns)):
                                for j in range(i+1, len(corr_matrix.columns)):
                                    corr_val = corr_matrix.iloc[i, j]
                                    if not pd.isna(corr_val):
                                        correlaciones.append({
                                            'Variable 1': corr_matrix.columns[i],
                                            'Variable 2': corr_matrix.columns[j],
                                            'Correlación': corr_val
                                        })
                            
                            if correlaciones:
                                df_corr = pd.DataFrame(correlaciones)
                                df_corr['Correlacion_abs'] = df_corr['Correlación'].abs()
                                df_corr = df_corr.sort_values('Correlacion_abs', ascending=False)
                                
                                st.markdown("##### Top 10 Correlaciones más Fuertes")
                                st.dataframe(
                                    df_corr.head(10)[['Variable 1', 'Variable 2', 'Correlación']].round(3),
                                    use_container_width=True, hide_index=True
                                )
                    else:
                        st.warning("No se pudo generar la matriz de correlación")
                else:
                    st.info("Selecciona al menos 2 variables para ver correlaciones")
        
        else:
            st.error("❌ No se pudieron calcular estadísticos. Verifica que las variables sean numéricas.")

elif analizar_btn:
    st.warning("⚠️ Por favor selecciona al menos una variable para analizar")
