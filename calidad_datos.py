"""
Funciones para calcular el Índice de Calidad de Datos (ICD)
"""
import pandas as pd
import numpy as np
from typing import Dict, List
from sklearn.cluster import KMeans
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

from utils import preparar_dataframe_numerico


def calcular_completitud(df: pd.DataFrame, variables: List[str] = None) -> Dict:
    """Calcula la completitud de los datos"""
    df_work = preparar_dataframe_numerico(df, variables)
    
    if df_work.empty:
        return {
            'score': 0, 'pct_completo': 0, 'columnas_problematicas': {},
            'total_nulos': 0, 'total_valores': 0
        }
    
    total_cells = df_work.shape[0] * df_work.shape[1]
    non_null_cells = df_work.count().sum()
    pct_completo = (non_null_cells / total_cells) * 100 if total_cells > 0 else 0
    
    null_pct_por_col = (df_work.isnull().sum() / len(df_work)) * 100
    columnas_problematicas = null_pct_por_col[null_pct_por_col > 50].to_dict()
    
    score = (pct_completo / 100) * 25
    penalizacion = len(columnas_problematicas) * 2
    score = max(0, score - penalizacion)
    
    return {
        'score': score, 'pct_completo': pct_completo,
        'columnas_problematicas': columnas_problematicas,
        'total_nulos': int(df_work.isnull().sum().sum()),
        'total_valores': total_cells
    }


def calcular_unicidad(df: pd.DataFrame, variables: List[str] = None) -> Dict:
    """Calcula la unicidad de los datos"""
    df_work = preparar_dataframe_numerico(df, variables)
    
    if df_work.empty:
        return {
            'score': 15, 'pct_registros_unicos': 100,
            'filas_duplicadas': 0, 'columnas_con_duplicados_altos': {}
        }
    
    filas_duplicadas = df_work.duplicated().sum()
    total_filas = len(df_work)
    pct_registros_unicos = ((total_filas - filas_duplicadas) / total_filas) * 100 if total_filas > 0 else 100
    
    columnas_con_duplicados_altos = {}
    for col in df_work.columns:
        valores_unicos = df_work[col].nunique()
        valores_totales = df_work[col].notna().sum()
        
        if valores_totales > 0:
            pct_unicos = (valores_unicos / valores_totales) * 100
            if pct_unicos < 20:
                columnas_con_duplicados_altos[col] = {
                    'valores_unicos': valores_unicos,
                    'pct_unicidad': pct_unicos
                }
    
    score_registros = (pct_registros_unicos / 100) * 10
    score = score_registros + 5
    
    return {
        'score': score, 'pct_registros_unicos': pct_registros_unicos,
        'filas_duplicadas': int(filas_duplicadas),
        'columnas_con_duplicados_altos': columnas_con_duplicados_altos
    }


def calcular_consistencia(df: pd.DataFrame, variables: List[str] = None) -> Dict:
    """Calcula la consistencia de los datos"""
    if variables is not None and len(variables) > 0:
        vars_disponibles = [v for v in variables if v in df.columns]
        if vars_disponibles:
            df_original = df[vars_disponibles].copy()
        else:
            return {'score': 15, 'pct_consistente': 100, 'columnas_tipo_mixto': {}, 'valores_inconsistentes': 0}
    else:
        df_original = df.copy()
    
    df_convertido = preparar_dataframe_numerico(df, variables)
    
    if df_convertido.empty:
        return {'score': 15, 'pct_consistente': 100, 'columnas_tipo_mixto': {}, 'valores_inconsistentes': 0}
    
    inconsistencias = 0
    columnas_tipo_mixto = {}
    
    for col in df_convertido.columns:
        nulos_original = df_original[col].isnull().sum()
        nulos_convertido = df_convertido[col].isnull().sum()
        valores_perdidos = nulos_convertido - nulos_original
        
        if valores_perdidos > 0:
            columnas_tipo_mixto[col] = int(valores_perdidos)
            inconsistencias += valores_perdidos
    
    total_valores = df_original.shape[0] * df_original.shape[1]
    pct_consistente = ((total_valores - inconsistencias) / total_valores) * 100 if total_valores > 0 else 100
    score = (pct_consistente / 100) * 15
    
    return {
        'score': score, 'pct_consistente': pct_consistente,
        'columnas_tipo_mixto': columnas_tipo_mixto,
        'valores_inconsistentes': int(inconsistencias)
    }


def calcular_precision_outliers(df: pd.DataFrame, variables_numericas: List[str] = None, metodo: str = 'iqr') -> Dict:
    """Calcula la precisión basada en detección de outliers"""
    df_numeric = preparar_dataframe_numerico(df, variables_numericas)
    
    if df_numeric.empty:
        return {
            'score': 20, 'pct_datos_precisos': 100, 'outliers_por_columna': {},
            'total_outliers': 0, 'total_datos_numericos': 0, 'metodo_usado': metodo,
            'df_outliers_completo': pd.DataFrame(), 'num_filas_con_outliers': 0
        }
    
    outliers_por_columna = {}
    total_outliers = 0
    total_datos = 0
    todos_indices_outliers = set()
    
    # Diccionario para guardar qué índices son outliers en cada columna
    outliers_por_indice = {}  # {indice: {columna: valor, ...}}
    
    for col in df_numeric.columns:
        data = df_numeric[col].dropna()
        if len(data) == 0:
            continue
            
        n_outliers = 0
        outlier_info = {'variable': col}
        
        n_outliers_iqr = 0
        n_outliers_kmeans = 0
        n_outliers_svm = 0
        indices_outliers_iqr = set()
        indices_outliers_kmeans = set()
        indices_outliers_svm = set()
        
        # MÉTODO IQR
        if metodo in ['iqr', 'combinado']:
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_iqr_mask = (data < lower_bound) | (data > upper_bound)
            outliers_iqr = data[outliers_iqr_mask]
            n_outliers_iqr = len(outliers_iqr)
            indices_outliers_iqr = set(data[outliers_iqr_mask].index)
            
            if metodo == 'iqr':
                n_outliers = n_outliers_iqr
                if n_outliers > 0:
                    outlier_info.update({
                        'cantidad': n_outliers,
                        'porcentaje': (n_outliers / len(data)) * 100,
                        'limite_inferior': lower_bound,
                        'limite_superior': upper_bound,
                        'min_outlier': outliers_iqr.min(),
                        'max_outlier': outliers_iqr.max()
                    })
        
        # MÉTODO K-MEANS
        if metodo in ['kmeans', 'combinado'] and len(data) >= 3:
            try:
                data_reshaped = data.values.reshape(-1, 1)
                scaler = StandardScaler()
                data_scaled = scaler.fit_transform(data_reshaped)
                
                kmeans = KMeans(n_clusters=min(3, len(data)), random_state=42, n_init=10)
                kmeans.fit(data_scaled)
                distances = np.min(kmeans.transform(data_scaled), axis=1)
                threshold = np.percentile(distances, 90)
                
                outliers_kmeans_mask = distances > threshold
                n_outliers_kmeans = outliers_kmeans_mask.sum()
                indices_outliers_kmeans = set(data.index[outliers_kmeans_mask])
                
                if metodo == 'kmeans':
                    n_outliers = n_outliers_kmeans
                    if n_outliers > 0:
                        outliers_kmeans_data = data[outliers_kmeans_mask]
                        outlier_info.update({
                            'cantidad': n_outliers,
                            'porcentaje': (n_outliers / len(data)) * 100,
                            'threshold': threshold,
                            'min_outlier': outliers_kmeans_data.min(),
                            'max_outlier': outliers_kmeans_data.max()
                        })
            except:
                n_outliers_kmeans = 0
                indices_outliers_kmeans = set()
        
        # MÉTODO SVM
        if metodo in ['svm', 'combinado'] and len(data) >= 10:
            try:
                data_reshaped = data.values.reshape(-1, 1)
                scaler = StandardScaler()
                data_scaled = scaler.fit_transform(data_reshaped)
                
                svm = OneClassSVM(nu=0.1, kernel='rbf', gamma='auto')
                predictions = svm.fit_predict(data_scaled)
                
                outliers_svm_mask = predictions == -1
                n_outliers_svm = outliers_svm_mask.sum()
                indices_outliers_svm = set(data.index[outliers_svm_mask])
                
                if metodo == 'svm':
                    n_outliers = n_outliers_svm
                    if n_outliers > 0:
                        outliers_svm_data = data[outliers_svm_mask]
                        outlier_info.update({
                            'cantidad': n_outliers,
                            'porcentaje': (n_outliers / len(data)) * 100,
                            'min_outlier': outliers_svm_data.min(),
                            'max_outlier': outliers_svm_data.max()
                        })
            except:
                n_outliers_svm = 0
                indices_outliers_svm = set()
        
        # MÉTODO COMBINADO
        if metodo == 'combinado':
            indices_unicos = indices_outliers_iqr | indices_outliers_kmeans | indices_outliers_svm
            n_outliers = len(indices_unicos)
            
            if n_outliers > 0:
                outlier_info.update({
                    'cantidad': n_outliers,
                    'porcentaje': (n_outliers / len(data)) * 100,
                    'outliers_iqr': n_outliers_iqr,
                    'outliers_kmeans': n_outliers_kmeans,
                    'outliers_svm': n_outliers_svm,
                    'outliers_unicos': n_outliers,
                    'overlapping': n_outliers_iqr + n_outliers_kmeans + n_outliers_svm - n_outliers,
                    'indices_outliers': list(indices_unicos)
                })
        
        # Guardar índices según método
        if metodo == 'iqr':
            indices_metodo = indices_outliers_iqr
        elif metodo == 'kmeans':
            indices_metodo = indices_outliers_kmeans
        elif metodo == 'svm':
            indices_metodo = indices_outliers_svm
        else:
            indices_metodo = indices_outliers_iqr | indices_outliers_kmeans | indices_outliers_svm
        
        todos_indices_outliers.update(indices_metodo)
        
        # Guardar el valor del outlier para cada índice
        for idx in indices_metodo:
            if idx not in outliers_por_indice:
                outliers_por_indice[idx] = {}
            # Guardar el valor original del outlier
            outliers_por_indice[idx][col] = df_numeric.loc[idx, col]
        
        if n_outliers > 0:
            if 'indices_outliers' not in outlier_info:
                outlier_info['indices_outliers'] = list(indices_metodo)
            outliers_por_columna[col] = outlier_info
        
        total_outliers += n_outliers
        total_datos += len(data)
    
    if total_datos > 0:
        pct_datos_precisos = ((total_datos - total_outliers) / total_datos) * 100
        if pct_datos_precisos >= 95:
            score = 20
        elif pct_datos_precisos >= 85:
            score = 15 + ((pct_datos_precisos - 85) / 10) * 5
        elif pct_datos_precisos >= 70:
            score = 5 + ((pct_datos_precisos - 70) / 15) * 10
        else:
            score = (pct_datos_precisos / 70) * 5
    else:
        pct_datos_precisos = 100
        score = 20
    
    # Crear DataFrame con filas de outliers incluyendo los valores
    if len(todos_indices_outliers) > 0:
        df_outliers_completo = df.loc[list(todos_indices_outliers)].copy()
        
        # Crear columnas para mostrar qué variables tienen outliers y sus valores
        variables_outlier = []
        valores_outlier = []
        
        for idx in df_outliers_completo.index:
            if idx in outliers_por_indice:
                vars_con_outlier = []
                vals_con_outlier = []
                
                for col, valor in outliers_por_indice[idx].items():
                    vars_con_outlier.append(col)
                    # Formatear el valor (redondear si es float)
                    if isinstance(valor, float):
                        vals_con_outlier.append(f"{col}: {valor:.4f}")
                    else:
                        vals_con_outlier.append(f"{col}: {valor}")
                
                variables_outlier.append(', '.join(vars_con_outlier))
                valores_outlier.append(' | '.join(vals_con_outlier))
            else:
                variables_outlier.append('')
                valores_outlier.append('')
        
        df_outliers_completo['🚨 Variables Outlier'] = variables_outlier
        df_outliers_completo['📊 Valores Outlier'] = valores_outlier
    else:
        df_outliers_completo = pd.DataFrame()
    
    return {
        'score': score, 'pct_datos_precisos': pct_datos_precisos,
        'outliers_por_columna': outliers_por_columna,
        'total_outliers': total_outliers, 'total_datos_numericos': total_datos,
        'metodo_usado': metodo, 'df_outliers_completo': df_outliers_completo,
        'num_filas_con_outliers': len(todos_indices_outliers)
    }


def calcular_variabilidad(df: pd.DataFrame, variables_numericas: List[str] = None) -> Dict:
    """
    Calcula la variabilidad mediante el Coeficiente de Variación (CV).
    
    Criterios de evaluación (alineados con la visualización):
    - CV < 10%: Baja variabilidad (puede indicar datos homogéneos o problema de escala)
    - CV 10-50%: Variabilidad moderada (ideal para la mayoría de variables)
    - CV 50-100%: Alta variabilidad (revisar si es esperado para el tipo de dato)
    - CV > 100%: Muy alta variabilidad (posibles outliers o problemas de datos)
    """
    df_numeric = preparar_dataframe_numerico(df, variables_numericas)
    
    if df_numeric.empty:
        return {
            'score': 15, 'cv_promedio': 0, 'cv_por_columna': {},
            'columnas_variabilidad_extrema': {}, 'pct_variabilidad_adecuada': 100
        }

    cv_por_columna = {}
    columnas_variabilidad_extrema = {}
    columnas_cv_adecuado = 0  # CV entre 10% y 100%
    cvs_validos = []

    for col in df_numeric.columns:
        data = df_numeric[col].dropna()

        if len(data) <= 1:
            continue

        mean = data.mean()
        std = data.std()

        # Manejo seguro cuando la media es muy pequeña o cero
        if abs(mean) < 1e-9:
            # Si la media es ~0, usar el rango como referencia alternativa
            data_range = data.max() - data.min()
            if data_range > 0:
                cv = (std / data_range) * 100  # CV relativo al rango
            else:
                cv = 0  # Todos los valores son iguales
        else:
            cv = (std / abs(mean)) * 100  # Usar valor absoluto de la media

        cv_por_columna[col] = cv
        cvs_validos.append(abs(cv))

        # Clasificar variabilidad con umbrales más estrictos
        if cv == 0 or (data.nunique() == 1):
            # Variabilidad nula - todos los valores son iguales
            columnas_variabilidad_extrema[col] = {
                'cv': cv, 
                'problema': 'Sin variabilidad (valores constantes)',
                'categoria': 'nula'
            }
        elif abs(cv) < 5:
            # Variabilidad muy baja - puede indicar problema
            columnas_variabilidad_extrema[col] = {
                'cv': cv, 
                'problema': 'Variabilidad muy baja (<5%)',
                'categoria': 'muy_baja'
            }
        elif abs(cv) < 10:
            # Variabilidad baja - aceptable pero revisar
            columnas_cv_adecuado += 0.5  # Cuenta parcialmente
        elif abs(cv) <= 100:
            # Variabilidad adecuada (10% - 100%)
            columnas_cv_adecuado += 1
        elif abs(cv) <= 150:
            # Variabilidad alta - revisar
            columnas_variabilidad_extrema[col] = {
                'cv': cv, 
                'problema': 'Variabilidad alta (100-150%)',
                'categoria': 'alta'
            }
            columnas_cv_adecuado += 0.5  # Cuenta parcialmente
        else:
            # Variabilidad excesiva (>150%)
            columnas_variabilidad_extrema[col] = {
                'cv': cv, 
                'problema': 'Variabilidad excesiva (>150%)',
                'categoria': 'excesiva'
            }

    n_columnas = len(cv_por_columna)

    if n_columnas > 0:
        # Calcular porcentaje de columnas con CV adecuado
        pct_adecuadas = (columnas_cv_adecuado / n_columnas) * 100
        
        # Score basado en el porcentaje de columnas con variabilidad adecuada
        # Máximo 15 puntos
        score = (pct_adecuadas / 100) * 15
        
        # Penalización adicional por columnas con problemas severos
        n_severas = sum(1 for info in columnas_variabilidad_extrema.values() 
                       if info.get('categoria') in ['nula', 'muy_baja', 'excesiva'])
        penalizacion = min(5, n_severas * 1.5)  # Máximo 5 puntos de penalización
        score = max(0, score - penalizacion)
    else:
        pct_adecuadas = 100
        score = 15

    cv_promedio = np.mean(cvs_validos) if cvs_validos else 0

    return {
        'score': round(score, 2),
        'cv_promedio': round(cv_promedio, 2),
        'cv_por_columna': cv_por_columna,
        'columnas_variabilidad_extrema': columnas_variabilidad_extrema,
        'pct_variabilidad_adecuada': round(pct_adecuadas, 2)
    }


def calcular_integridad(df: pd.DataFrame, columnas_esperadas: List[str] = None) -> Dict:
    """Calcula la integridad estructural de los datos"""
    if columnas_esperadas is None or len(columnas_esperadas) == 0:
        return {'score': 10, 'pct_integridad': 100, 'columnas_faltantes': [], 
                'columnas_extra': [], 'total_columnas': len(df.columns)}
    
    columnas_actuales = set(df.columns)
    columnas_esperadas_set = set(columnas_esperadas)
    
    columnas_faltantes = list(columnas_esperadas_set - columnas_actuales)
    columnas_extra = list(columnas_actuales - columnas_esperadas_set)
    
    columnas_coincidentes = len(columnas_esperadas_set & columnas_actuales)
    pct_integridad = (columnas_coincidentes / len(columnas_esperadas_set)) * 100
    
    score = (pct_integridad / 100) * 10
    
    return {
        'score': score, 'pct_integridad': pct_integridad,
        'columnas_faltantes': columnas_faltantes, 'columnas_extra': columnas_extra,
        'total_columnas': len(df.columns)
    }


def calcular_indice_calidad_datos(
    df: pd.DataFrame, 
    variables_numericas: List[str] = None,
    columnas_esperadas: List[str] = None,
    metodo_outliers: str = 'iqr'
) -> Dict:
    """Calcula el Índice de Calidad de Datos completo (0-100)"""
    completitud = calcular_completitud(df, variables_numericas)
    unicidad = calcular_unicidad(df, variables_numericas)
    consistencia = calcular_consistencia(df, variables_numericas)
    precision = calcular_precision_outliers(df, variables_numericas, metodo=metodo_outliers)
    variabilidad = calcular_variabilidad(df, variables_numericas)
    integridad = calcular_integridad(df, columnas_esperadas)
    
    icd_total = (
        completitud['score'] + unicidad['score'] + consistencia['score'] +
        precision['score'] + variabilidad['score'] + integridad['score']
    )
    
    if icd_total >= 90:
        nivel_calidad, color, emoji = "Excelente", "green", "🟢"
    elif icd_total >= 75:
        nivel_calidad, color, emoji = "Buena", "lightgreen", "🟡"
    elif icd_total >= 60:
        nivel_calidad, color, emoji = "Aceptable", "orange", "🟠"
    elif icd_total >= 40:
        nivel_calidad, color, emoji = "Baja", "orangered", "🟠"
    else:
        nivel_calidad, color, emoji = "Crítica", "red", "🔴"
    
    return {
        'icd_total': round(icd_total, 2),
        'nivel_calidad': nivel_calidad, 'color': color, 'emoji': emoji,
        'desglose': {
            'Completitud (25pts)': round(completitud['score'], 2),
            'Unicidad (15pts)': round(unicidad['score'], 2),
            'Consistencia (15pts)': round(consistencia['score'], 2),
            'Precisión (20pts)': round(precision['score'], 2),
            'Variabilidad (15pts)': round(variabilidad['score'], 2),
            'Integridad (10pts)': round(integridad['score'], 2)
        },
        'detalles': {
            'completitud': completitud, 'unicidad': unicidad,
            'consistencia': consistencia, 'precision': precision,
            'variabilidad': variabilidad, 'integridad': integridad
        }
    }


def generar_recomendaciones(resultado_icd: Dict) -> List[str]:
    """Genera recomendaciones basadas en el ICD calculado"""
    recomendaciones = []
    detalles = resultado_icd['detalles']
    
    if detalles['completitud']['score'] < 20:
        pct = detalles['completitud']['pct_completo']
        recomendaciones.append(
            f"⚠️ **Completitud baja ({pct:.1f}%)**: Considerar imputación de valores faltantes."
        )
    
    if detalles['unicidad']['filas_duplicadas'] > 0:
        n_dup = detalles['unicidad']['filas_duplicadas']
        recomendaciones.append(
            f"⚠️ **{n_dup} filas duplicadas detectadas**: Revisar si son errores de carga."
        )
    
    if detalles['precision']['score'] < 15:
        n_out = detalles['precision']['total_outliers']
        recomendaciones.append(
            f"⚠️ **Outliers significativos ({n_out} detectados)**: Revisar valores atípicos."
        )
    
    # Recomendaciones mejoradas para variabilidad
    variabilidad = detalles['variabilidad']
    if variabilidad['columnas_variabilidad_extrema']:
        problemas_por_tipo = {}
        for col, info in variabilidad['columnas_variabilidad_extrema'].items():
            categoria = info.get('categoria', 'otro')
            if categoria not in problemas_por_tipo:
                problemas_por_tipo[categoria] = []
            problemas_por_tipo[categoria].append(col)
        
        if 'nula' in problemas_por_tipo or 'muy_baja' in problemas_por_tipo:
            cols = problemas_por_tipo.get('nula', []) + problemas_por_tipo.get('muy_baja', [])
            recomendaciones.append(
                f"⚠️ **{len(cols)} columnas con variabilidad muy baja**: "
                f"Revisar si los datos son correctos o si hay problemas de captura. "
                f"Variables: {', '.join(cols[:5])}{'...' if len(cols) > 5 else ''}"
            )
        
        if 'excesiva' in problemas_por_tipo:
            cols = problemas_por_tipo['excesiva']
            recomendaciones.append(
                f"⚠️ **{len(cols)} columnas con variabilidad excesiva (CV>150%)**: "
                f"Posibles outliers o diferentes unidades de medida. "
                f"Variables: {', '.join(cols[:5])}{'...' if len(cols) > 5 else ''}"
            )
    
    if detalles['consistencia']['valores_inconsistentes'] > 0:
        n_incons = detalles['consistencia']['valores_inconsistentes']
        recomendaciones.append(
            f"⚠️ **{n_incons} valores inconsistentes**: Estandarizar formatos."
        )
    
    if not recomendaciones:
        recomendaciones.append("✅ **Los datos tienen buena calidad general**. Listo para análisis.")
    
    return recomendaciones
