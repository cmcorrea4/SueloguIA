"""
Utilidades compartidas para la aplicación Agrosavia
"""
import pandas as pd
import numpy as np
import unicodedata
from typing import Dict, List
from sklearn.cluster import KMeans
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

# Lista de variables para análisis estadístico
VARIABLES_ESTADISTICAS = [
    'ph_agua_suelo',
    'materia_organica',
    'fosforo_bray_ii',
    'azufre_fosfato_monocalcico',
    'acidez_kcl',
    'aluminio_intercambiable',
    'calcio_intercambiable',
    'magnesio_intercambiable',
    'potasio_intercambiable',
    'sodio_intercambiable',
    'capacidad_de_intercambio_cationico',
    'conductividad_electrica',
    'hierro_disponible_olsen',
    'cobre_disponible',
    'manganeso_disponible_olsen',
    'zinc_disponible_olsen',
    'boro_disponible',
    'hierro_disponible_doble_acido',
    'cobre_disponible_doble_acido',
    'manganeso_disponible_doble_acido',
    'zinc_disponible_doble_acido'
]

# Columnas numéricas para tipado
COLUMNAS_NUMERICAS = [
    "ph_agua_suelo", "materia_organica", "fosforo_bray_ii",
    "azufre_fosfato_monocalcico", "acidez_kcl",
    "aluminio_intercambiable", "calcio_intercambiable",
    "magnesio_intercambiable", "potasio_intercambiable",
    "sodio_intercambiable", "capacidad_de_intercambio_cationico",
    "conductividad_electrica", "hierro_disponible_olsen",
    "cobre_disponible", "manganeso_disponible_olsen",
    "zinc_disponible_olsen", "boro_disponible",
    "hierro_disponible_doble_acido", "cobre_disponible_doble_acido",
    "manganeso_disponible_doble_acido", "zinc_disponible_doble_acido"
]


def normalizar_nombres_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza los nombres de columnas para que coincidan con VARIABLES_ESTADISTICAS.
    Convierte a minúsculas, reemplaza espacios por guiones bajos, elimina tildes.
    
    Ejemplos de conversión:
        'Ph Agua Suelo' -> 'ph_agua_suelo'
        'Materia Orgánica' -> 'materia_organica'
        'FOSFORO_BRAY_II' -> 'fosforo_bray_ii'
    """
    def normalizar(texto):
        # Convertir a minúsculas
        texto = str(texto).lower().strip()
        # Eliminar tildes/acentos
        texto = unicodedata.normalize('NFD', texto)
        texto = ''.join(c for c in texto if unicodedata.category(c) != 'Mn')
        # Reemplazar espacios y guiones por guiones bajos
        texto = texto.replace(' ', '_').replace('-', '_')
        # Eliminar caracteres especiales (solo letras, números y guion bajo)
        texto = ''.join(c if c.isalnum() or c == '_' else '_' for c in texto)
        # Eliminar guiones bajos múltiples
        while '__' in texto:
            texto = texto.replace('__', '_')
        return texto.strip('_')
    
    df_copy = df.copy()
    df_copy.columns = [normalizar(col) for col in df_copy.columns]
    return df_copy


def asignar_tipos_datos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Asigna correctamente los tipos de datos a las columnas del DataFrame.
    Primero normaliza nombres de columnas, luego convierte tipos.
    Columnas numéricas específicas se convierten a float, el resto a string.
    """
    # Primero normalizar nombres de columnas
    df_typed = normalizar_nombres_columnas(df)
    
    # Convertir columnas numéricas
    for col in COLUMNAS_NUMERICAS:
        if col in df_typed.columns:
            if df_typed[col].dtype == 'object':
                df_typed[col] = pd.to_numeric(
                    df_typed[col].astype(str).str.replace(',', '.').str.strip(),
                    errors='coerce'
                )
            else:
                df_typed[col] = pd.to_numeric(df_typed[col], errors='coerce')
    
    # Convertir el resto de columnas a string
    for col in df_typed.columns:
        if col not in COLUMNAS_NUMERICAS:
            df_typed[col] = df_typed[col].astype(str)
    
    return df_typed


def preparar_dataframe_numerico(df: pd.DataFrame, variables: List[str] = None) -> pd.DataFrame:
    """
    Convierte DataFrame a numérico de forma consistente para todos los análisis.
    """
    if variables is not None and len(variables) > 0:
        vars_disponibles = [v for v in variables if v in df.columns]
        if vars_disponibles:
            df_work = df[vars_disponibles].copy()
        else:
            return pd.DataFrame()
    else:
        df_work = df.copy()
    
    for col in df_work.columns:
        if df_work[col].dtype == 'object':
            df_work[col] = pd.to_numeric(
                df_work[col].astype(str).str.replace(',', '.').str.strip(),
                errors='coerce'
            )
    
    return df_work


class DataCleaner:
    """Limpiador automático de datos"""
    
    @staticmethod
    def clean_dataframe(df: pd.DataFrame) -> tuple:
        """Limpia el DataFrame eliminando filas/columnas vacías, duplicados, etc."""
        original_shape = df.shape
        cleaning_report = []
        
        # Eliminar filas vacías
        rows_before = len(df)
        df = df.dropna(how='all')
        rows_removed = rows_before - len(df)
        if rows_removed > 0:
            cleaning_report.append(f"🗑️ Eliminadas {rows_removed} filas completamente vacías")
        
        # Eliminar columnas vacías
        cols_before = len(df.columns)
        df = df.dropna(axis=1, how='all')
        cols_removed = cols_before - len(df.columns)
        if cols_removed > 0:
            cleaning_report.append(f"🗑️ Eliminadas {cols_removed} columnas completamente vacías")
        
        # Convertir columnas numéricas
        numeric_cols_cleaned = 0
        for col in df.columns:
            if df[col].dtype == 'object':
                sample = df[col].dropna().head(100)
                if len(sample) > 0:
                    numeric_count = 0
                    for val in sample:
                        try:
                            float(str(val).replace(',', '.').strip())
                            numeric_count += 1
                        except:
                            pass
                    
                    if numeric_count / len(sample) > 0.8:
                        original_nulls = df[col].isna().sum()
                        df[col] = pd.to_numeric(
                            df[col].astype(str).str.replace(',', '.').str.strip(),
                            errors='coerce'
                        )
                        new_nulls = df[col].isna().sum()
                        invalid_values = new_nulls - original_nulls
                        
                        if invalid_values > 0:
                            cleaning_report.append(
                                f"🔢 Columna '{col}': convertida a numérica ({invalid_values} valores inválidos → NaN)"
                            )
                            numeric_cols_cleaned += 1
        
        # Eliminar filas sin datos numéricos
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            rows_before = len(df)
            df = df.dropna(subset=numeric_columns, how='all')
            rows_removed = rows_before - len(df)
            if rows_removed > 0:
                cleaning_report.append(
                    f"🗑️ Eliminadas {rows_removed} filas sin datos numéricos válidos"
                )
        
        # Eliminar duplicados
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            df = df.drop_duplicates()
            cleaning_report.append(f"🗑️ Eliminadas {duplicates} filas duplicadas")
        
        # Limpiar texto
        text_cols_cleaned = 0
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace(['nan', 'None', 'NaN', ''], np.nan)
            text_cols_cleaned += 1
        
        if text_cols_cleaned > 0:
            cleaning_report.append(f"✨ Limpiados espacios en {text_cols_cleaned} columnas de texto")
        
        df = df.reset_index(drop=True)
        
        final_shape = df.shape
        cleaning_report.insert(0, f"📊 Dimensiones: {original_shape} → {final_shape}")
        
        return df, cleaning_report
