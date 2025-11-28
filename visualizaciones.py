"""
Funciones para estadísticos y visualizaciones
"""
import pandas as pd
import numpy as np
from typing import List
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

from utils import preparar_dataframe_numerico


def calcular_estadisticos(df: pd.DataFrame, variables: list) -> pd.DataFrame:
    """Calcula estadísticos descriptivos para las variables especificadas"""
    df_numeric = preparar_dataframe_numerico(df, variables)
    
    if df_numeric.empty:
        return None
    
    means = df_numeric.mean()
    stds = df_numeric.std()
    
    # Calcular CV evitando división por cero
    cv_values = []
    for col in df_numeric.columns:
        if means[col] != 0:
            cv_values.append((stds[col] / means[col]) * 100)
        else:
            cv_values.append(np.nan)
    
    # Detección de outliers por variable
    outliers_iqr = []
    outliers_kmeans = []
    outliers_svm = []
    outliers_total = []
    
    for col in df_numeric.columns:
        data = df_numeric[col].dropna()
        n_outliers_iqr = 0
        n_outliers_kmeans = 0
        n_outliers_svm = 0
        
        if len(data) > 0:
            # Método IQR
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            n_outliers_iqr = ((data < lower_bound) | (data > upper_bound)).sum()
            
            # Método K-MEANS
            if len(data) >= 3:
                try:
                    data_reshaped = data.values.reshape(-1, 1)
                    scaler = StandardScaler()
                    data_scaled = scaler.fit_transform(data_reshaped)
                    
                    kmeans = KMeans(n_clusters=min(3, len(data)), random_state=42, n_init=10)
                    kmeans.fit(data_scaled)
                    distances = np.min(kmeans.transform(data_scaled), axis=1)
                    threshold = np.percentile(distances, 90)
                    n_outliers_kmeans = (distances > threshold).sum()
                except:
                    n_outliers_kmeans = 0
            
            # Método SVM
            if len(data) >= 10:
                try:
                    data_reshaped = data.values.reshape(-1, 1)
                    scaler = StandardScaler()
                    data_scaled = scaler.fit_transform(data_reshaped)
                    
                    svm = OneClassSVM(nu=0.1, kernel='rbf', gamma='auto')
                    predictions = svm.fit_predict(data_scaled)
                    n_outliers_svm = (predictions == -1).sum()
                except:
                    n_outliers_svm = 0
        
        outliers_iqr.append(n_outliers_iqr)
        outliers_kmeans.append(n_outliers_kmeans)
        outliers_svm.append(n_outliers_svm)
        outliers_total.append(n_outliers_iqr + n_outliers_kmeans + n_outliers_svm)
    
    stats = pd.DataFrame({
        'Variable': df_numeric.columns,
        'Count': df_numeric.count().values,
        'Media': means.values,
        'Mediana': df_numeric.median().values,
        'Desv. Std': stds.values,
        'Mínimo': df_numeric.min().values,
        'Q1 (25%)': df_numeric.quantile(0.25).values,
        'Q3 (75%)': df_numeric.quantile(0.75).values,
        'Máximo': df_numeric.max().values,
        'Rango': (df_numeric.max() - df_numeric.min()).values,
        'CV (%)': cv_values,
        'Asimetría': df_numeric.skew().values,
        'Curtosis': df_numeric.kurtosis().values,
        'Valores nulos': df_numeric.isnull().sum().values,
        'Outliers IQR': outliers_iqr,
        'Outliers K-means': outliers_kmeans,
        'Outliers SVM': outliers_svm,
        'Total Outliers': outliers_total
    })
    
    return stats


def crear_histogramas(df: pd.DataFrame, variables: list):
    """Crea histogramas para las variables seleccionadas"""
    df_numeric = preparar_dataframe_numerico(df, variables)
    
    if df_numeric.empty:
        return None
    
    n_vars = len(df_numeric.columns)
    n_cols = 3
    n_rows = (n_vars + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=df_numeric.columns.tolist(),
        vertical_spacing=0.12, horizontal_spacing=0.08
    )
    
    for idx, col in enumerate(df_numeric.columns):
        row = idx // n_cols + 1
        col_pos = idx % n_cols + 1
        data = df_numeric[col].dropna()
        
        fig.add_trace(
            go.Histogram(x=data, name=col, marker_color='steelblue', showlegend=False),
            row=row, col=col_pos
        )
    
    fig.update_layout(height=300 * n_rows, title_text="Distribución de Variables", showlegend=False)
    return fig


def crear_boxplots(df: pd.DataFrame, variables: list):
    """Crea boxplots para las variables seleccionadas"""
    df_numeric = preparar_dataframe_numerico(df, variables)
    
    if df_numeric.empty:
        return None
    
    n_vars = len(df_numeric.columns)
    n_cols = 3
    n_rows = (n_vars + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=df_numeric.columns.tolist(),
        vertical_spacing=0.12, horizontal_spacing=0.08
    )
    
    for idx, col in enumerate(df_numeric.columns):
        row = idx // n_cols + 1
        col_pos = idx % n_cols + 1
        data = df_numeric[col].dropna()
        
        fig.add_trace(
            go.Box(y=data, name=col, marker_color='lightseagreen', showlegend=False),
            row=row, col=col_pos
        )
    
    fig.update_layout(height=300 * n_rows, title_text="Boxplots de Variables", showlegend=False)
    return fig


def crear_matriz_correlacion(df: pd.DataFrame, variables: list):
    """Crea matriz de correlación para las variables seleccionadas"""
    df_numeric = preparar_dataframe_numerico(df, variables)
    
    if df_numeric.empty or len(df_numeric.columns) < 2:
        return None
    
    corr_matrix = df_numeric.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 8},
        colorbar=dict(title="Correlación")
    ))
    
    fig.update_layout(
        title="Matriz de Correlación",
        xaxis_title="Variables", yaxis_title="Variables",
        height=600, width=800
    )
    
    return fig
