# 🌱 SueloGuIA - Agente de Datos de Suelos Agrosavia

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-green.svg)](https://openai.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.2+-yellow.svg)](https://langchain.com)
[![Accesibilidad](https://img.shields.io/badge/Accesibilidad-Voz-orange.svg)](#-accesibilidad-por-voz)

Herramienta integral para el análisis de calidad de datos de suelos agrícolas, cálculo del **Índice de Calidad de Datos (ICD)** y consultas mediante asistentes conversacionales con IA. **Diseñada con funciones de voz para facilitar el acceso a población rural y campesina.**

---

## 👥 Usuarios y Niveles de Uso

SueloGuIA está diseñada para facilitar el uso de los datos de suelos al **personal de Agrosavia** y a la **comunidad agrícola**, apoyando el análisis y la toma de decisiones a partir de la información recolectada. La plataforma ofrece **tres niveles de interacción** adaptados a diferentes perfiles de usuario:

| Nivel | Módulo | Usuario objetivo | Tipo de información |
|-------|--------|------------------|---------------------|
| 🔬 **Experto** | Análisis Estadístico e ICD | Investigadores, técnicos de laboratorio | Estadísticos descriptivos, detección de outliers, índices de calidad, visualizaciones técnicas |
| 👨‍💻 **Técnico** | Agente de Datos (Pandas) | Profesionales agropecuarios, extensionistas | Consultas en lenguaje natural sobre los datos, correlaciones, filtros por cultivo/región |
| 👨‍🌾 **Campo** | Agente de Asistencia (RAG + Voz) | Agricultores, campesinos | Recomendaciones prácticas en lenguaje sencillo, interacción por voz, respuestas orientadas a la acción |

Esta arquitectura multinivel permite que:
- Los **investigadores de Agrosavia** realicen análisis profundos de calidad de datos antes de publicar o usar la información
- Los **extensionistas** consulten rápidamente información específica sin necesidad de manipular tablas de datos o programar
- Los **agricultores** reciban orientación clara y accionable sobre el manejo de sus suelos, sin barreras tecnológicas

---

## 🎯 Accesibilidad para el Campo Colombiano

> *"La Ciencia Más útil es aquella cuyo fruto es el más comunicable. (L. Davinci)"*

SueloGuIA incorpora **funcionalidades de voz** pensadas especialmente para la **población campesina** que puede tener dificultades con la lectura o escritura:

### 🎤 Haz preguntas con tu voz
No necesitas escribir. Simplemente **graba tu voz** y el sistema transcribirá automáticamente tu consulta.

### 🔊 Escucha las respuestas
Las respuestas del asistente pueden ser **reproducidas en audio**, facilitando la comprensión sin necesidad de leer textos extensos.

### 💡 ¿Por qué es importante?

- **Inclusión digital**: Democratiza el acceso a información técnica agrícola
- **Analfabetismo funcional**: Según la GEIH 2023 del DANE, la tasa de analfabetismo en la zona rural de Colombia es de aproximadamente 10,4 %
- **Comodidad**: Los agricultores pueden consultar mientras trabajan en campo
- **Idioma natural**: Permite hacer preguntas como se habla cotidianamente

---

## 🌐 Demo en Vivo

La aplicación está desplegada en **Streamlit Cloud**:

🔗 **[Acceder a SueloGuIA](https://idcmulagrosavia.streamlit.app/)**

---

## 📋 Descripción

SueloGuIA es una aplicación web desarrollada con Streamlit que permite:

- **Cargar y procesar** datos de análisis de suelos desde archivos CSV/Excel o APIs Socrata
- **Calcular el Índice de Calidad de Datos (ICD)** con 6 dimensiones de evaluación
- **Visualizar estadísticas** descriptivas y detectar outliers con múltiples métodos
- **Consultar datos** mediante lenguaje natural con un agente IA (GPT + Pandas)
- **Obtener recomendaciones** agronómicas mediante RAG (Retrieval-Augmented Generation)
- **Interactuar por voz** 🎤 para hacer preguntas y 🔊 escuchar respuestas

---

## 🎙️ Multimodalidad

SueloGuIA integra capacidades de **voz** para una experiencia más accesible e inclusiva:

| Funcionalidad | Tecnología | Descripción |
|---------------|------------|-------------|
| **Voz a Texto** | OpenAI Whisper | Transcribe preguntas habladas al sistema |
| **Texto a Voz** | OpenAI TTS | Reproduce las respuestas en audio con voces naturales |
| **Grabación** | audio-recorder-streamlit | Captura audio directamente desde el navegador |

**Formatos de audio soportados:** WAV, MP3, M4A, OGG

### 💰 Costos y Alternativa Local

La versión actual utiliza la API de OpenAI, que tiene costos por uso:

| Servicio | Costo aproximado |
|----------|------------------|
| GPT-4 (chat/RAG) | ~$0.01-0.03 por consulta |
| Whisper (voz a texto) | $0.006 por minuto de audio |
| TTS (texto a voz) | $0.015 por cada 1,000 caracteres |

> 💡 **Alternativa sin costo**: Este sistema puede implementarse completamente en local usando herramientas open source:
> - **[Ollama](https://ollama.ai/)** - Modelos de lenguaje locales (Llama, Mistral, etc.)
> - **[Whisper](https://github.com/openai/whisper)** - Transcripción de voz local
> - **[Piper](https://github.com/rhasspy/piper)** - Síntesis de voz local en español
>
> Esta configuración elimina la dependencia de APIs externas y los costos asociados, ideal para despliegues rurales con conectividad limitada.

---

## 🏗️ Estructura del Proyecto

```
sueloguia/
│
├── 📥_Inicio.py              # Página principal - Carga de datos
├── utils.py                  # Utilidades: limpieza, normalización, tipos
├── calidad_datos.py          # Cálculo del Índice de Calidad de Datos (ICD)
├── visualizaciones.py        # Estadísticos descriptivos y gráficos
├── recomendaciones.pdf       # Documento base para RAG (recomendaciones agronómicas)
│
├── pages/
│   ├── 2_📊_Análisis e IDC.py              # Análisis estadístico y cálculo de ICD
│   ├── 3_👨‍💻_Asistente de datos.py          # Agente conversacional con Pandas
│   └── 4_👨‍🌾_Asistente Campesino.py         # Asistente RAG con voz habilitada
│
├── .streamlit/
│   └── secrets.toml                         # Configuración de secrets (solo local)
│
├── requirements.txt                         # Dependencias del proyecto
└── README.md                                # Este archivo
```

---

## 🎯 Funcionalidades

### 1. Carga de Datos (`📥_Inicio.py`)

- **API Socrata**: Conexión directa a datos.gov.co y otros portales de datos abiertos
- **Archivos locales**: Soporte para CSV y Excel (.xlsx, .xls)
- **Normalización**: Estandarización de nombres de columnas (tildes, espacios, mayúsculas)

### 2. Índice de Calidad de Datos - ICD (`📊_Análisis e IDC.py`)

El ICD evalúa la calidad de los datos en **6 dimensiones** con un puntaje total de 0-100:

| Dimensión | Puntos | Descripción |
|-----------|--------|-------------|
| **Completitud** | 25 | Porcentaje de valores no nulos |
| **Precisión** | 20 | Detección de outliers (IQR, K-means, SVM) |
| **Unicidad** | 15 | Identificación de registros duplicados |
| **Consistencia** | 15 | Valores con tipos de datos mixtos |
| **Variabilidad** | 15 | Coeficiente de variación por columna |
| **Integridad** | 10 | Columnas esperadas vs. disponibles |

**Niveles de calidad:**
- 🟢 **Excelente** (≥90): Datos listos para análisis avanzados
- 🟡 **Buena** (75-89): Utilizables con limpieza menor
- 🟠 **Aceptable** (60-74): Requiere limpieza antes de análisis
- 🟠 **Baja** (40-59): Limpieza profunda requerida
- 🔴 **Crítica** (<40): Revisar proceso de captura

### 3. Detección de Outliers

Tres métodos disponibles para la dimensión de Precisión:

- **IQR (Cuartiles)**: Método tradicional basado en rango intercuartílico
- **K-means**: Clustering para identificar puntos distantes de centroides
- **SVM (One-Class)**: Aprendizaje automático para detección de anomalías
- **Combinado**: Unión de los tres métodos

### 4. Visualizaciones (`visualizaciones.py`)

- Histogramas de distribución
- Boxplots para detección visual de outliers
- Matriz de correlación con heatmap
- Tabla de estadísticos descriptivos completa

### 5. Agente IA para Consultas (`👨‍💻_Asistente de datos.py   `)

Utiliza LangChain + OpenAI GPT para responder preguntas en lenguaje natural:

```
Ejemplos de consultas:
- "¿Cuál es la media de pH en los cultivos de café?"
- "Muestra un resumen estadístico de materia orgánica"
- "¿Cuál es la correlación mayor entre las variables numéricas?"
- "¿Qué cultivos se dan en el municipio de Pasca?"
```

### 6. RAG con Recomendaciones y Voz (`👨‍🌾_Asistente Campesino.py`)

Sistema de Retrieval-Augmented Generation que consulta el documento `recomendaciones.pdf`, **con soporte completo de voz**:

```
Ejemplos de consultas (escritas o habladas):
- 🎤 "¿Qué hago si mi tierra tiene mucho aluminio?"
- 🎤 "¿Por qué el pH de mi suelo está bajito?"
- 🎤 "¿Cómo mejoro la materia orgánica de mi finca?"
```

**Características:**
- ⌨️ **Escribir**: Entrada tradicional por texto
- 🎤 **Grabar voz**: Grabación directa desde micrófono
- 🔊 **Respuesta en audio**: Activa desde la barra lateral

---

## 🛠️ Instalación Local

### Prerrequisitos

- Python 3.9 o superior
- pip (gestor de paquetes de Python)
- API Key de OpenAI (para funcionalidades de IA)
- Micrófono (opcional, para funciones de voz)

### Pasos de instalación

1. **Clonar el repositorio**
   ```bash
   git clone https://github.com/tu-usuario/sueloguia.git
   cd sueloguia
   ```

2. **Crear entorno virtual** (recomendado)
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

3. **Instalar dependencias**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configurar secrets** (ver sección de configuración)

5. **Ejecutar la aplicación**
   ```bash
   streamlit run Inicio.py
   ```

---

## ⚙️ Configuración

### Configuración de API Key (Secrets)

La aplicación utiliza `st.secrets` para manejar las credenciales de forma segura.

#### Desarrollo Local

Crea el archivo `.streamlit/secrets.toml` en la raíz del proyecto:

```toml
[settings]
key = "sk-proj-tu-api-key-de-openai"
```

> ⚠️ **Importante**: Agrega `.streamlit/secrets.toml` a tu `.gitignore` para no exponer tu API Key.

#### Streamlit Cloud

1. Ve a tu aplicación en [share.streamlit.io](https://share.streamlit.io)
2. Haz clic en **Settings** (⚙️) → **Secrets**
3. Agrega la configuración:

```toml
[settings]
key = "sk-proj-tu-api-key-de-openai"
```

4. Guarda los cambios y reinicia la aplicación

### Configuración de Socrata

Para conectar a datos.gov.co:
- **Dominio**: `www.datos.gov.co`
- **Dataset ID**: `ch4u-f3i5` (datos de suelos Agrosavia)
- **App Token**: Opcional, pero recomendado para mayor límite de requests

---

## 📦 Dependencias

```txt
# Core
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0

# Visualización
plotly>=5.18.0

# Machine Learning (detección de outliers)
scikit-learn>=1.3.0

# API Socrata
sodapy>=2.2.0

# IA y LangChain
langchain>=0.2.0
langchain-openai>=0.1.0
langchain-experimental>=0.0.50
langchain-community>=0.2.0
openai>=1.0.0

# RAG / Procesamiento de PDF
pypdf>=3.0.0
faiss-cpu>=1.7.0

# Multimodalidad (Voz)
audio-recorder-streamlit==0.0.10
```

---

## 🚀 Uso

### 1. Cargar datos

Desde la página principal, puedes:

- **Conectar a API Socrata** (ej: datos.gov.co, dataset `ch4u-f3i5`)
- **Subir un archivo** CSV o Excel con datos de suelos

### 2. Analizar calidad de datos

En la página **📊 Análisis e IDC**:

1. Selecciona las variables a analizar
2. Elige el método de detección de outliers
3. Haz clic en "Generar Análisis"
4. Revisa el ICD, estadísticos, algoritmos de Machine learning para identificación de oultliers y visualizaciones

### 3. Consultas con IA

En la página **👨‍💻_Asistente de datos.py **:

1. Las credenciales se cargan automáticamente desde secrets
2. Escribe tu pregunta en lenguaje natural
3. El agente analizará y responderá sobre los datos consultados

### 4. Consultas sobre recomendaciones (con voz) 🎤🔊

En la página **👨‍🌾_Asistente Campesino.py**:

1. Las credenciales se cargan automáticamente desde secrets
2. **Escribir**: Escribe tu pregunta en el formulario
3. **Hablar**: Graba tu voz y presiona "Transcribir y preguntar"
4. **Escuchar**: Activa "🔊 Habilitar respuesta por voz" en la barra lateral

---

## 📊 Variables de Suelos Soportadas

La aplicación está optimizada para las siguientes variables de análisis de suelos:

| Variable | Descripción |
|----------|-------------|
| `ph_agua_suelo` | pH del suelo en agua |
| `materia_organica` | Contenido de materia orgánica (%) |
| `fosforo_bray_ii` | Fósforo disponible (ppm) |
| `azufre_fosfato_monocalcico` | Azufre disponible (ppm) |
| `acidez_kcl` | Acidez intercambiable |
| `aluminio_intercambiable` | Aluminio intercambiable (cmol/kg) |
| `calcio_intercambiable` | Calcio intercambiable (cmol/kg) |
| `magnesio_intercambiable` | Magnesio intercambiable (cmol/kg) |
| `potasio_intercambiable` | Potasio intercambiable (cmol/kg) |
| `sodio_intercambiable` | Sodio intercambiable (cmol/kg) |
| `capacidad_de_intercambio_cationico` | CIC (cmol/kg) |
| `conductividad_electrica` | CE (dS/m) |
| `hierro_disponible_olsen` | Hierro disponible - Olsen (ppm) |
| `cobre_disponible` | Cobre disponible (ppm) |
| `manganeso_disponible_olsen` | Manganeso disponible - Olsen (ppm) |
| `zinc_disponible_olsen` | Zinc disponible - Olsen (ppm) |
| `boro_disponible` | Boro disponible (ppm) |

---

## 🌾 Impacto Social

SueloGuIA busca contribuir a:

- **Democratización del conocimiento agrícola**: Información técnica accesible para todos
- **Inclusión digital rural**: Tecnología adaptada a las necesidades del campo
- **Mejora de la productividad**: Decisiones informadas basadas en datos de calidad
- **Sostenibilidad agrícola**: Mejor manejo de suelos basado en evidencia

---

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Haz fork del repositorio
2. Crea una rama para tu feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit tus cambios (`git commit -am 'Agrega nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

---

## 👥 Autores

- **SUME** - Desarrollo inicial

---

## 🙏 Agradecimientos

- [Agrosavia](https://www.agrosavia.co/) - Datos de análisis de suelos
- [Datos Abiertos Colombia](https://www.datos.gov.co/) - Plataforma de datos abiertos
- [Streamlit](https://streamlit.io/) - Framework de aplicaciones web
- [LangChain](https://langchain.com/) - Framework para aplicaciones con LLMs
- [OpenAI](https://openai.com/) - Modelos de lenguaje GPT, Whisper y TTS

---

## 📞 Soporte

Si tienes preguntas o problemas, por favor abre un issue en el repositorio.

---

<p align="center">
  <i>Hecho por SUME con ❤️ para el campo colombiano</i>
</p>
