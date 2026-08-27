








# Optimización y despliegue cloud de sistemas RAG (TFG-RAG-VDB)

Repositorio oficial del Trabajo de Fin de Grado en Ingeniería Informática y Telemática. Este proyecto comprende el diseño, implementación, evaluación empírica y optimización de una arquitectura desacoplada para sistemas de Generación Aumentada por Recuperación (RAG, *Retrieval-Augmented Generation*), transitando desde un prototipo monolítico en local hasta una infraestructura distribuida y elástica en Amazon Web Services (AWS).

---

## 1. Descripción de la Arquitectura

El sistema implementa un patrón híbrido desacoplado que separa la persistencia documental estructurada del índice de búsqueda vectorial:

1. **Fuente de Verdad (Persistencia Relacional):**  
   Almacena los textos completos, títulos y metadatos asociados a cada identificador único (`doc_id`).
   * *Entorno Local:* Contenedor Docker de PostgreSQL.
   * *Entorno Cloud:* Instancia *serverless* en Neon PostgreSQL (región de Frankfurt).

2. **Índice Vectorial (Búsqueda Semántica):**  
   Indexa exclusivamente las incrustaciones matemáticas (*embeddings*) y los identificadores de documento para realizar búsquedas por vecinos más cercanos con baja latencia.
   * *Entorno Local:* ChromaDB en memoria / Docker.
   * *Entorno Cloud:* LanceDB persistido sobre un *bucket* de Amazon S3, empleando índices aproximados cuantizados (**IVF-PQ**) y refinamiento de candidatos (*Refine Factor*).

3. **Módulo de Generación y Evaluación:**  
   Orquestación del ciclo generativo mediante modelos de lenguaje (APIs de OpenAI y Google) y auditoría semántica integral a través del *framework* DeepEval (métrica GEval / *Correctness*) bajo el paradigma *LLM-as-a-Judge*.

---

## 2. Estructura del Repositorio

```text
TFG-RAG-VDB/
├── data_test/                  # Conjunto de datos SQuAD en formato Parquet y CSVs de resultados
├── src/                        # Código fuente del sistema
│   ├── api-client/             # Módulos de inferencia con LLMs y scripts de evaluación
│   └── back_end/               # Funciones de recuperación, conexión relacional y gestión vectorial
├── .deepeval/                  # Configuraciones y registros temporales de evaluación
├── .env.example                # Plantilla de variables de entorno requeridas
├── requirements.txt            # Dependencias del proyecto
└── README.md                   # Documentación principal
```

## 3. Requisitos Previos
Python 3.11 o superior.
* Docker y Docker Compose (opcional, para ejecución de la línea base en local).
* Cuenta de AWS con permisos IAM configurados para Amazon S3 y EC2.
* Cuenta de Neon.tech para PostgreSQL serverless.
* Claves de API activas para OpenAI (OPENAI_API_KEY) y Google GenAI (GEMINI_API_KEY).

## 4. Configuración del Entorno
### 4.1. Clonado del repositorio
```Bash
git clone [https://github.com/IvanGP7/TFG-RAG-VDB.git](https://github.com/IvanGP7/TFG-RAG-VDB.git)
cd TFG-RAG-VDB
```
### 4.2. Creación del entorno virtual
```Bash
python -m venv .venv
```
En Windows:

```PowerShell
.\.venv\Scripts\Activate.ps1
```
En Linux / macOS / AWS EC2:
```Bash
source .venv/bin/activate
```
### 4.3. Instalación de dependencias
```Bash
pip install --upgrade pip
pip install -r requirements.txt
```
Nota para entornos Linux/EC2: En sistemas basados en Linux se recomienda el uso de psycopg2-binary para evitar errores de compilación de librerías nativas C.

### 4.4. Configuración de Variables de Entorno
Crea un archivo .env en la raíz del proyecto a partir de la plantilla .env.example:

```Ini, TOML
# Credenciales APIs LLMs
OPENAI_API_KEY=tu_openai_api_key
GEMINI_API_KEY=tu_gemini_api_key

# Conexión Base de Datos Relacional (Local / Neon Cloud)
DB_HOST=tu_host_neon_o_localhost
DB_PORT=5432
DB_NAME=tu_base_de_datos
DB_USER=tu_usuario
DB_PASSWORD=tu_contraseña

# Configuración AWS S3 (LanceDB Cloud)
AWS_ACCESS_KEY_ID=tu_aws_access_key
AWS_SECRET_ACCESS_KEY=tu_aws_secret_key
AWS_REGION=eu-central-1
S3_BUCKET_URI=s3://tu-bucket-lancedb/vectores/
```
## 5. Guía de Ejecución
### 5.1. Despliegue de la línea base local (ChromaDB + PostgreSQL)
Para levantar los contenedores locales de ChromaDB y PostgreSQL:

```Bash
docker run -v ./chroma-data:/data -p 8000:8000 chromadb/chroma
```
### 5.2. Ingesta y ETL de datos
Ejecuta el procesamiento del dataset SQuAD para poblar la base de datos relacional y construir el índice vectorial:

```Bash
python src/back_end/data_loader.py
```
### 5.3. Benchmark de modelos de incrustación (Hit Rate @ K)
Para ejecutar la comparativa sistemática de modelos de embeddings sobre las 87.355 preguntas del conjunto de pruebas:

```Bash
python src/back_end/test_question.py
```
### 5.4. Pipeline generativo y evaluación semántica (DeepEval)
Para procesar las consultas contra el pipeline RAG y evaluar la corrección conceptual mediante gpt-4o-mini:

```Bash
python src/main.py
```
## 6. Resultados y Métricas Destacadas
* Ventana de Recuperación: Se determinó K=5 como el umbral óptimo de balance entre ganancia marginal de contexto y eficiencia de tokens (HR@5 > 81,7%).
* Modelo Seleccionado: all-MiniLM-L6-v2 se estableció como la alternativa más eficiente en la frontera de Pareto, con una latencia de aproximadamente 1,75 ms por inferencia.
* Evaluación Generativa End-to-End: Tasa de aprobados de 77,78% en la arquitectura local (ChromaDB) y 71,21% en la infraestructura distribuida en la nube (LanceDB en S3 + Neon).
* Mitigación de Latencia de Red: La coubicación de cómputo en AWS EC2 junto al almacenamiento S3 redujo el tiempo de procesamiento monohilo en un 63% frente al acceso desde redes residenciales.

## 7. Autor y Licencia
Autor: Iván García (@IvanGP7)

Grado: Trabajo de Fin de Grado en Ingeniería Informática.