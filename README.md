# Optimización y despliegue cloud de sistemas RAG (TFG-RAG-VDB)

Repositorio oficial del Trabajo de Fin de Grado en Ingeniería Informática y Telemática. Este proyecto comprende el diseño, implementación, evaluación empírica y optimización de una arquitectura desacoplada para sistemas de Generación Aumentada por Recuperación (RAG), transitando desde un entorno local hasta una infraestructura distribuida en la nube con Amazon Web Services (AWS) y Neon PostgreSQL.

---

## 1. Arquitectura y Variantes de Despliegue

El repositorio se estructura en tres variantes de ejecución correspondientes a las fases de experimentación del proyecto:

1. **RAG Local (Línea Base):**
   - **Persistencia Relacional:** PostgreSQL ejecutado en local (Docker) para textos y títulos.
   - **Índice Vectorial:** ChromaDB en memoria/local para búsqueda de identificadores (`doc_id`).
   - **Cómputo:** Procesamiento e inferencia íntegramente en la máquina de desarrollo.

2. **RAG Serverless (Híbrido - Local / Nube):**
   - **Persistencia Relacional:** Neon PostgreSQL (Serverless en la región de Frankfurt).
   - **Índice Vectorial:** LanceDB persistido sobre almacenamiento de objetos en Amazon S3, utilizando índices aproximados cuantizados (IVF-PQ) y refinamiento (*Refine Factor*).
   - **Cómputo:** Procesamiento en la máquina de desarrollo local interactuando mediante peticiones remotas a AWS S3 y Neon.

3. **RAG Serverless en AWS EC2 (Infraestructura Distribuida):**
   - **Persistencia Relacional:** Neon PostgreSQL en la nube.
   - **Índice Vectorial:** LanceDB persistido en Amazon S3.
   - **Cómputo:** Instancia virtual AWS EC2 (Ubuntu) coubicada en la misma región que el *bucket* de S3, optimizada con hilos y procesos (*workers*) para eliminar la sobrecarga de red doméstica.

---

## 2. Requisitos Previos

- Python 3.11 o superior.
- Docker y Docker Compose (requerido para la variante RAG Local).
- Cuenta en Neon.tech para la base de datos PostgreSQL Serverless.
- Cuenta de AWS con un bucket en Amazon S3 y par de claves SSH (`.pem`) para acceso a EC2.
- Claves de API activas para OpenAI (`OPENAI_API_KEY`) y Google GenAI (`GEMINI_API_KEY`).

---

## 3. Configuración de Variables de Entorno (`.env`)

Crea un archivo `.env` en la raíz del entorno con las siguientes variables:

```ini
# Credenciales APIs de LLMs
OPENAI_API_KEY="tu_openai_api_key"

# Conexión Relacional (Local o Neon Serverless)
NEON_DATABASE_URL="postgresql://usuario:contraseña@servidor.neon.tech/neondb?sslmode=require"

# Almacenamiento Vectorial y AWS S3
AWS_ACCESS_KEY_ID="tu_aws_access_key"
AWS_SECRET_ACCESS_KEY="tu_aws_secret_key"
AWS_REGION="eu-central-1"
LANCEDB_DATABASE="s3://nombre-de-tu-bucket"
```
## 4\. Guía de Ejecución

### 4.1. Ejecución en Entorno Local (RAG Local)

1.  Crear y activar el entorno virtual:  
```PowerShell
python -m venv .venv  
.\\.venv\\Scripts\\Activate.ps1  
```
2.  Instalar dependencias:  
```PowerShell
pip install -r requirements.txt  
```
3.  Iniciar servicios en Docker y ejecutar el pipeline:  
```PowerShell
python src/main.py  
```  

### 4.2. Ejecución en Entorno Serverless Local (Híbrido)

1.  Configurar en el archivo `.env` la URI del bucket S3 (`LANCEDB_DATABASE`) y la URL de Neon PostgreSQL.
2.  Ejecutar el pipeline desde la terminal local:  
```PowerShell
python src/main.py  
```

### 4.3. Despliegue y Ejecución en AWS EC2

Para ejecutar el procesamiento en la nube y mitigar el retardo de la red doméstica:

1.  ****Transferir el paquete a la instancia EC2:****  
    Comprimir el repositorio en `serverless_code.zip` (excluyendo el directorio `.venv`) y enviarlo mediante SCP:  
```PowerShell
scp -i "C:\\ruta\\a\\claves-tfg.pem" serverless\_code.zip ubuntu@<IP\_PUBLICA\_EC2>:/home/ubuntu/  
```
2.  ****Conectarse a la máquina EC2:****  
```PowerShell
ssh -i "C:\\ruta\\a\\claves-tfg.pem" ubuntu@<IP\_PUBLICA\_EC2>  
```
3.  ****Descomprimir e instalar dependencias en Ubuntu:****  
```Bash 
unzip serverless\_code.zip -d ~/TFG-RAG-VDB  
cd ~/TFG-RAG-VDB  
python3 -m venv .venv  
source .venv/bin/activate  
pip install -r requirements.txt  
```
4.  ****Configurar el entorno y ejecutar:****  
    Crear el archivo `.env` en la máquina con las credenciales y lanzar la ejecución:  
```Bash
python src/main.py  
```
__(Para ejecuciones en segundo plano desacopladas de la sesión SSH:__ _`_nohup python src/main.py > salida.log 2>&1 &_`___)__

## 5\. Resultados Destacados

-   ****Ventana de Contexto:**** $K=5$ se determinó como el límite óptimo de recuperación ($HR@5 > 81{,}7\\%$).
-   ****Eficiencia del Modelo:**** `all-MiniLM-L6-v2` se consolidó como el más eficiente con una latencia media de 1,75 ms por consulta.
-   ****Reducción de Latencia en Cloud:**** El procesamiento de consultas masivas desde AWS EC2 junto a Amazon S3 redujo el tiempo de búsqueda en un 63% frente a la conexión local.
-   ****Rendimiento Generativo (End-to-End):**** Tasa de aprobación en DeepEval (`gpt-4o-mini`, umbral $\\ge 0{,}50$) del 77,78% en local y 71,21% en la nube.