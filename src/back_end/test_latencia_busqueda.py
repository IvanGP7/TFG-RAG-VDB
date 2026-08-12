import pandas as pd
import time
import chromadb
from sentence_transformers import SentenceTransformer

def benchmark_latencia(ruta_parquet, db_nombre, modelo_nombre, n_preguntas=100):
    # 1. Cargar preguntas de prueba
    df = pd.read_parquet(ruta_parquet)
    preguntas_test = df['question'].head(n_preguntas).tolist()
    
    # 2. Configurar cliente BBDD (ChromaDB en tu Docker local)
    cliente = chromadb.HttpClient(host='localhost', port=8000)
    coleccion = cliente.get_or_create_collection(name='test_vectores')
    
    # 3. Inicializar el modelo de IA local
    print(f"Cargando modelo {modelo_nombre}...")
    modelo = SentenceTransformer(modelo_nombre)
    
    resultados = []
    print(f"Iniciando test de latencia para {n_preguntas} búsquedas...")
    
    # 4. Bucle de búsquedas
    for i, pregunta in enumerate(preguntas_test):
        # --- CRONÓMETRO GLOBAL ---
        inicio_total = time.perf_counter()
        
        # PASO A: Vectorizar la pregunta (Medimos el coste computacional)
        inicio_embedding = time.perf_counter()
        # ChromaDB exige formato de lista de Python, así que convertimos el numpy array
        vector_pregunta = modelo.encode(pregunta).tolist()
        fin_embedding = time.perf_counter()
        
        # PASO B: Buscar en BBDD (Medimos el Overhead real de Red + Búsqueda)
        inicio_db = time.perf_counter()
        resultados_db = coleccion.query(
            query_embeddings=[vector_pregunta],
            n_results=5
        )
        fin_db = time.perf_counter()
        
        fin_total = time.perf_counter()
        
        # 5. Cálculos de tiempos en milisegundos
        latencia_embedding_ms = (fin_embedding - inicio_embedding) * 1000
        latencia_db_ms = (fin_db - inicio_db) * 1000
        latencia_total_ms = (fin_total - inicio_total) * 1000
        
        resultados.append({
            "base_de_datos": db_nombre,
            "modelo": modelo_nombre,
            "id_pregunta": i,
            "latencia_embedding_ms": round(latencia_embedding_ms, 2),
            "latencia_db_ms": round(latencia_db_ms, 2),
            "latencia_total_ms": round(latencia_total_ms, 2)
        })
    
    # 6. Guardado de métricas
    df_resultados = pd.DataFrame(resultados)
    df_resultados.to_csv("benchmark_latencia.csv", mode='a', index=False, header=True)
    
    # 7. Resumen por consola
    latencia_media_total = df_resultados['latencia_total_ms'].mean()
    latencia_media_db = df_resultados['latencia_db_ms'].mean()
    
    print(f"[+] Test Búsqueda finalizado.")
    print(f"   Latencia MEDIA TOTAL: {latencia_media_total:.2f} ms")
    print(f"   Latencia MEDIA BBDD (Overhead puro): {latencia_media_db:.2f} ms")

if __name__ == '__main__':
    # Sustituye la ruta por tu parquet real y el modelo que uses actualmente
    benchmark_latencia(
        ruta_parquet="data_test/train-00000-of-00001.parquet",
        db_nombre="Chroma_Docker",
        modelo_nombre="all-MiniLM-L6-v2",
        n_preguntas=1000
    )