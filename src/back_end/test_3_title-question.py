import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def preparar_datos_titulos(ruta_parquet):
    print("1. Cargando datos originales para extraer preguntas y sus TÍTULOS esperados...")
    df = pd.read_parquet(ruta_parquet)
    
    # Simplemente quitamos las preguntas repetidas. 
    # Nos quedamos con la pregunta y el título de Wikipedia al que pertenece
    df_evaluacion = df.drop_duplicates(subset=['question']).copy()
    
    return df_evaluacion

def ejecutar_benchmark_por_titulos():
    ruta_parquet = "data_test/train-00000-of-00001.parquet" 
    
    df_eval = preparar_datos_titulos(ruta_parquet)
    
    preguntas = df_eval['question'].tolist()
    # En lugar de IDs, ahora nuestra respuesta correcta ("Ground Truth") es el Título
    titulos_esperados = df_eval['title'].tolist()
    total_preguntas = len(preguntas)
    
    print("2. Cargando modelo de Inteligencia Artificial...")
    modelo = SentenceTransformer('all-MiniLM-L6-v2')

    print("3. Conectando a ChromaDB...")
    client = chromadb.HttpClient(host='localhost', port=8000)
    coleccion = client.get_collection(name='tfg_vectores')

    print(f"4. Evaluando {total_preguntas} preguntas únicas por TÍTULO...")
    
    aciertos_top1 = 0
    aciertos_top3 = 0
    aciertos_top5 = 0
    batch_size = 100 

    for i in tqdm(range(0, total_preguntas, batch_size), desc="Procesando", unit="lote"):
        lote_preguntas = preguntas[i : i + batch_size]
        lote_titulos_esperados = titulos_esperados[i : i + batch_size]
        
        vectores_lote = modelo.encode(lote_preguntas).tolist()
        
        resultados = coleccion.query(
            query_embeddings=vectores_lote,
            n_results=5
        )
        
        # Evaluamos extrayendo los metadatos
        for j in range(len(lote_preguntas)):
            titulo_correcto = lote_titulos_esperados[j]
            
            # SINTAXIS CORRECTA DE CHROMADB: 
            # Sacamos el valor 'titulo' de los 5 diccionarios de metadatos devueltos para esta pregunta
            titulos_recuperados = [meta['titulo'] for meta in resultados['metadatas'][j]]
            
            # Comparamos Strings (Título esperado vs Títulos recuperados)
            if titulo_correcto == titulos_recuperados[0]:
                aciertos_top1 += 1
            if titulo_correcto in titulos_recuperados[:3]:
                aciertos_top3 += 1
            if titulo_correcto in titulos_recuperados[:5]:
                aciertos_top5 += 1

    print("\n" + "="*50)
    print(" 📊 RESULTADOS DEL BENCHMARK (POR TÍTULO DE WIKIPEDIA)")
    print("="*50)
    print(f"Total de preguntas únicas evaluadas: {total_preguntas:,}")
    print("-" * 50)
    print(f"✅ Document Hit Rate @ 1: {(aciertos_top1 / total_preguntas) * 100:.2f}%")
    print(f"✅ Document Hit Rate @ 3: {(aciertos_top3 / total_preguntas) * 100:.2f}%")
    print(f"✅ Document Hit Rate @ 5: {(aciertos_top5 / total_preguntas) * 100:.2f}%")
    print("="*50)

if __name__ == "__main__":
    ejecutar_benchmark_por_titulos()