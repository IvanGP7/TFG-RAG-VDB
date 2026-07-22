import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from tqdm import tqdm

if __name__ == "__main__":
    print("1. Cargando datos y aislando las primeras 50 filas...")
    # 1. Recuperar las columnas de contextos hacer drop y obtener las preguntas unicas
    df = pd.read_parquet("data_test/train-00000-of-00001.parquet", engine='fastparquet')
    df_unicos = df.drop_duplicates(subset=['context']).copy()
    
    # Tomamos las 50 primeras para coincidir con la base de datos
    df_prueba = df_unicos.copy()
    
    # Recreamos la columna doc_id EXACTAMENTE igual que hicimos al insertarlos
    df_prueba['doc_id'] = [f"doc_{i}" for i in range(len(df_prueba))]
    
    preguntas = df_prueba['question'].tolist()
    ids_esperados = df_prueba['doc_id'].tolist()
        
    print("2. Convirtiendo las 50 preguntas en embeddings (esto tomará unos segundos)...")
    # convertir en embadding las preguntas en una lista nueva
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # Añadimos .tolist() porque ChromaDB necesita listas nativas de Python
    vectores_preguntas = model.encode(preguntas).tolist()

    print("3. Conectando a ChromaDB...")
    # Conectar en chromadb
    client = chromadb.HttpClient(host='localhost', port=8000)
    # Usamos get_collection porque la base de datos ya debería existir
    coleccion = client.get_collection(name='tfg_vectores')

    print("4. Realizando búsqueda semántica masiva en ChromaDB (Top-5)...")
    
    total = len(preguntas)
    aciertos_top1 = 0
    aciertos_top3 = 0
    aciertos_top5 = 0

    for i in tqdm(range(total), desc="Evaluando preguntas", unit="preg"):
        
        # Le enviamos solo 1 vector a ChromaDB en cada iteración
        resultados = coleccion.query(
            query_embeddings=[vectores_preguntas[i]], 
            n_results=5 
        )
        
        id_correcto = ids_esperados[i]
        
        # Como solo hemos enviado 1 pregunta, los resultados están en la posición [0]
        ids_recuperados = resultados['ids'][0] 
        
        # Comprobamos si acertó en la primera opción
        if id_correcto == ids_recuperados[0]:
            aciertos_top1 += 1
            
        # Comprobamos si el correcto estaba entre los 3 primeros
        if id_correcto in ids_recuperados[:3]:
            aciertos_top3 += 1
            
        # Comprobamos si el correcto estaba entre los 5 primeros
        if id_correcto in ids_recuperados[:5]:
            aciertos_top5 += 1

    # 5. EVALUACIÓN Y HIT RATE @ K
    print("\n" + "="*40)
    print(" 📊 RESULTADOS DEL BENCHMARK (Top-1, Top-3, Top-5)")
    print("="*40)
    print(f"✅ Hit Rate @ 1 (Tasa de Acierto Top-1): {(aciertos_top1 / total) * 100}%")
    print(f"✅ Hit Rate @ 3 (Tasa de Acierto Top-3): {(aciertos_top3 / total) * 100}%")
    print(f"✅ Hit Rate @ 5 (Tasa de Acierto Top-5): {(aciertos_top5 / total) * 100}%")