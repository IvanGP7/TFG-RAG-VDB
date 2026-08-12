import pandas as pd
import time
import chromadb

def benchmark_insercion(ruta_parquet, db_nombre, tamaño_lote=100):
    # 1. Preparar datos
    df = pd.read_parquet(ruta_parquet).head(1000)
    
    # 2. Configurar cliente de BBDD
    cliente = chromadb.HttpClient(host='localhost', port=8000)
    coleccion = cliente.get_or_create_collection(name='test_vectores')
    
    resultados = []
    
    # 3. Medir tiempo de inserción por lotes
    inicio_total = time.perf_counter()
    
    # AQUI ESTÁ LA MAGIA DEL LOTE (BATCHING)
    # Recorremos el DataFrame dando saltos del tamaño del lote
    for i in range(0, len(df), tamaño_lote):
        # Extraemos solo el trozo correspondiente (ej. del 0 al 100, 100 al 200...)
        lote_df = df.iloc[i : i + tamaño_lote]
        
        # ChromaDB necesita listas, así que transformamos las columnas
        # (Asegúrate de que 'context' es el nombre real de la columna en tu parquet)
        documentos = lote_df['context'].tolist() 
        
        # Creamos IDs únicos para este lote (pasando el índice a String)
        ids = [str(idx) for idx in lote_df.index]
        
        # Insertamos este bloque específico en ChromaDB
        coleccion.add(
            documents=documentos,
            ids=ids
        )
    
    fin_total = time.perf_counter()
    
    tiempo_total = fin_total - inicio_total
    
    # 4. Guardamos las métricas
    resultados.append({
        "base_de_datos": db_nombre,
        "registros_insertados": len(df),
        "tamaño_lote": tamaño_lote,
        "tiempo_total_insercion_seg": round(tiempo_total, 4),
        "registros_por_segundo": round(len(df) / tiempo_total, 2)
    })
    
    df_resultados = pd.DataFrame(resultados)
    df_resultados.to_csv("benchmark_insercion.csv", mode='a', index=False, header=True)
    print(f"[+] Test Inserción {db_nombre} completado en {tiempo_total:.2f}s | {len(df)/tiempo_total:.2f} reg/s")

if __name__ == '__main__':
    # OJO AQUÍ: En tu código pusiste 1000 como segundo parámetro. 
    # El segundo parámetro es el nombre (db_nombre), el tamaño_lote es el tercero.
    
    # Prueba 1: Lotes de 100
    benchmark_insercion("data_test/train-00000-of-00001.parquet", "Chroma_Docker", tamaño_lote=100)
    
    # Prueba 2: Lotes de 500 (¡Podrás ver la diferencia en el CSV!)
    benchmark_insercion("data_test/train-00000-of-00001.parquet", "Chroma_Docker", tamaño_lote=500)