import pandas as pd
import time
# Modelo
from sentence_transformers import SentenceTransformer

def benchmark_embeddings(ruta_parquet, modelo_nombre):
    
    df = pd.read_parquet(ruta_parquet, engine='fastparquet')
    textos = df['context'].head(1000).tolist()
    
    # 2. Cargar modelo (no medimos el tiempo de carga del modelo en sí)
    modelo = SentenceTransformer(modelo_nombre)
    
    resultados = []
    
    # 3. Medir el tiempo de vectorización (Batch o individual)
    inicio = time.perf_counter()
    vectores = modelo.encode(textos)
    fin = time.perf_counter()
    
    tiempo_total = fin - inicio
    tiempo_por_texto = tiempo_total / len(textos)
    
    # 4. Guardar métricas
    resultados.append({
        "modelo": modelo_nombre,
        "textos_procesados": len(textos),
        "tiempo_total_seg": round(tiempo_total, 4),
        "tiempo_medio_por_texto_seg": round(tiempo_por_texto, 4)
    })
    
    df_resultados = pd.DataFrame(resultados)
    # mode='a' para ir añadiendo resultados de otros modelos sin borrar los anteriores
    df_resultados.to_csv("benchmark_embeddings.csv", mode='a', index=False, header=True)
    print(f"[+] Test Embedding completado en {tiempo_total:.2f}s")


if __name__ == '__main__':
    benchmark_embeddings("data_test/train-00000-of-00001.parquet", "all-MiniLM-L6-v2")