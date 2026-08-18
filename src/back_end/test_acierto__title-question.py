import pandas as pd
import lancedb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import os
from dotenv import load_dotenv
load_dotenv()

LANCEDB_LINK = os.getenv("LANCEDB_DATABASE")

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

    print("3. Conectando a Lancedb...")
    db = lancedb.connect(LANCEDB_LINK)
    tabla = db.open_table("tfg_vectores")

    print(f"4. Evaluando {total_preguntas} preguntas únicas por TÍTULO...")
    
    aciertos_top1 = 0
    aciertos_top3 = 0
    aciertos_top5 = 0
    batch_size = 100 

    for i in tqdm(range(0, total_preguntas, batch_size), desc="Procesando", unit="lote"):
        lote_preguntas = preguntas[i : i + batch_size]
        lote_titulos_esperados = titulos_esperados[i : i + batch_size]
        
        vectores_lote = modelo.encode(lote_preguntas).tolist()
        
        # Evaluamos extrayendo los metadatos
        for j in range(len(lote_preguntas)):
            titulo_correcto = lote_titulos_esperados[j]
            vector_actual = vectores_lote[j]
            resultados = tabla.search(vector_actual).nprobes(10).limit(5).to_list()
            
            titulos_recuperados = [meta['titulo'] for meta in resultados]
            
            # Comparamos Strings (Título esperado vs Títulos recuperados)
            if titulo_correcto == titulos_recuperados[0]:
                aciertos_top1 += 1
            if titulo_correcto in titulos_recuperados[:3]:
                aciertos_top3 += 1
            if titulo_correcto in titulos_recuperados[:5]:
                aciertos_top5 += 1
    # Guardar Resultados
    resultados = []
    resultados.append({
    "Total_Preguntas": total_preguntas,
    "Hit_Rate_1": f"{(aciertos_top1 / total_preguntas) * 100:.2f}%",
    "Hit_Rate_2": f"{(aciertos_top3 / total_preguntas) * 100:.2f}%",
    "Hit_Rate_3": f"{(aciertos_top5 / total_preguntas) * 100:.2f}%"
    })
    
    df_resultados = pd.DataFrame(resultados)
    # mode='a' para ir añadiendo resultados de otros modelos sin borrar los anteriores
    df_resultados.to_csv("benchmark_aciertos_titulo.csv", mode='a', index=False, header=True)

    print("\n" + "="*50)
    print("  RESULTADOS DEL BENCHMARK (POR TÍTULO DE WIKIPEDIA)")
    print("="*50)
    print(f"Total de preguntas únicas evaluadas: {total_preguntas:,}")
    print("-" * 50)
    print(f"[+] Document Hit Rate @ 1: {(aciertos_top1 / total_preguntas) * 100:.2f}%")
    print(f"[+] Document Hit Rate @ 3: {(aciertos_top3 / total_preguntas) * 100:.2f}%")
    print(f"[+] Document Hit Rate @ 5: {(aciertos_top5 / total_preguntas) * 100:.2f}%")
    print("="*50)

if __name__ == "__main__":
    ejecutar_benchmark_por_titulos()