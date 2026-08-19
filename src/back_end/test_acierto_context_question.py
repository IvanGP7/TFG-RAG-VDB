import pandas as pd
import lancedb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import os
from dotenv import load_dotenv
load_dotenv()

LANCEDB_LINK = os.getenv("LANCEDB_DATABASE")
RUTA_S3 = "s3://bucket-s3-tfg-8722/benchmark_aciertos_contextos.csv"
def preparar_datos_ground_truth(ruta_parquet):
    print("1. Cargando y preparando el Ground Truth desde Pandas...")
    df = pd.read_parquet(ruta_parquet)
    
    # 1. Aislamos los contextos únicos y les regeneramos el doc_id 
    # (Hacemos EXACTAMENTE lo mismo que hiciste el día de la inserción)
    df_contextos = df.drop_duplicates(subset=['context']).copy()
    df_contextos = df_contextos.reset_index(drop=True)
    df_contextos['doc_id'] = ['doc_' + str(i) for i in range(len(df_contextos))]
    
    # 2. Hacemos lo que tú sugeriste: nos quedamos solo con preguntas únicas
    df_preguntas = df.drop_duplicates(subset=['question']).copy()
    
    # 3. MAGIA DE PANDAS (Left Join): 
    # Cruzamos las preguntas con la tabla de contextos usando la columna 'context'
    # Ahora cada pregunta tendrá al lado su 'doc_id' correcto sin tener que ir a Postgres.
    df_evaluacion = pd.merge(
        df_preguntas[['question', 'context']], # Nos quedamos con lo importante
        df_contextos[['context', 'doc_id']],   # El diccionario de traducción
        on='context', 
        how='left'
    )
    
    return df_evaluacion

def ejecutar_benchmark_definitivo():
    # Asegúrate de poner la ruta a tu archivo original
    ruta_parquet = "data_test/train-00000-of-00001.parquet" 
    
    df_eval = preparar_datos_ground_truth(ruta_parquet)
    
    preguntas = df_eval['question'].tolist()
    ids_esperados = df_eval['doc_id'].tolist()
    total_preguntas = len(preguntas)
    
    print("2. Cargando modelo de Inteligencia Artificial...")
    modelo = SentenceTransformer('all-MiniLM-L6-v2')

    print("3. Conectando a Lancedb...")
    db = lancedb.connect(LANCEDB_LINK)
    tabla = db.open_table("tfg_vectores")

    print(f"4. Evaluando {total_preguntas} preguntas únicas...")
    
    aciertos_top1 = 0
    aciertos_top3 = 0
    aciertos_top5 = 0
    batch_size = 100 

    for i in tqdm(range(0, total_preguntas, batch_size), desc="Procesando", unit="lote"):
        lote_preguntas = preguntas[i : i + batch_size]
        lote_ids_esperados = ids_esperados[i : i + batch_size]
        
        vectores_lote = modelo.encode(lote_preguntas).tolist()
        
        # Comprobación mediante IDs (Matemáticamente infalible y rapidísimo)
        for j in range(len(lote_preguntas)):
            id_correcto = lote_ids_esperados[j]
            vector_actual = vectores_lote[j]
            resultados = tabla.search(vector_actual).nprobes(20).refine_factor(10).limit(5).to_list()
            ids_recuperados = [meta['id'] for meta in resultados] 
            
            if id_correcto == ids_recuperados[0]:
                aciertos_top1 += 1
            if id_correcto in ids_recuperados[:3]:
                aciertos_top3 += 1
            if id_correcto in ids_recuperados[:5]:
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
    df_resultados.to_csv(RUTA_S3, mode='a', index=False, header=True,storage_options={
        "key": os.getenv("AWS_ACCESS_KEY_ID"),
        "secret": os.getenv("AWS_SECRET_ACCESS_KEY")
    })

    print("\n" + "="*50)
    print("   RESULTADOS DEL BENCHMARK (PREGUNTAS ÚNICAS)")
    print("="*50)
    print(f"Total de preguntas únicas evaluadas: {total_preguntas:,}")
    print("-" * 50)
    print(f"[+] Hit Rate @ 1: {(aciertos_top1 / total_preguntas) * 100:.2f}%")
    print(f"[+] Hit Rate @ 3: {(aciertos_top3 / total_preguntas) * 100:.2f}%")
    print(f"[+] Hit Rate @ 5: {(aciertos_top5 / total_preguntas) * 100:.2f}%")
    print("="*50)

if __name__ == "__main__":
    ejecutar_benchmark_definitivo()