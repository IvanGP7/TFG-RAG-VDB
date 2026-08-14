import pandas as pd
import os
import time

from reset_enviroment import limpiar_bases_de_datos
from carga_datos import carga_datos
from test_acierto_context_question import ejecutar_benchmark_contextos
from test_acierto__title_question import ejecutar_benchmark_titulos

if __name__ == "__main__":
    modelos = [
        #"all-mpnet-base-v2",
        "multi-qa-mpnet-base-dot-v1",
        "all-distilroberta-v1",
        "all-MiniLM-L12-v2",
        "multi-qa-distilbert-cos-v1",
        "all-MiniLM-L6-v2",
        "multi-qa-MiniLM-L6-cos-v1",
        "paraphrase-multilingual-mpnet-base-v2",
        "paraphrase-albert-small-v2",
        "paraphrase-multilingual-MiniLM-L12-v2",
        "paraphrase-MiniLM-L3-v2",
        "distiluse-base-multilingual-cased-v1",
        "distiluse-base-multilingual-cased-v2"      
    ]
    dataset_parquet = "data_test/train-00000-of-00001.parquet"

    for modelo_embedding in modelos:
        print(f"\n\n\n==============================\nIniciando pruebas para el modelo: {modelo_embedding}\n==============================\n\n\n")
        print(f"1. Limpiando bases de datos...")
        limpiar_bases_de_datos()

        print(f"2. Cargando datos y conectando a bases de datos...")
        inicio_carga = time.perf_counter()
        carga_datos(modelo_embedding, dataset_parquet)
        fin_carga = time.perf_counter()
        tiempo_carga = fin_carga - inicio_carga

        print(f"3. Ejecutando benchmark de aciertos por contexto...")
        inicio_benchmark_contextos = time.perf_counter()
        ejecutar_benchmark_contextos(modelo_embedding, dataset_parquet)
        fin_benchmark_contextos = time.perf_counter()
        tiempo_benchmark_contextos = fin_benchmark_contextos - inicio_benchmark_contextos

        print(f"4. Ejecutando benchmark de aciertos por título...")
        inicio_benchmark_titulos = time.perf_counter()
        ejecutar_benchmark_titulos(modelo_embedding, dataset_parquet)
        fin_benchmark_titulos = time.perf_counter()
        tiempo_benchmark_titulos = fin_benchmark_titulos - inicio_benchmark_titulos

        resultados = []
        resultados.append({
            "modelo": modelo_embedding,
            "func_tiempo_carga": tiempo_carga,
            "func_tiempo_benchmark_contextos": tiempo_benchmark_contextos,
            "func_tiempo_benchmark_titulos": tiempo_benchmark_titulos
        })

        df_resultados = pd.DataFrame(resultados)
        nombre_archivo = "benchmark_aciertos_tiempos.csv"
        archivo_existe = os.path.isfile(nombre_archivo)
        df_resultados.to_csv(nombre_archivo, mode='a', index=False, header=not archivo_existe)
