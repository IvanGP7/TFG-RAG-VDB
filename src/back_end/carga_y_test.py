import pandas as pd
import os
import time

from reset_enviroment import limpiar_bases_de_datos
from carga_datos import carga_de_datos_y_conectar
from test_acierto_context_question import ejecutar_benchmark_definitivo

import time

if __name__ == "__main__":
    print("Iniciando el proceso de limpieza de bases de datos...")
    limpiar_bases_de_datos()


    print("\nIniciando el proceso de carga de datos y conexión a bases de datos...")
    inicio_carga = time.perf_counter()
    carga_de_datos_y_conectar()
    fin_carga = time.perf_counter()
    tiempo_carga = fin_carga - inicio_carga


    print("\nIniciando el proceso de evaluación del modelo...")
    inicio_benchmark_contextos = time.perf_counter()
    ejecutar_benchmark_definitivo()
    fin_benchmark_contextos = time.perf_counter()
    tiempo_benchmark_contextos = fin_benchmark_contextos - inicio_benchmark_contextos


    resultados = []
    resultados.append({
        "modelo": 'all-MiniLM-L6-v2',
        "func_tiempo_carga": tiempo_carga,
        "func_tiempo_benchmark_contextos": tiempo_benchmark_contextos
    })

    df_resultados = pd.DataFrame(resultados)
    nombre_archivo = "benchmark_aciertos_tiempos.csv"
    archivo_existe = os.path.isfile(nombre_archivo)
    df_resultados.to_csv(nombre_archivo, mode='a', index=False, header=not archivo_existe)