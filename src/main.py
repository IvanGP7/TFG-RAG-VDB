import os
import time
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from deepeval import evaluate
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval.metrics.utils import SingleTurnParams

from sentence_transformers import SentenceTransformer

from api_client.api_functions import api_question
from back_end.rag_functions import get_context_list_from_question as rag_model

from dotenv import load_dotenv
load_dotenv()

MAX_TEST = 4000
CLOSEST_VECTOR = 5
BATCH_SIZE = 10
MAX_REINTENTOS = 5
NAME_CSV = f"resultados_rag_{MAX_TEST}"

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_KEY")
cliente_openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
modelo_embeddings = SentenceTransformer('all-MiniLM-L6-v2')

def get_dataframe_set():
    print("1. Cargando preguntas del dataset original...")
    df_original = pd.read_parquet('data_test/train-00000-of-00001.parquet', engine='fastparquet')
    df_original = df_original.drop_duplicates(subset=['question'])
    df_muestra = df_original.sample(n=MAX_TEST, random_state=47).reset_index(drop=True)
    return df_muestra

def results_api(df_muestra):

    casos_de_prueba = []
    introduccion = "Answer briefly and concisely as you can with the fewer number of words/numbers. Question: "

    print(f"2. LLM está haciendo las respuestas de {len(df_muestra)} test...")
    for index, row in tqdm(df_muestra.iterrows(), total=len(df_muestra)):

        pregunta = row['question']
        respuesta_esperada = row['answers.text'][0] 
        
        # Obtenemos el contenido a traves de la pregunta con el parametro de vectores
        context_list = rag_model(pregunta, CLOSEST_VECTOR, modelo_embeddings)

        # Mejoramos la respuesta a traves de una introduccion
        # Consume entre 300 y 500 tokens
        result_obj = api_question(introduccion + pregunta, context_list)
        respuesta_generada = result_obj.text if result_obj != 0 else "Error"

        # Empaquetamos los resultados junto con una respueta combinada con un prompt para evitar de respuestas muy tajantes
        added_promt=f"The correct answer to the question '{pregunta}' is: {respuesta_esperada}"
        test_case = LLMTestCase(
            input=pregunta,
            actual_output=respuesta_generada,
            retrieval_context=context_list,
            expected_output=added_promt
        )
        casos_de_prueba.append(test_case)

    return casos_de_prueba

def create_csv(nombre_archivo_csv):
    ruta_archivo = f"{nombre_archivo_csv}.csv"
    
    # Comprobamos si el archivo YA existe en el disco duro
    if os.path.exists(ruta_archivo):
        print(f"[*] Archivo existente detectado: '{ruta_archivo}'. Se reanudará la escritura sobre él.")
        return ruta_archivo
    
    # Si no existe, lo creamos y le añadimos la cabecera
    columnas = ["Pregunta", "Respuesta_Generada", "Respuesta_Esperada", "Nota_Correctness", "Justificacion"]
    pd.DataFrame(columns=columnas).to_csv(ruta_archivo, mode='w', index=False, encoding='utf-8-sig')
    print(f"[*] Nuevo archivo creado: '{ruta_archivo}'.")
    return ruta_archivo


def log_error(fail_counter: int, error):
    with open("output.txt", "w", encoding="utf-8") as f:
        if fail_counter == 1:
            f.write(f"=== REPORTE DE ERRORES RAG BENCHMARK ===\n")
            f.write(f"Total de fallos capturados: {fail_counter}\n")
            f.write("=" * 50 + "\n\n")
        
        
        f.write(f"[-] Test #{error['index']}\n")
        f.write(f"    Pregunta:      {error['pregunta']}\n")
        f.write(f"    Tipo de error: {error['tipo_error']}\n")
        f.write(f"    Detalle:       {error['mensaje_error']}\n")
        f.write("-" * 50 + "\n")


def evaluacion_main():
# Obtenemos el dataset de pruebas
    df_muestra = get_dataframe_set()
    # Obtenemos los resultados y los datos junto con los contextos
    casos_de_prueba = results_api(df_muestra)

    print("\n3. OpenAI evaluando el RAG (gpt-4o-mini)...")

    # Metricas de evaluacion del LLM
    metricas = GEval(
        name="Correctness",
        # He mejorado un poco el inglés para que el Juez lo entienda a la perfección
        criteria="Determine if the 'actual output' contains the core truth of the 'expected output'. It is HIGHLY ACCEPTABLE and encouraged for the 'actual output' to include additional relevant details from the 'retrieval context'. Do NOT penalize the output for being more comprehensive, detailed, or explanatory than the expected output, as long as the core question is answered correctly.",
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT, 
            LLMTestCaseParams.EXPECTED_OUTPUT, 
            LLMTestCaseParams.RETRIEVAL_CONTEXT
        ],
        threshold=0.5,
        model="gpt-4o-mini"
    )

    # Creacion del csv para almacenar los resultados
    nombre_archivo_csv = create_csv(NAME_CSV)
    fail_counter = 0

    # Comprobamos si existe cuántas preguntas están ya guardadas en el CSV
    start_index = 0
    try:
        df_existente = pd.read_csv(nombre_archivo_csv)
        start_index = len(df_existente)
        if start_index > 0:
            print(f"[*] Reanudando automáticamente desde la pregunta {start_index + 1} (Lote {start_index // BATCH_SIZE})...")
    except Exception:
        start_index = 0

    # Bucle secuencial (Uno a uno para evitar Timeouts)
    for i in range(start_index, len(casos_de_prueba), BATCH_SIZE):

        batch = casos_de_prueba[i : i + BATCH_SIZE]

        print(f"\n" + "="*50)
        print(f"EVALUANDO BATCH {i // BATCH_SIZE} (Preguntas {i+1} a {min(i+ BATCH_SIZE, len(casos_de_prueba))})")
        print("="*50)

        reintentos = 0
        while reintentos < MAX_REINTENTOS:
            try:
                # Consume entre 1200 y 1500 tokens sleep para evitar superar los tokens por minuto
                time.sleep(2.5)
                resultados_batch = evaluate(test_cases=batch, metrics=[metricas])
                lista_resultados = resultados_batch.test_results

                # Iteramos sobre la lista real de TestResults
                for result in lista_resultados:

                    try:
                        result_data = {
                            "Pregunta": result.input,
                            "Respuesta_Generada": result.actual_output,
                            "Respuesta_Esperada": result.expected_output,
                            "Nota_Correctness": result.metrics_data[0].score,
                            "Justificacion": result.metrics_data[0].reason
                        }
                        pd.DataFrame([result_data]).to_csv(nombre_archivo_csv, mode='a', index=False, header=False, encoding='utf-8-sig')
                        print(f"   Nota guardada: {result.metrics_data[0].score:.2f}")
                    except Exception as ex_interna:
                        print(f"   Error al extraer datos: {ex_interna}")
                reintentos = MAX_REINTENTOS

            except Exception as e:
                fail_counter += 1
                error = {
                    'index': i + 1,
                    'pregunta': f"Lote {i // BATCH_SIZE} (Preguntas {i+1} a {min(i + BATCH_SIZE, len(casos_de_prueba))})",
                    'tipo_error': type(e).__name__,
                    'mensaje_error': str(e)
                }
                log_error(fail_counter, error)
                print(f"[!] Error capturado en el lote {i // BATCH_SIZE}: {type(e).__name__} - {e}")
                print(f"[*] Esperando {65 * (reintentos + 1)} segundos antes de reintentar...")
                reintentos += 1
                time.sleep(65 * reintentos)

    if nombre_archivo_csv:
        df_resultados = pd.read_csv(nombre_archivo_csv)

        print("\n" + "="*25)
        print(f"[+] ¡Datos guardados con éxito en '{nombre_archivo_csv}'!")
        print(f"[+] Preguntas totales ejecutadas: {MAX_TEST}")
        
        # Calculamos la nota media para imprimirla por terminal
        nota_media = df_resultados['Nota_Correctness'].mean()
        aprobados = len(df_resultados[df_resultados['Nota_Correctness'] >= 0.5])
        porcentaje_aprobados = (aprobados / len(df_resultados)) * 100
        print(f"[*] Nota Media General (Correctness): {nota_media:.2f} / 1.00")
        print(f"[*] Test aprobados (Correctness): {porcentaje_aprobados:.2f}")
        print(f"[*] Test fallidos por conexión (OpenAI): {fail_counter}")
        print("="*25)


            
if __name__ == "__main__":
    evaluacion_main()