import os
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

MAX_TEST = 5
CLOSEST_VECTOR = 5
BATCH_SIZE = 10

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

    csv_data = []
    
    # Bucle secuencial (Uno a uno para evitar Timeouts)
    for i in range(0, len(casos_de_prueba), BATCH_SIZE):

        batch = casos_de_prueba[i : i + BATCH_SIZE]

        print(f"\n" + "="*50)
        print(f"EVALUANDO BATCH {i // BATCH_SIZE} (Preguntas {i+1} a {min(i+ BATCH_SIZE, len(casos_de_prueba))})")
        print("="*50)
        
        try:
            resultados_batch = evaluate(test_cases=batch, metrics=[metricas])
            lista_resultados = resultados_batch.test_results

            # Iteramos sobre la lista real de TestResults
            for result in lista_resultados:

                try:
                    csv_data.append({
                        "Pregunta": result.input,
                        "Respuesta_Generada": result.actual_output,
                        "Respuesta_Esperada": result.expected_output,
                        "Nota_Correctness": result.metrics_data[0].score,
                        "Justificacion": result.metrics_data[0].reason
                    })
                    print(f"   Nota guardada: {result.metrics_data[0].score:.2f}")
                except Exception as ex_interna:
                    print(f"   Error al extraer datos: {ex_interna}")
            
        except Exception as e:
            print(f"[!] Error al evaluar la pregunta {i+1}: {e}")

    if csv_data:
        df_resultados = pd.DataFrame(csv_data)
        
        # utf-8-sig garantiza que Excel abra el archivo mostrando las tildes y ñ correctamente
        nombre_archivo = "resultados_rag.csv"
        df_resultados.to_csv(nombre_archivo, index=False, encoding='utf-8-sig')
        
        print("\n" + "="*25)
        print(f"[+] ¡Datos guardados con éxito en '{nombre_archivo}'!")
        
        # Calculamos la nota media para imprimirla por terminal
        nota_media = df_resultados['Nota_Correctness'].mean()
        print(f"[*] Nota Media General (Correctness): {nota_media:.2f} / 1.00")
        print("="*25)

if __name__ == "__main__":
    evaluacion_main()