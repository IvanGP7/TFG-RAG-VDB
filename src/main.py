import os
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import time

from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, ContextualRecallMetric
from deepeval.test_case import LLMTestCase

from api_client.api_functions import api_queston
from back_end.rag_functions import vector_question, get_top_5_contexts, get_contexts_from_postgresql

from dotenv import load_dotenv
load_dotenv()


os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_KEY")
cliente_openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def ejecutar_evaluacion_masiva():
    print("1. Cargando 30 preguntas del dataset original...")
    df_original = pd.read_parquet('data_test/train-00000-of-00001.parquet', engine='fastparquet')
    df_original = df_original.drop_duplicates(subset=['question'])
    df_muestra = df_original.sample(n=30, random_state=47).reset_index(drop=True)

    casos_de_prueba = []
    introduccion = "If the answer isn't clear from the context, explicitly say, “I don't have enough information”; don't make anything up. Question: "
    print("2. 🤖 Gemini está haciendo el examen lentamente para no pagar (Tardará ~7 mins)...")
    for index, row in tqdm(df_muestra.iterrows(), total=len(df_muestra)):

        pregunta = row['question']
        respuesta_esperada = row['answers.text'][0] 
        
        # --- FASE RETRIEVER ---
        v = vector_question(pregunta)
        top_5_ids = get_top_5_contexts(v)
        contexts_df = get_contexts_from_postgresql(top_5_ids)
        context_list = contexts_df['context'].tolist()
        
        # --- FASE GENERATOR ---
        result_obj = api_queston(introduccion + pregunta, context_list)
        respuesta_generada = result_obj.text if result_obj != 0 else "Error"
        
        # --- EMPAQUETADO ---
        #print(f"PREGUNTA: {pregunta}\n")
        #print(respuesta_generada)
        #for n in context_list:
        #    print(f"CONTEXTO: {n}\n")

        #print(f"RESPUESTA ESPERADA: {respuesta_esperada}\n")
        added_promt=f"The correct answer to the question '{pregunta}' is: {respuesta_esperada}"
        test_case = LLMTestCase(
            input=pregunta,
            actual_output=respuesta_generada,
            retrieval_context=context_list,
            expected_output=added_promt
        )
        casos_de_prueba.append(test_case)
        
        # ⚠️ EL FRENO ANTI-BANEOS DE GOOGLE (15 segundos = 4 peticiones/min)
        #time.sleep(15) 

    print("\n3. ⚖️ OpenAI evaluando en modo Low-Cost (gpt-4o-mini)...")
    
    # ⚠️ Forzamos a DeepEval a usar el modelo súper barato
    modelo_barato = "gpt-4o-mini"
    # 1. Instanciamos a nuestros 3 Jueces
    juez_fidelidad = FaithfulnessMetric(threshold=0.7, model=modelo_barato, async_mode=False)
    juez_relevancia = AnswerRelevancyMetric(threshold=0.7, model=modelo_barato, async_mode=False)
    juez_recall = ContextualRecallMetric(threshold=0.7, model=modelo_barato, async_mode=False)

    # Lista para guardar las notas y hacer gráficas luego
    boletin_notas = []

    # 2. EL BUCLE DE CONTROL TOTAL
    for i, caso in enumerate(casos_de_prueba):
        print(f"\n" + "="*50)
        print(f"📊 EVALUANDO PREGUNTA {i+1} DE {len(casos_de_prueba)}")
        print("="*50)
        print(f"PREGUNTA: {caso.input}")
        for n in caso.retrieval_context:
            print(f"CONTEXTO: {n}\n")
        print(f"RESPUESTA GENERADA: {caso.actual_output}")
        print(f"RESPUESTA ESPERADA: {caso.expected_output}")

        max_reintentos = 1
        for intento in range(max_reintentos):
            try:
                # Obligamos a los jueces a evaluar paso a paso
                juez_fidelidad.measure(caso)
                juez_relevancia.measure(caso)
                juez_recall.measure(caso)
                
                # Extraemos las notas exactas (0.0 a 1.0)
                print(f"✅ Fidelidad: {juez_fidelidad.score}")
                print(f"✅ Relevancia: {juez_relevancia.score}")
                print(f"✅ Recall: {juez_recall.score}")
                
                # Guardamos los resultados para nuestro TFG
                boletin_notas.append({
                    "PREGUNTA": caso.input,
                    "FIDELIDAD": juez_fidelidad.score,
                    "RELEVANCIA": juez_relevancia.score,
                    "RECALL": juez_recall.score,
                    "MOTIVO_RECALL": juez_recall.reason,
                    "RESPUESTA GENERADA": caso.actual_output,
                    "RESPUESTA ESPERADA": caso.expected_output,
                    "CONTEXTO 1": caso.retrieval_context[0],
                    "CONTEXTO 2": caso.retrieval_context[1],
                    "CONTEXTO 3": caso.retrieval_context[2],
                    "CONTEXTO 4": caso.retrieval_context[3],
                    "CONTEXTO 5": caso.retrieval_context[4]

                })
                break
            except Exception as e:
                print(f"⚠️ Intento {intento + 1} fallido por error de red: {type(e).__name__}")
                
                # Si es el último intento, nos rendimos de verdad
                if intento == max_reintentos - 1:
                    print(f"❌ Error definitivo al evaluar la pregunta {i+1}. Pasamos a la siguiente.")
                else:
                    print("⏳ Esperando 3 segundos antes de reintentar...")
                    time.sleep(3) # Requiere hacer 'import time' arriba del todo en tu archivo

    # (Opcional) Guardar las notas en un Excel para la memoria del TFG
    df_notas = pd.DataFrame(boletin_notas)
    df_notas.to_excel("resultados_evaluacion.xlsx", index=False)
    print("\n¡Resultados guardados en resultados_evaluacion.xlsx!")

if __name__ == "__main__":
    ejecutar_evaluacion_masiva()