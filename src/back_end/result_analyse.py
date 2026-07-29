import pandas as pd
from pandas import DataFrame
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from rag_functions import get_context_list_from_question



if __name__ == '__main__':
    result_file = r'C:\Users\playe\Documents\Trabajos\TFG-RAG-VDB\src\Resulsts\resultados_rag_1000_2.csv'
    
    df = pd.read_csv(result_file)
    df_suspensos = df[df['Nota_Correctness'] < 0.5]
    df_questions = df_suspensos['Pregunta'].copy().tolist()


    parquet_file=r'C:\Users\playe\Documents\Trabajos\TFG-RAG-VDB\data_test\train-00000-of-00001.parquet'
    df_all = pd.read_parquet(parquet_file, engine='fastparquet')
    df_all.set_index('question', inplace=True)

    list_correct = []
    list_not_correct = []

    model = SentenceTransformer('all-MiniLM-L6-v2')

    for question in tqdm(df_questions, total=len(df_questions)):
        contextos = get_context_list_from_question(question, 5, model)
        correct_context = df_all.loc[question]['context']
        if correct_context in contextos:
            list_correct.append(question)
        else:
            list_not_correct.append(question)

    print(f"Numero de preguntas correctas {len(list_correct)} de {len(df_questions)}")