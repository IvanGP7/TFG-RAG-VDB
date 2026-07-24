import chromadb
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text
import pandas as pd
from google import genai
from config import API_KEY

def vector_question(question):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    vector = model.encode([question])
    return vector

def get_top_5_contexts(vector):
    print("Iniciando Chromadb...")
    client = chromadb.HttpClient(host='localhost', port=8000)
    collection = client.get_collection(name='tfg_vectores')
    results = collection.query(
        query_embeddings=vector,
        n_results=5
    )
    return results['ids'][0]


def get_contexts_from_postgresql(top_5_ids):
    print("Conectado a la base de datos PostgreSQL...")
    # Conexión a la base de datos PostgreSQL
    sql_engine = create_engine('postgresql://admin:password123@localhost:5432/tfg_dataset')

    # Crear una cadena de consulta SQL para obtener los contextos correspondientes a los IDs
    query = text(f"SELECT doc_id, context FROM documentos_squad WHERE doc_id IN :ids")
    
    # Ejecutar la consulta y obtener los resultados en un DataFrame
    with sql_engine.connect() as connection:
        df_contexts = pd.read_sql_query(query, connection, params={"ids": tuple(top_5_ids)})

    return df_contexts

def api_queston(question, context: list):
    client = genai.Client(api_key=API_KEY)
    l = [question]
    peticion = l + context
    peticion = " ".join(peticion)
    try:
        print("\nConexión con la API...")
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=peticion
        )
        return response

    except Exception as e:
        print(f"\n[!] Error con API: {e}")
    return 0

if __name__ == "__main__":
    # To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?
    #question = input("Enter your question: ")
    question = "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?"

    # Vectorizamos la pregunta
    v = vector_question(question)

    # Obtenemos los 5 contextos más similares de ChromaDB
    top_5_ids = get_top_5_contexts(v)

    # Obtenemos los contextos completos de la base de datos PostgreSQL
    contexts = get_contexts_from_postgresql(top_5_ids)
    l = contexts['context'].tolist()

    # Pregunta a la API
    result = api_queston(question, l)
    print(f"Resultado:\n{result.text}")