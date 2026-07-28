
import pandas as pd
from sqlalchemy import create_engine, text
import chromadb
from sentence_transformers import SentenceTransformer



def vector_question(question):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode([question]).tolist()

def get_top_5_contexts(vector):
    client = chromadb.HttpClient(host='localhost', port=8000)
    collection = client.get_collection(name='tfg_vectores')
    results = collection.query(query_embeddings=vector, n_results=5)
    return results['ids'][0]

def get_contexts_from_postgresql(top_5_ids):
    sql_engine = create_engine('postgresql://admin:password123@localhost:5432/tfg_dataset')
    query = text(f"SELECT doc_id, context FROM documentos_squad WHERE doc_id IN :ids")
    with sql_engine.connect() as connection:
        df_contexts = pd.read_sql_query(query, connection, params={"ids": tuple(top_5_ids)})
    return df_contexts