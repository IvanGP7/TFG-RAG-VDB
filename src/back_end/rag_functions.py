
import pandas as pd
from sqlalchemy import create_engine, text
import chromadb
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
load_dotenv()


SQL_LINK = os.getenv("NEON_SQL")
client = chromadb.HttpClient(host='localhost', port=8000)

def vector_question(question: str, model: SentenceTransformer):
    return model.encode([question], show_progress_bar=False).tolist()

def get_top_contexts(vector: list, number_context: int):
    
    collection = client.get_collection(name='tfg_vectores')
    results = collection.query(query_embeddings=vector, n_results=number_context)
    return results['ids'][0]

def get_contexts_from_postgresql(top_ids):
    sql_engine = create_engine(SQL_LINK)
    query = text(f"SELECT doc_id, context FROM documentos_squad WHERE doc_id IN :ids")
    with sql_engine.connect() as connection:
        df_contexts = pd.read_sql_query(query, connection, params={"ids": tuple(top_ids)})
    return df_contexts

def get_context_list_from_question(question: str, number_context: int, model: SentenceTransformer):
    v = vector_question(question, model)
    top_ids = get_top_contexts(v, number_context)
    contexts_df = get_contexts_from_postgresql(top_ids)
    context_list = contexts_df['context'].tolist()
    return context_list