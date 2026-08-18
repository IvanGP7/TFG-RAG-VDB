
import pandas as pd
from sqlalchemy import create_engine, text
import lancedb
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
load_dotenv()


SQL_LINK = os.getenv("NEON_SQL")
LANCEDB_LINK = os.getenv("LANCEDB_DATABASE")
db = lancedb.connect(LANCEDB_LINK)

def vector_question(question: str, model: SentenceTransformer):
    return model.encode([question], show_progress_bar=False).tolist()

def get_top_contexts(vector: list, number_context: int):
    
    tabla = db.open_table("tfg_vectores")
    resultados = tabla.search(vector[0]).limit(number_context).to_list()
    ids = [resultado['id'] for resultado in resultados]
    return ids

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