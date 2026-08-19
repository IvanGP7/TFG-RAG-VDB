import pandas as pd
from sqlalchemy import create_engine, text
import lancedb
from lancedb.index import IvfPq
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import os
from dotenv import load_dotenv
load_dotenv()


SQL_LINK = os.getenv("NEON_SQL")
LANCEDB_LINK = os.getenv("LANCEDB_DATABASE")
def carga_de_datos():
    #Limpiar contextos de los datos y crear un id unico para cada fila
    print("Cargando datos desde el archivo parquet...")
    df = pd.read_parquet('data_test/train-00000-of-00001.parquet', engine='fastparquet')
    df_unicos = df.drop_duplicates(subset=['context']).copy()
    df_unicos = df_unicos[['title', 'context']]
    df_unicos['doc_id'] = [f"doc_{i}" for i in range(len(df_unicos))]
    df_prueba = df_unicos
    return df_prueba


def conectar_a_postgresql(df_prueba: pd.DataFrame):
    #Conexion a Postgress
    print("Conectando a la base de datos LanceDB...")
    sql_engine = create_engine(SQL_LINK)

    df_prueba.to_sql('documentos_squad', sql_engine, if_exists='replace', index=False)
    
    # Aplicar clave primaria a la columna doc_id
    with sql_engine.connect() as conexion:
        conexion.execute(text('ALTER TABLE documentos_squad ADD PRIMARY KEY (doc_id);'))
        conexion.commit()

    print("Datos cargados en la base de datos PostgreSQL.")

def conectar_a_lancedb(df_prueba: pd.DataFrame):
    # Conexión a la base de datos Chromadb
    database = lancedb.connect(LANCEDB_LINK)

    # Usamos el mismo modelo embedding para liberar faena dentro del contenedor
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Configuración del lote (Batch)
    batch_size = 100
    datos_para_insertar = []

    print(f"Iniciando inserción por lotes de {batch_size} en {batch_size}...")
    
    # tqdm crea la barra de progreso verde
    for i in tqdm(range(0, len(df_prueba), batch_size), desc="Vectorizando valores"):
        lote = df_prueba.iloc[i:i+batch_size]
        textos = lote['context'].tolist()
        ids = lote['doc_id'].tolist()
        titulos = lote['title'].tolist()
        
        # 2. Calcular los embeddings de los 100 textos a la vez (mucho más eficiente)
        vectores = embedding_model.encode(textos).tolist()
        
        for j in range(len(lote)):
            datos_para_insertar.append({
                        "vector": vectores[j],
                        "id":ids[j],
                        "titulo":titulos[j],
                        "texto": textos[j]
                    })
                    
    print("Insertando datos en LanceDB")
    try:
        tabla = database.create_table("tfg_vectores", data=datos_para_insertar)
        print("-> Construyendo índice vectorial (ANN)...")
        tabla.create_index("vector", config=IvfPq(distance_type="cosine"))
    except Exception as e:
        print(f"[x] Error al interactuar con LanceDB: {e}")
    
    print(f"¡Insertados {len(datos_para_insertar)} vectores en LanceDB local!")


def carga_de_datos_y_conectar():
    print("Iniciando el proceso de carga de datos y conexión a bases de datos...")

    df_prueba = carga_de_datos()
    #print(df_prueba)

    conectar_a_postgresql(df_prueba)

    conectar_a_lancedb(df_prueba)

    print("Creación Arquitectura funcionando")

if __name__ == "__main__":
    carga_de_datos_y_conectar()