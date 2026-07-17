import pandas as pd
from sqlalchemy import create_engine, text
import chromadb
from sentence_transformers import SentenceTransformer

def carga_de_datos():
    #Limpiar contextos de los datos y crear un id unico para cada fila
    print("Cargando datos desde el archivo parquet...")
    df = pd.read_parquet('data_test/train-00000-of-00001.parquet', engine='fastparquet')
    df_unicos = df.drop_duplicates(subset=['context']).copy()
    df_unicos = df_unicos[['title', 'context']]
    df_unicos['doc_id'] = [f"doc_{i}" for i in range(len(df_unicos))]
    df_prueba = df_unicos.head(50)
    return df_prueba


def conectar_a_postgresql(df_prueba: pd.DataFrame):
    #Conexion a Postgress
    print("Conectando a la base de datos PostgreSQL...")
    sql_engine = create_engine('postgresql://admin:password123@localhost:5432/tfg_dataset')

    df_prueba.to_sql('documentos_squad', sql_engine, if_exists='replace', index=False)
    
    # Aplicar clave primaria a la columna doc_id
    with sql_engine.connect() as conexion:
        conexion.execute(text('ALTER TABLE documentos_squad ADD PRIMARY KEY (doc_id);'))
        conexion.commit()

    print("Datos cargados en la base de datos PostgreSQL.")

def conectar_a_chromadb(df_prueba: pd.DataFrame):
    # Conexión a la base de datos Chromadb
    client = chromadb.HttpClient(host='localhost', port=8000)
    coleccion = client.get_or_create_collection(name='tfg_vectores')

    # Usamos el mismo modelo embedding para liberar faena dentro del contenedor
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    for index, row in df_prueba.iterrows():
        # Tranformar contexto en vectores
        vector = embedding_model.encode(row['context']).tolist()

        # Añadir el vector del context y el id de postgres a Chroma
        coleccion.add(
            ids=[row['doc_id']],
            embeddings=[vector],
            metadatas=[{
                "titulo":row['title'],
                "postgres_table": "documentos_squad",
                "postgres_id": row['doc_id']
            }]
        )

    print("Vectores guardados en ChromaDB!")


if __name__ == "__main__":
    print("Iniciando el proceso de carga de datos y conexión a bases de datos...")

    df_prueba = carga_de_datos()
    #print(df_prueba)

    conectar_a_postgresql(df_prueba)

    conectar_a_chromadb(df_prueba)

    print("Creación Arquitectura funcionando")
