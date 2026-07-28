import pandas as pd
from sqlalchemy import create_engine, text
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

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

    # Configuración del lote (Batch)
    batch_size = 100
    
    print(f"Iniciando inserción por lotes de {batch_size} en {batch_size}...")
    
    # tqdm crea la barra de progreso verde
    for i in tqdm(range(0, len(df_prueba), batch_size), desc="Insertando en ChromaDB"):
        lote = df_prueba.iloc[i:i+batch_size]
        
        # 1. Preparar las listas para el lote entero
        textos = lote['context'].tolist()
        ids = lote['doc_id'].tolist()
        metadatos = [{"titulo": row['title'], "postgres_table": "documentos_squad", "postgres_id": row['doc_id']} for _, row in lote.iterrows()]
        
        # 2. Calcular los embeddings de los 100 textos a la vez (mucho más eficiente)
        vectores = embedding_model.encode(textos).tolist()
        
        # 3. Insertar el lote entero en ChromaDB con una sola petición HTTP
        coleccion.add(
            ids=ids,
            embeddings=vectores,
            metadatas=metadatos
        )

    print("Vectores guardados en ChromaDB!")


if __name__ == "__main__":
    print("Iniciando el proceso de carga de datos y conexión a bases de datos...")

    df_prueba = carga_de_datos()
    #print(df_prueba)

    conectar_a_postgresql(df_prueba)

    conectar_a_chromadb(df_prueba)

    print("Creación Arquitectura funcionando")
