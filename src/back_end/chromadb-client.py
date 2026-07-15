import string
import chromadb
from chromadb.types import Collection

def add_data_to_chromadb(collection: Collection):
    # Chroma convertirá automáticamente el texto en vectores/embeddings
    collection.add(
        documents=["El cielo es azul y bonito", "Me encanta comer pizza de queso", "El mar es profundo y azul"],
        metadatas=[{"tema": "naturaleza"}, {"tema": "comida"}, {"tema": "naturaleza"}],
        ids=["doc1", "doc2", "doc3"]
    )
    print("Datos insertados!")
    

if __name__ == "__main__":
    # Conectamos al servidor ChromaDB
    client = chromadb.HttpClient(host='localhost', port=8000)

    # Creamos o obtenemos una colección llamada "my_collection"
    collection = client.get_or_create_collection("my_collection")

    #add_data_to_chromadb(collection)

    # Realizamos busqueda semántica
    resultado = collection.query(
        query_texts=["Me gusta observar el oceano"],
        n_results=1
    )

    print("Resultado de la búsqueda:")
    for key in resultado:
        print(key)
        if resultado[key] is not None:
            for value in resultado[key]:
                if value is not dict:
                    print("- ", value)
                else:
                    for item in value:
                        print("- ", item)
        else:
            print("- None")