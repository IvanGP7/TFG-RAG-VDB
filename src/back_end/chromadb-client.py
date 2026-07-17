import chromadb
from chromadb.types import Collection

if __name__ == "__main__":
    # Conectamos al servidor ChromaDB
    client = chromadb.HttpClient(host='localhost', port=8000)
    # Listar todas las colecciones existentes
    colecciones = client.list_collections()
    nombres_colecciones = [col.name for col in colecciones]
    print(f"Colecciones actuales en la base de datos: {nombres_colecciones}")

    # Comprobar si la colección tiene datos
    nombre_tu_coleccion = "tfg_vectores"

    if nombre_tu_coleccion in nombres_colecciones:
        coleccion = client.get_collection(name=nombre_tu_coleccion)
        cantidad = coleccion.count()
        print(f"La colección '{nombre_tu_coleccion}' tiene {cantidad} vectores almacenados.")
    else:
        print(f"La colección '{nombre_tu_coleccion}' NO existe. ¡Tu base de datos está completamente limpia para empezar!")