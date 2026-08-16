import chromadb
import psycopg2
import sys

def limpiar_bases_de_datos():
    print("ADVERTENCIA: Este script borrará TODOS los datos insertados.")
    #confirmacion = input("¿Estás seguro de que quieres limpiar el entorno? (s/n): ")
    
    #if confirmacion.lower() != 's':
    #    print("Operación cancelada. Tus datos están a salvo.")
    #    sys.exit()

    # 1. LIMPIAR CHROMADB (Contenedor Docker)
    try:
        print("\nConectando a ChromaDB...")
        cliente_chroma = chromadb.HttpClient(host='localhost', port=8000)
        colecciones = cliente_chroma.list_collections()
        
        if not colecciones:
            print(" -> ChromaDB ya está vacío.")
        else:
            for coleccion in colecciones:
                cliente_chroma.delete_collection(name=coleccion.name)
                print(f" -> Colección vectorial eliminada: {coleccion.name}")
    except Exception as e:
        print(f"Error al limpiar ChromaDB: {e}")

    # 2. LIMPIAR POSTGRESQL (Contenedor Docker)
    try:
        print("\nConectando a PostgreSQL...")
        # Usa las credenciales que definiste en tu docker-compose.yml
        conexion_pg = psycopg2.connect(
            dbname="tfg_dataset",
            user="admin",
            password="password123",
            host="localhost",
            port="5432"
        )
        cursor = conexion_pg.cursor()
        
        nombre_tabla = "documentos_squad" 
        
        # TRUNCATE borra los datos rapidísimo y RESTART IDENTITY pone los IDs a 0 de nuevo
        cursor.execute(f"TRUNCATE TABLE {nombre_tabla} RESTART IDENTITY CASCADE;")
        conexion_pg.commit()
        
        cursor.close()
        conexion_pg.close()
        print(f" -> Tabla relacional '{nombre_tabla}' vaciada y contadores reiniciados.")
        
    except Exception as e:
        print(f"Error al limpiar PostgreSQL (¿Pusiste el nombre de tabla correcto?): {e}")

    print("\n[+] Limpieza completada. Tienes un entorno 'limpio' para tu próximo Benchmark.")

if __name__ == "__main__":
    limpiar_bases_de_datos()