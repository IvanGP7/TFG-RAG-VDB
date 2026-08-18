import json
import lancedb
import os


URI_S3 = os.environ.get("LANCEDB_DATABASE", "s3://tu-bucket-tfg-vectores/tfg_vectores.lance")

print("Inicializando conexión Serverless con LanceDB en S3...")
db = lancedb.connect(URI_S3)
tabla = db.open_table("tfg_vectores")

# =====================================================================
# 2. HANDLER PRINCIPAL
# =====================================================================
def lambda_handler(event, context):
    try:
        # Extraer el body de la petición HTTP (Function URL o API Gateway)
        # Si pruebas desde la consola de AWS, a veces el body viene directamente como dict
        if isinstance(event.get('body'), str):
            body = json.loads(event['body'])
        else:
            body = event.get('body', event)
            
        vector_pregunta = body.get('vector')
        
        if not vector_pregunta:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'No se proporcionó ningún vector en la petición.'})
            }

        # -------------------------------------------------------------
        # 3. BÚSQUEDA VECTORIAL AVANZADA (IVF-PQ + Refinamiento)
        # -------------------------------------------------------------
        # Buscamos en 20 clústeres y recuperamos los vectores originales 
        # (refine_factor=10) para no perder el 10% de acierto por la compresión.
        resultados = tabla.search(vector_pregunta) \
                          .nprobes(20) \
                          .refine_factor(10) \
                          .limit(5) \
                          .to_list()
        
        # -------------------------------------------------------------
        # 4. EMPAQUETADO DE RESPUESTA
        # -------------------------------------------------------------
        # Al igual que pasaba en tus primeras pruebas con ChromaDB, 
        # el objetivo de la base de datos vectorial es devolver únicamente los IDs 
        # (ej. 'doc_14') para que luego tu sistema consulte los textos largos en Postgres.
        ids_recuperados = [meta['id'] for meta in resultados]
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json'
            },
            'body': json.dumps({
                'mensaje': 'Búsqueda exitosa',
                'ids': ids_recuperados
            })
        }
        
    except Exception as e:
        print(f"Error crítico en la ejecución: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json'
            },
            'body': json.dumps({'error': str(e)})
        }