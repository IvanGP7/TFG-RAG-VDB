from google import genai
from config import API_KEY

client = genai.Client(api_key=API_KEY)

peticion='Hola escribe si estas recibiendo mi peticion con un si o un no.'

try:
    print("\nConexión con la API...")
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=peticion
    )
    print("\nRespuesta:\n")
    print(response.text)

except Exception as e:
    print(f"\n[!] Error con API: {e}")