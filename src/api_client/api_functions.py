import os
from openai import OpenAI
from tqdm import tqdm

from dotenv import load_dotenv
load_dotenv()


os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_KEY")
cliente_openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def api_question(question, context: list):
    peticion = f"Use this information:\n{' '.join(context)}\n\nTo asnwer this question concisely: {question}"
    try:
        response = cliente_openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": peticion}]
        )
        
        class ObjetoSimulado:
            pass
        resultado = ObjetoSimulado()
        resultado.text = response.choices[0].message.content
        return resultado
    except Exception as e:
        tqdm.write(f"\n[!] Error de OpenAI en generación: {e}")
        return 0