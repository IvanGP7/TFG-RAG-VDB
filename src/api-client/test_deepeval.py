import os
from deepeval import evaluate
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from config import OPENAI_KEY

# ⚠️ DeepEval utiliza OpenAI por debajo para actuar como "Juez Evaluador".
# Pon aquí tu clave de OpenAI.
os.environ["OPENAI_API_KEY"] = OPENAI_KEY

def prueba_sintetica_deepeval():
    print("1. ⚖️ Preparando el examen de DeepEval...")
    
    # Definimos las métricas y la nota mínima para aprobar (threshold del 0 al 1)
    # Exigimos un 70% (0.7) de nota mínima para considerar que el LLM ha aprobado.
    metrica_fidelidad = FaithfulnessMetric(threshold=0.7)
    metrica_relevancia = AnswerRelevancyMetric(threshold=0.7)

    print("2. 📝 Creando un caso de prueba (Muestra inventada)...")
    caso_de_prueba = LLMTestCase(
        input="¿A quién se le apareció la Virgen María en Lourdes?",
        # Esta es la respuesta simulada que daría tu Gemini
        actual_output="Según los textos, la Virgen María se le apareció a Santa Bernadette Soubirous en la gruta.",
        # Este es el contexto simulado que traería tu ChromaDB
        retrieval_context=["En 1858, en la ciudad de Lourdes, la Virgen María supuestamente se le apareció a una joven llamada Santa Bernadette Soubirous en la gruta de Massabielle."]
    )

    print("3. 🚀 Ejecutando evaluación (El Juez de OpenAI está leyendo)...")
    print("-" * 50)
    
    # Lanzamos la evaluación. Pasamos una lista con nuestro caso, y una lista con las métricas.
    evaluate([caso_de_prueba], [metrica_fidelidad, metrica_relevancia])

if __name__ == "__main__":
    prueba_sintetica_deepeval()