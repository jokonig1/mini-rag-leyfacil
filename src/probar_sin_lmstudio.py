"""
Prueba de humo del pipeline completo SIN LM Studio: reemplaza el LLM por uno falso.
Sirve para verificar que evaluar.py (generación, juez y preguntas sin respuesta) funciona.
NO produce métricas reales de calidad: las respuestas son simuladas.

Uso:  python probar_sin_lmstudio.py
"""
import re
import sys

import evaluar


class LLMFalso:
    modelo = "llm-falso"

    def __init__(self, cfg):
        pass

    def generar(self, prompt, temperatura=None, max_tokens=None):
        if "evaluador estricto" in prompt:                      # llamada del juez
            return "VEREDICTO: CORRECTA | MOTIVO: simulado"
        pregunta = prompt.rsplit("PREGUNTA:", 1)[1].split("RESPUESTA:")[0]
        if any(t in pregunta for t in ("alimentos", "cárcel", "bono de invierno", "marca comercial",
                                        "cédula", "divorcio", "cuota mortuoria", "licencia de conducir")):
            return "No encuentro esa información en los documentos disponibles."
        doc = re.search(r"\[Fragmento 1\] Documento: (\S+)", prompt).group(1)
        return f"Respuesta simulada.\nFuente: {doc}"


evaluar.ClienteLLM = LLMFalso
sys.argv = ["evaluar.py", "--embeddings", "ninguno", "--modo", "bm25", "--salida", "resultados_mock"]
evaluar.main()
