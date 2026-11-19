"""Métricas de evaluación: recuperación (Recall@k, MRR) y calidad de respuesta."""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from .texto import normalizar, tokenizar


# --------------------------------------------------------------------------- #
# Recuperación
# --------------------------------------------------------------------------- #
def normalizar_nombre_doc(nombre: str) -> str:
    nombre = Path(str(nombre).strip().strip('"').replace("\\", "/")).name
    nombre = normalizar(nombre)
    return nombre if nombre.endswith(".txt") else nombre + ".txt"


def documentos_esperados(valor: str) -> set[str]:
    """Admite varios documentos separados por ';' o '|'."""
    return {normalizar_nombre_doc(v) for v in re.split(r"[;|]", str(valor)) if v.strip()}


def rango_primer_acierto(docs_recuperados: list[str], esperados: set[str]) -> int | None:
    """Posición (1-based) del primer fragmento que pertenece a un documento esperado."""
    for i, d in enumerate(docs_recuperados, start=1):
        if normalizar_nombre_doc(d) in esperados:
            return i
    return None


def acierto_en_k(rango: int | None, k: int) -> int:
    return int(rango is not None and rango <= k)


def rango_reciproco(rango: int | None) -> float:
    return 0.0 if rango is None else 1.0 / rango


# --------------------------------------------------------------------------- #
# Respuesta
# --------------------------------------------------------------------------- #
def f1_tokens(prediccion: str, referencia: str) -> float:
    """F1 a nivel de palabras (estilo SQuAD), sin tildes ni stopwords."""
    p, r = tokenizar(prediccion), tokenizar(referencia)
    if not p or not r:
        return float(p == r)
    comunes = sum((Counter(p) & Counter(r)).values())
    if comunes == 0:
        return 0.0
    precision, recall = comunes / len(p), comunes / len(r)
    return 2 * precision * recall / (precision + recall)


PROMPT_JUEZ = """Eres un evaluador estricto de un sistema de preguntas y respuestas sobre derecho laboral chileno.

Compara la RESPUESTA GENERADA con la RESPUESTA ESPERADA para la PREGUNTA.
- CORRECTA: contiene la información esencial de la respuesta esperada y no la contradice (puede usar otras palabras o añadir detalles compatibles).
- PARCIAL: contiene solo una parte de la información esencial, o es imprecisa.
- INCORRECTA: contradice la respuesta esperada, omite lo esencial o dice que no encuentra la información.

PREGUNTA: {pregunta}

RESPUESTA ESPERADA: {esperada}

RESPUESTA GENERADA: {generada}

Responde en UNA sola línea con este formato exacto:
VEREDICTO: <CORRECTA|PARCIAL|INCORRECTA> | MOTIVO: <explicación breve>"""

PUNTAJE_VEREDICTO = {"CORRECTA": 1.0, "PARCIAL": 0.5, "INCORRECTA": 0.0}


def parsear_veredicto(texto: str) -> tuple[str, str]:
    t = normalizar(texto).upper()
    m = re.search(r"VEREDICTO\s*:?\s*\**\s*(INCORRECTA|PARCIAL|CORRECTA)\b", t) \
        or re.search(r"\b(INCORRECTA|PARCIAL|CORRECTA)\b", t)
    veredicto = m.group(1) if m else "SIN_VEREDICTO"
    motivo = re.search(r"MOTIVO\s*:\s*(.*)", texto, re.IGNORECASE | re.DOTALL)
    return veredicto, (motivo.group(1) if motivo else texto).strip().replace("\n", " ")


def juzgar_con_llm(llm, pregunta: str, esperada: str, generada: str) -> tuple[str, str, float | None]:
    prompt = PROMPT_JUEZ.format(pregunta=pregunta, esperada=esperada, generada=generada)
    salida = llm.generar(prompt, temperatura=0.0, max_tokens=150)
    veredicto, motivo = parsear_veredicto(salida)
    return veredicto, motivo, PUNTAJE_VEREDICTO.get(veredicto)
