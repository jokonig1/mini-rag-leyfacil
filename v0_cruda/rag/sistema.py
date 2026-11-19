"""Orquestación del Mini-RAG: recuperar fragmentos -> armar prompt -> generar respuesta."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from .indice import Indice, Resultado, cargar_o_construir
from .llm import ClienteLLM
from .texto import normalizar

SIN_INFORMACION = "No encuentro esa información en los documentos disponibles."

PLANTILLA_PROMPT = """Eres un asistente que responde preguntas sobre derecho laboral chileno.

INSTRUCCIONES:
1. Responde ÚNICAMENTE con la información de los FRAGMENTOS de abajo. No uses conocimiento propio.
2. Si los fragmentos no contienen la respuesta, escribe exactamente: "{sin_info}"
3. Responde en español, de forma clara y breve (máximo 5 oraciones).
4. Termina con una línea con este formato exacto:
Fuente: <nombre del archivo .txt del que sacaste la respuesta>

FRAGMENTOS:
{contexto}

PREGUNTA: {pregunta}

RESPUESTA:"""


@dataclass
class RespuestaRAG:
    pregunta: str
    respuesta: str                 # texto completo devuelto por el LLM
    respuesta_limpia: str          # sin la línea "Fuente: ..."
    fuentes: list[str]             # documentos citados (validados)
    fuentes_inferidas: bool        # True si el LLM no citó y se usó el 1.er fragmento
    sin_informacion: bool
    resultados: list[Resultado] = field(default_factory=list)
    tiempo_recuperacion_s: float = 0.0
    tiempo_generacion_s: float = 0.0


def formatear_contexto(resultados: list[Resultado]) -> str:
    bloques = []
    for r in resultados:
        f = r.fragmento
        bloques.append(f"[Fragmento {r.rango}] Documento: {f.documento} | Título: {f.titulo}\n{f.texto}")
    return "\n\n---\n\n".join(bloques)


def separar_fuente(texto: str) -> tuple[str, str]:
    """Devuelve (respuesta_sin_fuente, línea_fuente)."""
    m = re.search(r"^\s*\**\s*fuentes?\s*(?::\s*\**|\**\s*:)\s*(.*)$", texto, re.IGNORECASE | re.MULTILINE)
    if not m:
        return texto.strip(), ""
    return (texto[:m.start()] + texto[m.end():]).strip(), m.group(1).strip()


def extraer_fuentes(texto: str, resultados: list[Resultado]) -> list[str]:
    """Identifica qué documentos recuperados cita el LLM (por nombre o por n.º de fragmento)."""
    norm = normalizar(texto)
    encontrados: list[tuple[int, str]] = []
    for r in resultados:
        doc = r.fragmento.documento
        stem = normalizar(Path(doc).stem)
        for candidato in {normalizar(doc), stem, stem.replace("_", " "), stem.replace("-", " ")}:
            pos = norm.find(candidato)
            if pos != -1:
                encontrados.append((pos, doc))
                break
    for m in re.finditer(r"fragmento\s*(\d+)", norm):
        n = int(m.group(1))
        if 1 <= n <= len(resultados):
            encontrados.append((m.start(), resultados[n - 1].fragmento.documento))
    vistos, salida = set(), []
    for _, doc in sorted(encontrados):
        if doc not in vistos:
            vistos.add(doc)
            salida.append(doc)
    return salida


class MiniRAG:
    def __init__(self, cfg, indice: Indice | None = None, llm: ClienteLLM | None = None,
                 reconstruir: bool = False):
        self.cfg = cfg
        self.indice = indice or cargar_o_construir(cfg, reconstruir=reconstruir)
        self.llm = llm or ClienteLLM(cfg)

    def recuperar(self, pregunta: str, k: int | None = None) -> list[Resultado]:
        return self.indice.buscar(pregunta, k or self.cfg.top_k, self.cfg.modo_busqueda)

    def responder(self, pregunta: str, k: int | None = None,
                  resultados: list[Resultado] | None = None) -> RespuestaRAG:
        k = k or self.cfg.top_k
        t0 = time.time()
        if resultados is None:
            resultados = self.recuperar(pregunta, k)
        resultados = resultados[:k]
        t_rec = time.time() - t0

        prompt = PLANTILLA_PROMPT.format(sin_info=SIN_INFORMACION,
                                         contexto=formatear_contexto(resultados),
                                         pregunta=pregunta.strip())
        t1 = time.time()
        texto = self.llm.generar(prompt)
        t_gen = time.time() - t1

        limpia, linea_fuente = separar_fuente(texto)
        sin_info = normalizar(SIN_INFORMACION[:30]) in normalizar(limpia)
        fuentes = extraer_fuentes(linea_fuente or texto, resultados)
        inferidas = False
        if not fuentes and not sin_info and resultados:
            fuentes = [resultados[0].fragmento.documento]
            inferidas = True
        return RespuestaRAG(pregunta, texto, limpia, fuentes, inferidas, sin_info,
                            resultados, t_rec, t_gen)
