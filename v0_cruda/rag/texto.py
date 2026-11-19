"""Utilidades de normalización y tokenización para español."""
from __future__ import annotations

import re
import unicodedata

STOPWORDS_ES = set("""
a al algo algun alguna algunas alguno algunos ante antes aquel aquella aquellas aquello aquellos
aqui asi aun bajo bien cada como con contra cual cuales cualquier cuando de del desde donde dos
el ella ellas ello ellos en entre era eran es esa esas ese eso esos esta estaba estado estan estar
estas este esto estos fue fueron ha habia han hasta hay la las le les lo los mas me mi mis mismo
mucho muy nada ni no nos nosotros o otra otras otro otros para pero poco por porque que quien
quienes se segun ser si sido siempre sin sobre su sus tal tambien tan tanto te tiene tienen todo
todos tu tus un una unas uno unos usted ustedes y ya yo puede pueden debe deben hace hacer
""".split())


def quitar_tildes(texto: str) -> str:
    descompuesto = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in descompuesto if not unicodedata.combining(c))


def normalizar(texto: str) -> str:
    return quitar_tildes(texto.lower())


def tokenizar(texto: str, quitar_stopwords: bool = True) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", normalizar(texto))
    if quitar_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS_ES and (len(t) > 1 or t.isdigit())]
    return tokens


def dividir_oraciones(texto: str) -> list[str]:
    partes = re.split(r"(?<=[.;:!?])\s+", texto.strip())
    return [p.strip() for p in partes if p.strip()]
