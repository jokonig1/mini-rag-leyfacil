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


_STEMMER = None


def raiz(token: str) -> str:
    """Stemming Snowball en español (trabajadores -> trabaj). Si nltk falta, no hace nada."""
    global _STEMMER
    if _STEMMER is None:
        try:
            from nltk.stem.snowball import SpanishStemmer
            _STEMMER = SpanishStemmer()
        except ImportError:
            _STEMMER = False
    return _STEMMER.stem(token) if _STEMMER and not token.isdigit() else token


def tokenizar(texto: str, quitar_stopwords: bool = True, stem: bool = False) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", normalizar(texto))
    if quitar_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS_ES and (len(t) > 1 or t.isdigit())]
    if stem:
        tokens = [raiz(t) for t in tokens]
    return tokens


# Léxico de chilenismos y lenguaje coloquial -> términos del lenguaje legal de los documentos.
SINONIMOS = {
    "guagua": "hijo lactante bebe", "bebe": "hijo lactante", "papa": "padre", "mama": "madre",
    "jefe": "empleador", "patron": "empleador", "plata": "dinero pago", "sueldo": "remuneracion",
    "salario": "remuneracion", "echaron": "despido termino contrato", "echar": "despido termino",
    "despidieron": "despido termino contrato", "despedir": "despido", "desempleo": "cesantia",
    "cesante": "cesantia", "casar": "matrimonio", "fallecio": "duelo muerte", "murio": "duelo muerte",
    "libres": "permiso", "horario": "jornada", "extra": "extraordinarias", "joven": "adolescente",
    "chico": "adolescente", "juzgado": "tribunales", "juzgados": "tribunales",
    "pega": "trabajo empleo", "chamba": "trabajo empleo",
}


def expandir_consulta(consulta: str) -> str:
    """Query expansion: agrega términos legales equivalentes a palabras coloquiales."""
    norm = normalizar(consulta)
    extra = [v for k, v in SINONIMOS.items() if normalizar(k) in norm.split() or (" " in k and normalizar(k) in norm)]
    return consulta + " " + " ".join(extra) if extra else consulta


def dividir_oraciones(texto: str) -> list[str]:
    partes = re.split(r"(?<=[.;:!?])\s+", texto.strip())
    return [p.strip() for p in partes if p.strip()]
