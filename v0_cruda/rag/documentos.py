"""
Carga de documentos .txt y fragmentación (chunking).

Formato esperado de cada archivo:

    TÍTULO: ...
    FUENTE: ...

    ¿Pregunta 1?
    Respuesta 1 ...

    ¿Pregunta 2?
    Respuesta 2 ...
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

from .texto import dividir_oraciones, quitar_tildes

CODIFICACIONES = ("utf-8-sig", "cp1252", "latin-1")
_RE_CAMPO = re.compile(r"^\s*(T[ÍI]TULO|FUENTE)\s*:\s*(.*)$", re.IGNORECASE)


@dataclass
class Documento:
    nombre: str   # nombre del archivo, p. ej. "vacaciones.txt"
    titulo: str
    fuente: str
    cuerpo: str


@dataclass
class Fragmento:
    id: str
    documento: str
    titulo: str
    fuente: str
    texto: str
    posicion: int

    @property
    def texto_indexado(self) -> str:
        # El título del documento se antepone para dar contexto a la búsqueda.
        return f"{self.titulo}\n{self.texto}"

    def a_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Carga
# --------------------------------------------------------------------------- #
def leer_texto(ruta: Path) -> str:
    datos = ruta.read_bytes()
    for cod in CODIFICACIONES:
        try:
            return datos.decode(cod)
        except UnicodeDecodeError:
            continue
    return datos.decode("utf-8", errors="replace")


def cargar_documento(ruta: Path) -> Documento:
    texto = leer_texto(ruta).replace("\r\n", "\n").replace("\r", "\n")
    lineas = texto.split("\n")
    titulo, fuente = "", ""
    cuerpo_lineas = []
    for i, linea in enumerate(lineas):
        m = _RE_CAMPO.match(linea) if i < 10 else None
        if m:
            campo = quitar_tildes(m.group(1)).upper()
            if campo == "TITULO" and not titulo:
                titulo = m.group(2).strip()
                continue
            if campo == "FUENTE" and not fuente:
                fuente = m.group(2).strip()
                continue
        cuerpo_lineas.append(linea)
    return Documento(
        nombre=ruta.name,
        titulo=titulo or ruta.stem.replace("_", " "),
        fuente=fuente,
        cuerpo="\n".join(cuerpo_lineas).strip(),
    )


def cargar_documentos(carpeta: Path) -> list[Documento]:
    carpeta = Path(carpeta)
    if not carpeta.is_dir():
        raise FileNotFoundError(f"No existe la carpeta de documentos: {carpeta}")
    rutas = sorted(carpeta.glob("*.txt"), key=lambda p: p.name.lower())
    if not rutas:
        raise FileNotFoundError(f"No hay archivos .txt en {carpeta}")
    return [cargar_documento(r) for r in rutas]


# --------------------------------------------------------------------------- #
# Fragmentación
# --------------------------------------------------------------------------- #
def dividir_bloques(cuerpo: str) -> list[str]:
    return [b.strip() for b in re.split(r"\n\s*\n", cuerpo) if b.strip()]


def es_pregunta(bloque: str) -> bool:
    primera = bloque.split("\n", 1)[0].strip()
    return len(primera) <= 300 and (primera.endswith("?") or primera.startswith("¿"))


def agrupar_qa(bloques: list[str]) -> list[list[str]]:
    """Agrupa los bloques en unidades: una pregunta + los párrafos de su respuesta."""
    unidades, actual = [], []
    for b in bloques:
        if es_pregunta(b) and actual:
            unidades.append(actual)
            actual = []
        actual.append(b)
    if actual:
        unidades.append(actual)
    return unidades


def _empaquetar(piezas: list[str], presupuesto: int, sep: str) -> list[str]:
    grupos, actual = [], ""
    for p in piezas:
        if not actual:
            actual = p
        elif len(actual) + len(sep) + len(p) <= presupuesto:
            actual += sep + p
        else:
            grupos.append(actual)
            actual = p
    if actual:
        grupos.append(actual)
    return grupos


def _cortar_largo(texto: str, presupuesto: int) -> list[str]:
    """Divide un texto largo por oraciones y, en último caso, por caracteres."""
    oraciones = []
    for o in dividir_oraciones(texto):
        while len(o) > presupuesto:
            corte = o.rfind(" ", 0, presupuesto)
            corte = corte if corte > presupuesto // 2 else presupuesto
            oraciones.append(o[:corte].strip())
            o = o[corte:].strip()
        if o:
            oraciones.append(o)
    return _empaquetar(oraciones, presupuesto, " ")


def _partir_unidad(unidad: list[str], max_c: int) -> list[str]:
    texto = "\n\n".join(unidad)
    if len(texto) <= max_c:
        return [texto]
    # Unidad demasiado larga: se divide y se repite la pregunta en cada parte.
    encabezado, piezas = "", list(unidad)
    if es_pregunta(unidad[0]):
        partes = unidad[0].split("\n", 1)
        encabezado = partes[0].strip()
        piezas = ([partes[1].strip()] if len(partes) > 1 and partes[1].strip() else []) + unidad[1:]
    presupuesto = max(200, max_c - (len(encabezado) + 2 if encabezado else 0))
    expandidas = []
    for p in piezas:
        expandidas.extend(_cortar_largo(p, presupuesto) if len(p) > presupuesto else [p])
    grupos = _empaquetar(expandidas, presupuesto, "\n\n")
    return [f"{encabezado}\n\n{g}" if encabezado else g for g in grupos]


def fragmentar_qa(doc: Documento, max_c: int) -> list[str]:
    textos = []
    for unidad in agrupar_qa(dividir_bloques(doc.cuerpo)):
        textos.extend(_partir_unidad(unidad, max_c))
    return textos


def fragmentar_ventana(doc: Documento, max_c: int, solapamiento: int) -> list[str]:
    texto = re.sub(r"\s+", " ", doc.cuerpo).strip()
    salida, inicio = [], 0
    while inicio < len(texto):
        fin = min(inicio + max_c, len(texto))
        if fin < len(texto):
            corte = texto.rfind(" ", inicio + max_c // 2, fin)
            if corte != -1:
                fin = corte
        salida.append(texto[inicio:fin].strip())
        if fin >= len(texto):
            break
        nuevo = max(fin - solapamiento, inicio + 1)
        espacio = texto.find(" ", nuevo, fin)
        inicio = espacio + 1 if espacio != -1 else nuevo
    return [s for s in salida if s]


def fragmentar_documentos(docs: list[Documento], estrategia: str = "qa",
                          max_c: int = 1200, solapamiento: int = 200) -> list[Fragmento]:
    fragmentos: list[Fragmento] = []
    for doc in docs:
        if estrategia == "ventana":
            textos = fragmentar_ventana(doc, max_c, solapamiento)
        else:
            textos = fragmentar_qa(doc, max_c)
        stem = Path(doc.nombre).stem
        for i, t in enumerate(textos):
            fragmentos.append(Fragmento(
                id=f"{stem}#{i:03d}", documento=doc.nombre, titulo=doc.titulo,
                fuente=doc.fuente, texto=t, posicion=i,
            ))
    return fragmentos
