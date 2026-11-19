"""
Descarga guías de Ley Fácil (Biblioteca del Congreso Nacional) sobre derecho laboral
y las guarda como texto plano en la carpeta docs/.

Uso:
    pip install requests beautifulsoup4
    python descargar_leyfacil.py

Salida:
    docs/<slug>.txt      -> una guía por archivo (título, URL y preguntas/respuestas)
    docs/_indice.csv     -> slug, título, URL, nº de caracteres, estado
"""

import csv
import json
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

API = "https://www.bcn.cl/api-leyfacil/servicio/ObtenerGuiaPublicadaHTML?uri={slug}"
PORTAL = "https://www.bcn.cl/portal/leyfacil/recurso/{slug}"
CARPETA = Path("docs")
PAUSA_SEG = 1.5  # para no saturar el servidor de la BCN

# 37 guías confirmadas, agrupadas por tema
GUIAS = [
    # --- Contrato, término y despido ---
    "termino-de-contrato-de-trabajo",
    "indemnizacion-por-despido",
    "trabajadores-por-obra-o-faena",
    "empresas-de-servicios-transitorios",
    "subcontratacion",
    "tribunales-laborales",
    # --- Jornada y descansos ---
    "reduccion-de-la-jornada-laboral-a-40-horas-semanales",
    "horas-extraordinarias",
    "jornada-especial-de-trabajo",
    "jornada-parcial-alternativa-para-estudiantes-trabajadores",
    "trabajar-en-festividades",  # Feriados irrenunciables
    "trabajo-a-distancia-y-teletrabajo",
    # --- Remuneraciones y subsidios ---
    "sueldo-minimo-sueldo-base-derecho-a-semana-corrida",
    "ingreso-minimo-garantizado",
    "asignacion-familiar",
    "subsidio-al-empleo-de-la-mujer",
    "subsidio-unificado-de-empleo",
    "contratacion-de-aprendices",
    # --- Maternidad, paternidad y permisos ---
    "proteccion-a-la-maternidad",
    "postnatal",
    "derecho-de-amamantamiento",
    "permiso-laboral-por-nacimiento",  # nacimiento, matrimonio, unión civil o duelo
    "licencias-medicas",
    # --- Seguridad social ---
    "seguro-de-cesantia",
    "accidentes-del-trabajo",
    "incorporacion-de-trabajadores-y-trabajadoras-independientes-al-sistema-de-prevision-y-proteccion-social",
    "reforma-previsional--trabajadores-jovenes",
    "reforma-de-pensiones-de-2025",
    "pensiones-de-invalidez",
    # --- Grupos específicos de trabajadores ---
    "trabajadoras-y-trabajadores-de-casa-particular-nueva-ley",
    "trabajadores-de-empresas-de-plataformas-digitales-de-servicios",
    "temporeros-y-temporeras",
    "trabajo-adolescente-protegido",
    "inclusion-de-personas-en-situacion-de-discapacidad-al-mundo-laboral",
    # --- Derechos colectivos y acoso ---
    "reforma-laboral-negociacion-colectiva",
    "reforma-laboral-practicas-antisindicales-y-practicas-desleales",
    "acoso-sexual-acoso-laboral-y-violencia-en-el-trabajo",  # Ley Karin
]

HEADERS = {"User-Agent": "Mozilla/5.0 (proyecto academico NLP - UAI)"}


def bajar(url: str, intentos: int = 3) -> str | None:
    for i in range(intentos):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 200 and r.text.strip():
                r.encoding = r.encoding or "utf-8"
                return r.text
            print(f"    HTTP {r.status_code} en {url}")
        except requests.RequestException as e:
            print(f"    error ({e.__class__.__name__}), reintento {i + 1}/{intentos}")
        time.sleep(2 * (i + 1))
    return None


def desenvolver(texto: str) -> str:
    """La API podría devolver HTML directo o un JSON que contiene el HTML."""
    t = texto.strip()
    if t.startswith("{") or t.startswith('"'):
        try:
            data = json.loads(t)
            if isinstance(data, str):
                return data
            if isinstance(data, dict):
                for v in data.values():
                    if isinstance(v, str) and "<" in v:
                        return v
        except json.JSONDecodeError:
            pass
    return t


def html_a_texto(html: str) -> tuple[str, str]:
    """Devuelve (título, cuerpo). Conserva saltos de línea entre párrafos,
    para poder hacer chunking por pregunta más adelante."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "noscript"]):
        tag.decompose()

    titulo_tag = soup.find(["h1", "h2", "title"])
    titulo = titulo_tag.get_text(" ", strip=True) if titulo_tag else ""
    titulo = re.sub(r"\s*-\s*Ley Fácil.*$", "", titulo)

    bloques = []
    for el in soup.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
        if el.find_parent(["li", "p"]):  # evita duplicar <p> dentro de <li>
            continue
        txt = el.get_text(" ", strip=True)
        if not txt:
            continue
        if el.name == "li":
            txt = "- " + txt
        bloques.append(txt)

    # Quitar duplicados consecutivos (pasa cuando hay <p> dentro de <li>)
    limpio = [b for i, b in enumerate(bloques) if i == 0 or b != bloques[i - 1]]
    cuerpo = "\n\n".join(limpio)
    if not cuerpo:  # respaldo si el HTML no tiene esas etiquetas
        cuerpo = soup.get_text("\n", strip=True)
    return titulo, cuerpo


def main() -> None:
    CARPETA.mkdir(exist_ok=True)
    filas = []
    for n, slug in enumerate(GUIAS, 1):
        destino = CARPETA / f"{slug}.txt"
        url_portal = PORTAL.format(slug=slug)
        print(f"[{n:02d}/{len(GUIAS)}] {slug}")

        if destino.exists():
            print("    ya existe, se omite")
            filas.append([slug, "", url_portal, destino.stat().st_size, "ya_existia"])
            continue

        html = bajar(API.format(slug=slug))
        fuente = "api"
        if html is None:  # respaldo: página pública
            html = bajar(url_portal)
            fuente = "portal"
        if html is None:
            filas.append([slug, "", url_portal, 0, "FALLO"])
            continue

        titulo, cuerpo = html_a_texto(desenvolver(html))
        if len(cuerpo) < 300:
            print(f"    OJO: texto muy corto ({len(cuerpo)} caracteres), revisar a mano")
        destino.write_text(
            f"TÍTULO: {titulo or slug}\nFUENTE: {url_portal}\n\n{cuerpo}\n",
            encoding="utf-8",
        )
        filas.append([slug, titulo, url_portal, len(cuerpo), f"ok_{fuente}"])
        time.sleep(PAUSA_SEG)

    with open(CARPETA / "_indice.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["slug", "titulo", "url", "caracteres", "estado"])
        w.writerows(filas)

    ok = sum(1 for f in filas if f[4] != "FALLO")
    print(f"\nListo: {ok}/{len(GUIAS)} guías en {CARPETA}/ (detalle en _indice.csv)")


if __name__ == "__main__":
    main()
