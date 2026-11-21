"""
Comparación de configuraciones de RECUPERACIÓN (sin LLM, corre en segundos).

Prueba modos de búsqueda x estrategia de fragmentación x k sobre uno o más CSV de
preguntas y escribe una tabla en resultados/comparacion_recuperacion.{csv,md}.

Las métricas se calculan solo sobre las preguntas que SÍ tienen respuesta en los
documentos (documento_esperado distinto de NINGUNO).

Uso:
    python comparar.py                                  # BM25 y TF-IDF (sin descargas)
    python comparar.py --con-embeddings                 # agrega denso e híbrido (requiere el modelo HF)
    python comparar.py --csvs preguntas.csv preguntas_dificiles.csv
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import pandas as pd

from config import Config, RAIZ, consola_utf8
from evaluar import es_sin_respuesta, leer_preguntas
from rag.documentos import cargar_documentos, fragmentar_documentos
from rag.embeddings import crear_embedder
from rag.indice import Indice
from rag.metricas import (acierto_en_k, documentos_esperados, rango_primer_acierto,
                          rango_reciproco)

KS = (1, 3, 5)


def evaluar_config(indice: Indice, preguntas: pd.DataFrame, modo: str) -> dict:
    rangos = []
    for _, f in preguntas.iterrows():
        if es_sin_respuesta(f["documento_esperado"]):
            continue
        res = indice.buscar(f["pregunta"], 10, modo)
        rangos.append(rango_primer_acierto([r.fragmento.documento for r in res],
                                           documentos_esperados(f["documento_esperado"])))
    out = {f"R@{k}": sum(acierto_en_k(r, k) for r in rangos) / len(rangos) for k in KS}
    out["MRR"] = sum(rango_reciproco(r) for r in rangos) / len(rangos)
    out["n"] = len(rangos)
    return out


def main():
    consola_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--csvs", nargs="+", type=Path,
                    default=[RAIZ / "preguntas.csv", RAIZ / "preguntas_dificiles.csv", RAIZ / "preguntas_prueba.csv"])
    ap.add_argument("--con-embeddings", action="store_true")
    ap.add_argument("--salida", type=Path, default=RAIZ / "resultados")
    args = ap.parse_args()

    base = Config()
    docs = cargar_documentos(base.carpeta_docs)
    modos = ["bm25", "tfidf", "tfidf_char", "hibrido_tfidf"]
    if args.con_embeddings:
        modos += ["denso", "hibrido"]
    estrategias = [("qa", 1200, 200), ("ventana", 800, 200), ("ventana", 1200, 200), ("ventana", 400, 100)]
    variantes = [("base", False, False), ("stem", True, False), ("expansion", False, True),
                 ("stem+expansion", True, True)]
    preguntas = {p.stem: leer_preguntas(p, True) for p in args.csvs}
    embedder = crear_embedder(base) if args.con_embeddings else None

    filas = []
    for estrategia, max_c, solap in estrategias:
        frs = fragmentar_documentos(docs, estrategia, max_c, solap)
        emb = embedder.codificar_pasajes([f.texto_indexado for f in frs]) if embedder else None
        for var, stem, expandir in variantes:
            indice = Indice(frs, emb, embedder, base.rrf_k, stem, expandir)
            for modo in modos:
                for nombre, df in preguntas.items():
                    m = evaluar_config(indice, df, modo)
                    filas.append({"preguntas": nombre, "modo": modo, "variante": var,
                                  "fragmentacion": f"{estrategia}-{max_c}", "n_fragmentos": len(frs),
                                  **{k: round(v, 3) for k, v in m.items()}})

    tabla = pd.DataFrame(filas)
    args.salida.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(args.salida / "comparacion_recuperacion.csv", index=False, encoding="utf-8-sig")
    md = []
    for nombre in preguntas:
        t = tabla[tabla.preguntas == nombre].drop(columns="preguntas").sort_values(["MRR", "R@1"], ascending=False)
        md.append(f"### {nombre}\n\n" + t.to_markdown(index=False) + "\n")
    (args.salida / "comparacion_recuperacion.md").write_text("\n".join(md), encoding="utf-8")
    print("\nGuardado en", args.salida)


if __name__ == "__main__":
    main()
