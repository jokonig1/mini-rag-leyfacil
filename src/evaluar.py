"""
Paso 4: evaluación del Mini-RAG con preguntas.csv
(columnas: pregunta, respuesta_esperada, documento_esperado).

Mide:
  * Recuperación: Recall@k (k = 1, 3, 5 por defecto) y MRR, a nivel de documento:
    ¿aparece algún fragmento del documento esperado entre los k recuperados?
  * Respuesta: juez LLM local (CORRECTA / PARCIAL / INCORRECTA), F1 de palabras,
    similitud semántica y si el documento citado es el correcto.

Uso:
    python evaluar.py                               # usa preguntas.csv
    python evaluar.py --csv otras_preguntas.csv --ks 1,3,5,10
    python evaluar.py --solo-recuperacion          # rápido, sin LLM
    python evaluar.py --sin-juez                   # sin juez LLM (usa similitud/F1)
    python evaluar.py --modo bm25                  # comparar configuraciones
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from config import RAIZ, agregar_argumentos_comunes, config_desde_args, consola_utf8
from rag.indice import cargar_o_construir
from rag.llm import ClienteLLM
from rag.metricas import (acierto_en_k, documentos_esperados, f1_tokens, juzgar_con_llm,
                          normalizar_nombre_doc, rango_primer_acierto, rango_reciproco)
from rag.sistema import MiniRAG
from rag.texto import normalizar

COLUMNAS = ["pregunta", "respuesta_esperada", "documento_esperado"]
# Preguntas cuya respuesta NO está en los documentos: documento_esperado = NINGUNO.
# Se evalúan aparte: el sistema acierta si se abstiene en vez de inventar.
SIN_RESPUESTA = {"", "NINGUNO", "NINGUNA", "NA", "N/A", "-"}


def es_sin_respuesta(valor: str) -> bool:
    return str(valor).strip().upper() in SIN_RESPUESTA


def leer_preguntas(ruta: Path, solo_recuperacion: bool) -> pd.DataFrame:
    """Lee el CSV detectando separador (, o ;) y codificación (UTF-8 o la de Excel en Windows)."""
    ultimo_error = None
    for cod in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(ruta, sep=None, engine="python", encoding=cod, dtype=str,
                             keep_default_na=False)
            break
        except UnicodeDecodeError as e:
            ultimo_error = e
    else:
        raise RuntimeError(f"No se pudo leer {ruta}: {ultimo_error}")
    df.columns = [normalizar(c).strip().replace(" ", "_") for c in df.columns]
    requeridas = ["pregunta", "documento_esperado"] + ([] if solo_recuperacion else ["respuesta_esperada"])
    faltan = [c for c in requeridas if c not in df.columns]
    if faltan:
        raise RuntimeError(f"Faltan columnas en {ruta.name}: {faltan}. Encontradas: {list(df.columns)}")
    if "respuesta_esperada" not in df.columns:
        df["respuesta_esperada"] = ""
    df = df[df["pregunta"].str.strip() != ""].reset_index(drop=True)
    return df


def main():
    consola_utf8()
    parser = argparse.ArgumentParser(description="Evalúa el Mini-RAG")
    parser.add_argument("--csv", type=Path, default=RAIZ / "preguntas.csv")
    parser.add_argument("--ks", default="1,3,5", help="Valores de k para Recall@k, p. ej. 1,3,5,10")
    parser.add_argument("--solo-recuperacion", action="store_true", help="No genera respuestas")
    parser.add_argument("--sin-juez", action="store_true", help="No usa el LLM como juez")
    parser.add_argument("--limite", type=int, help="Evalúa solo las primeras N preguntas")
    parser.add_argument("--salida", type=Path, help="Carpeta de resultados (por defecto resultados/)")
    parser.add_argument("--reconstruir", action="store_true", help="Fuerza reconstruir el índice")
    agregar_argumentos_comunes(parser)
    args = parser.parse_args()
    cfg = config_desde_args(args)
    ks = sorted({int(x) for x in args.ks.split(",") if x.strip()})
    usar_juez = cfg.usar_juez_llm and not args.sin_juez

    df = leer_preguntas(args.csv, args.solo_recuperacion)
    if args.limite:
        df = df.head(args.limite)

    indice = cargar_o_construir(cfg, reconstruir=args.reconstruir)
    disponibles = {normalizar_nombre_doc(d) for d in indice.documentos}
    for i, valor in enumerate(df["documento_esperado"], start=2):
        if es_sin_respuesta(valor):
            continue
        for d in documentos_esperados(valor):
            if d not in disponibles:
                print(f"[aviso] Fila {i}: '{d}' no está en docs/ (nunca podrá recuperarse).")

    rag = None
    if not args.solo_recuperacion:
        rag = MiniRAG(cfg, indice=indice, llm=ClienteLLM(cfg))
        _ = rag.llm.modelo  # verifica la conexión con LM Studio antes de empezar

    salida_dir = args.salida or cfg.carpeta_resultados
    salida_dir.mkdir(parents=True, exist_ok=True)
    marca = time.strftime("%Y%m%d_%H%M%S")
    etiqueta = f"{cfg.modo_busqueda}_{cfg.estrategia_fragmentacion}_k{cfg.top_k}_p{cfg.prompt}"
    ruta_csv = salida_dir / f"resultados_{etiqueta}_{marca}.csv"
    ruta_resumen = salida_dir / f"resumen_{etiqueta}_{marca}.json"

    k_recuperar = max(max(ks), cfg.top_k)
    filas = []
    for idx, fila in tqdm(df.iterrows(), total=len(df), desc="Evaluando", unit="preg"):
        pregunta = fila["pregunta"].strip()
        esperada = fila["respuesta_esperada"].strip()
        sin_resp = es_sin_respuesta(fila["documento_esperado"])
        esperados = set() if sin_resp else documentos_esperados(fila["documento_esperado"])

        t0 = time.time()
        resultados = indice.buscar(pregunta, k_recuperar, cfg.modo_busqueda)
        t_rec = time.time() - t0
        docs_rec = [r.fragmento.documento for r in resultados]
        rango = rango_primer_acierto(docs_rec, esperados)

        reg = {
            "id": idx + 1,
            "pregunta": pregunta,
            "documento_esperado": fila["documento_esperado"],
            "documentos_recuperados": " | ".join(docs_rec),
            "fragmentos_recuperados": " | ".join(r.fragmento.id for r in resultados),
            "sin_respuesta": int(sin_resp),
            "rango_documento_correcto": rango if rango is not None else "",
            "puntaje_top1": round(resultados[0].puntaje, 4) if resultados else "",
        }
        for k in ks:
            reg[f"acierto@{k}"] = None if sin_resp else acierto_en_k(rango, k)
        reg["rango_reciproco"] = None if sin_resp else round(rango_reciproco(rango), 4)
        reg["tiempo_recuperacion_s"] = round(t_rec, 3)

        if rag is not None:
            reg["respuesta_esperada"] = esperada
            try:
                resp = rag.responder(pregunta, resultados=resultados[:cfg.top_k])
                generada = resp.respuesta_limpia
                reg["respuesta_generada"] = generada
                reg["documentos_citados"] = " | ".join(resp.fuentes)
                reg["cita_inferida"] = int(resp.fuentes_inferidas)
                reg["cita_correcta"] = None if sin_resp else int(any(normalizar_nombre_doc(d) in esperados for d in resp.fuentes))
                reg["sin_informacion"] = int(resp.sin_informacion)
                reg["f1_tokens"] = round(f1_tokens(generada, esperada), 4)
                if indice.embedder is not None and esperada:
                    reg["similitud_semantica"] = round(indice.embedder.similitud(generada, esperada), 4)
                if sin_resp:
                    # Correcto = el sistema se abstuvo. Si responde algo, es una alucinación.
                    reg["respuesta_correcta"] = int(resp.sin_informacion)
                    reg["veredicto_juez"] = "ABSTENCION_OK" if resp.sin_informacion else "ALUCINACION"
                elif usar_juez and esperada:
                    veredicto, motivo, puntaje = juzgar_con_llm(rag.llm, pregunta, esperada, generada)
                    reg["veredicto_juez"] = veredicto
                    reg["puntaje_juez"] = puntaje if puntaje is not None else ""
                    reg["motivo_juez"] = motivo
                    reg["respuesta_correcta"] = int(veredicto == "CORRECTA")
                else:
                    sim = reg.get("similitud_semantica")
                    ok = (sim is not None and sim >= cfg.umbral_similitud) or reg["f1_tokens"] >= cfg.umbral_f1
                    reg["respuesta_correcta"] = int(ok and not resp.sin_informacion)
                reg["tiempo_generacion_s"] = round(resp.tiempo_generacion_s, 2)
            except RuntimeError as e:
                reg["error"] = str(e)
                tqdm.write(f"[error] Pregunta {idx + 1}: {e}")
        filas.append(reg)
        # Se guarda tras cada pregunta para no perder resultados si algo falla.
        pd.DataFrame(filas).to_csv(ruta_csv, index=False, encoding="utf-8-sig")

    res = pd.DataFrame(filas)
    resumen = {
        "fecha": marca,
        "archivo_preguntas": str(args.csv),
        "n_preguntas": len(res),
        "configuracion": {
            "modo_busqueda": cfg.modo_busqueda, "top_k_llm": cfg.top_k,
            "estrategia_fragmentacion": cfg.estrategia_fragmentacion,
            "max_caracteres": cfg.max_caracteres, "n_fragmentos_indice": len(indice.fragmentos),
            "embeddings": cfg.backend_embeddings if cfg.backend_embeddings == "ninguno"
            else (cfg.modelo_embeddings if cfg.backend_embeddings != "lmstudio" else cfg.modelo_embeddings_lmstudio),
            "llm": rag.llm.modelo if rag else None, "juez_llm": usar_juez if rag else None,
        },
        "n_con_respuesta": int((res["sin_respuesta"] == 0).sum()),
        "n_sin_respuesta": int((res["sin_respuesta"] == 1).sum()),
        # Recall y MRR solo sobre las preguntas que SÍ tienen respuesta en los documentos.
        "recuperacion": {f"recall@{k}": round(float(pd.to_numeric(res[f"acierto@{k}"], errors="coerce").mean()), 4) for k in ks},
    }
    resumen["configuracion"]["prompt"] = cfg.prompt
    resumen["recuperacion"]["mrr"] = round(float(pd.to_numeric(res["rango_reciproco"], errors="coerce").mean()), 4)
    def media(col, dec=4):
        return round(float(pd.to_numeric(res[col], errors="coerce").mean()), dec) if col in res else None

    if rag is not None and "respuesta_correcta" in res:
        con = res[res["sin_respuesta"] == 0]
        sin = res[res["sin_respuesta"] == 1]
        gen = {
            "exactitud (solo con respuesta)": round(float(pd.to_numeric(con["respuesta_correcta"], errors="coerce").mean()), 4) if len(con) else None,
            "abstencion_correcta (solo sin respuesta)": round(float(pd.to_numeric(sin["respuesta_correcta"], errors="coerce").mean()), 4) if len(sin) else None,
            "exactitud_global": media("respuesta_correcta"),
            "f1_tokens_promedio": media("f1_tokens"),
            "cita_correcta": media("cita_correcta"),
            "tasa_sin_informacion": media("sin_informacion"),
            "tiempo_generacion_promedio_s": media("tiempo_generacion_s", 2),
        }
        if "puntaje_juez" in res:
            puntajes = pd.to_numeric(res["puntaje_juez"], errors="coerce")
            gen["puntaje_juez_promedio (parcial=0.5)"] = round(puntajes.mean(), 4)
            gen["veredictos"] = res["veredicto_juez"].value_counts().to_dict()
        if "similitud_semantica" in res:
            gen["similitud_semantica_promedio"] = media("similitud_semantica")
        if "error" in res:
            gen["errores"] = int(res["error"].notna().sum())
        resumen["generacion"] = gen

    ruta_resumen.write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    # Histórico acumulado: una fila por ejecución, útil para comparar configuraciones.
    plano = {"fecha": marca, **resumen["configuracion"], **resumen["recuperacion"],
             **{k: v for k, v in resumen.get("generacion", {}).items() if not isinstance(v, dict)}}
    hist = salida_dir / "historial_resumenes.csv"
    historial = pd.DataFrame([plano])
    if hist.exists():
        historial = pd.concat([pd.read_csv(hist, encoding="utf-8-sig"), historial], ignore_index=True)
    historial.to_csv(hist, index=False, encoding="utf-8-sig")

    print("\n=========== RESUMEN ===========")
    print(f"Preguntas evaluadas: {len(res)}")
    for k, v in resumen["recuperacion"].items():
        print(f"  {k.upper():<12} {v:.3f}")
    for k, v in resumen.get("generacion", {}).items():
        print(f"  {k:<38} {v}")
    print(f"\nDetalle por pregunta: {ruta_csv}")
    print(f"Resumen:              {ruta_resumen}")
    print(f"Historial:            {hist}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        raise SystemExit(f"[error] {e}")
