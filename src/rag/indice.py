"""
Índice de recuperación: BM25 (léxico) + embeddings (semántico), combinados
con Reciprocal Rank Fusion (RRF). El índice se guarda en disco y se reconstruye
automáticamente si cambian los documentos o la configuración.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

from .documentos import Fragmento, cargar_documentos, fragmentar_documentos
from .embeddings import Embedder, crear_embedder, nombre_modelo_embeddings
from functools import partial

from .texto import expandir_consulta, normalizar, tokenizar


@dataclass
class Resultado:
    fragmento: Fragmento
    puntaje: float
    rango: int                       # 1 = más relevante
    detalle: dict = field(default_factory=dict)


def calcular_firma(cfg) -> str:
    h = hashlib.sha256()
    for ruta in sorted(Path(cfg.carpeta_docs).glob("*.txt"), key=lambda p: p.name.lower()):
        h.update(ruta.name.encode("utf-8"))
        h.update(ruta.read_bytes())
    h.update(json.dumps([
        cfg.estrategia_fragmentacion, cfg.max_caracteres, cfg.solapamiento_caracteres,
        nombre_modelo_embeddings(cfg),
    ]).encode("utf-8"))
    return h.hexdigest()[:16]


class Indice:
    def __init__(self, fragmentos: list[Fragmento], embeddings: np.ndarray | None,
                 embedder: Embedder | None, rrf_k: int = 60,
                 stem: bool = False, expandir: bool = False):
        self.stem = stem
        self.expandir = expandir
        self.fragmentos = fragmentos
        self.embeddings = embeddings
        self.embedder = embedder
        self.rrf_k = rrf_k
        self.bm25 = BM25Okapi([tokenizar(f.texto_indexado, stem=stem) or ["_"] for f in fragmentos])
        self.documentos = sorted({f.documento for f in fragmentos})
        # TF-IDF (reemplazo liviano de los embeddings): se calcula en memoria, es instantáneo.
        textos = [f.texto_indexado for f in fragmentos]
        self.tfidf = TfidfVectorizer(tokenizer=partial(tokenizar, stem=stem), lowercase=False, token_pattern=None,
                                     sublinear_tf=True, ngram_range=(1, 2))
        self.m_tfidf = self.tfidf.fit_transform(textos)
        # Variante por caracteres: tolera plurales, conjugaciones y erratas.
        self.tfidf_c = TfidfVectorizer(preprocessor=normalizar, analyzer="char_wb",
                                       ngram_range=(3, 5), sublinear_tf=True)
        self.m_tfidf_c = self.tfidf_c.fit_transform(textos)

    # ------------------------------------------------------------------ #
    def buscar(self, consulta: str, k: int = 5, modo: str = "hibrido") -> list[Resultado]:
        n = len(self.fragmentos)
        k = max(1, min(k, n))
        if self.expandir:
            consulta = expandir_consulta(consulta)
        if modo in ("denso", "hibrido") and (self.embeddings is None or self.embedder is None):
            if modo == "denso":
                raise RuntimeError("El índice no tiene embeddings: usa --modo bm25 o hibrido")
            modo = "bm25"

        if modo in ("tfidf", "tfidf_char", "hibrido_tfidf"):
            return self._buscar_tfidf(consulta, k, modo)

        p_bm25 = p_denso = None
        if modo in ("bm25", "hibrido"):
            p_bm25 = np.asarray(self.bm25.get_scores(tokenizar(consulta, stem=self.stem)), dtype=np.float32)
        if modo in ("denso", "hibrido"):
            p_denso = self.embeddings @ self.embedder.codificar_consulta(consulta)

        if modo == "bm25":
            final = p_bm25
        elif modo == "denso":
            final = p_denso
        else:
            final = np.zeros(n, dtype=np.float32)
            orden_d = np.argsort(-p_denso)
            final[orden_d] += 1.0 / (self.rrf_k + np.arange(1, n + 1))
            orden_b = [i for i in np.argsort(-p_bm25) if p_bm25[i] > 0]
            for r, i in enumerate(orden_b, start=1):
                final[i] += 1.0 / (self.rrf_k + r)

        orden = np.argsort(-final, kind="stable")[:k]
        resultados = []
        for r, i in enumerate(orden, start=1):
            detalle = {}
            if p_bm25 is not None:
                detalle["bm25"] = round(float(p_bm25[i]), 4)
            if p_denso is not None:
                detalle["coseno"] = round(float(p_denso[i]), 4)
            resultados.append(Resultado(self.fragmentos[i], float(final[i]), r, detalle))
        return resultados

    def _buscar_tfidf(self, consulta: str, k: int, modo: str) -> list[Resultado]:
        n = len(self.fragmentos)
        p_w = (self.m_tfidf @ self.tfidf.transform([consulta]).T).toarray().ravel()
        p_c = (self.m_tfidf_c @ self.tfidf_c.transform([consulta]).T).toarray().ravel()
        if modo == "tfidf":
            final = p_w
        elif modo == "tfidf_char":
            final = p_c
        else:  # hibrido_tfidf: RRF entre BM25 y TF-IDF por caracteres
            p_b = np.asarray(self.bm25.get_scores(tokenizar(consulta, stem=self.stem)), dtype=np.float32)
            final = np.zeros(n, dtype=np.float32)
            for pun in (p_b, p_c):
                for r, i in enumerate((i for i in np.argsort(-pun, kind="stable") if pun[i] > 0), start=1):
                    final[i] += 1.0 / (self.rrf_k + r)
        orden = np.argsort(-final, kind="stable")[:k]
        return [Resultado(self.fragmentos[i], float(final[i]), r,
                          {"tfidf": round(float(p_w[i]), 4), "tfidf_char": round(float(p_c[i]), 4)})
                for r, i in enumerate(orden, start=1)]

    # ------------------------------------------------------------------ #
    def guardar(self, carpeta: Path, meta: dict) -> None:
        carpeta.mkdir(parents=True, exist_ok=True)
        with open(carpeta / "fragmentos.json", "w", encoding="utf-8") as f:
            json.dump([fr.a_dict() for fr in self.fragmentos], f, ensure_ascii=False, indent=1)
        if self.embeddings is not None:
            np.save(carpeta / "embeddings.npy", self.embeddings)
        elif (carpeta / "embeddings.npy").exists():
            (carpeta / "embeddings.npy").unlink()
        with open(carpeta / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)


def construir_indice(cfg, embedder: Embedder | None = None) -> Indice:
    t0 = time.time()
    docs = cargar_documentos(cfg.carpeta_docs)
    fragmentos = fragmentar_documentos(docs, cfg.estrategia_fragmentacion,
                                       cfg.max_caracteres, cfg.solapamiento_caracteres)
    print(f"[índice] {len(docs)} documentos -> {len(fragmentos)} fragmentos "
          f"(estrategia '{cfg.estrategia_fragmentacion}', máx. {cfg.max_caracteres} caracteres)")
    if embedder is None:
        embedder = crear_embedder(cfg)
    embeddings = None
    if embedder is not None:
        print("[índice] Calculando embeddings...")
        embeddings = embedder.codificar_pasajes([f.texto_indexado for f in fragmentos], progreso=True)
    indice = Indice(fragmentos, embeddings, embedder, cfg.rrf_k,
                  getattr(cfg, 'stemming', False), getattr(cfg, 'expandir_consulta', False))
    meta = {
        "firma": calcular_firma(cfg),
        "n_documentos": len(docs),
        "n_fragmentos": len(fragmentos),
        "estrategia": cfg.estrategia_fragmentacion,
        "max_caracteres": cfg.max_caracteres,
        "modelo_embeddings": nombre_modelo_embeddings(cfg),
        "creado": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    indice.guardar(cfg.carpeta_indice, meta)
    print(f"[índice] Guardado en {cfg.carpeta_indice} ({time.time() - t0:.1f} s)")
    return indice


def cargar_o_construir(cfg, reconstruir: bool = False) -> Indice:
    carpeta = Path(cfg.carpeta_indice)
    meta_ruta = carpeta / "meta.json"
    firma = calcular_firma(cfg)
    if not reconstruir and meta_ruta.exists():
        meta = json.loads(meta_ruta.read_text(encoding="utf-8"))
        if meta.get("firma") == firma:
            datos = json.loads((carpeta / "fragmentos.json").read_text(encoding="utf-8"))
            fragmentos = [Fragmento(**d) for d in datos]
            embedder = crear_embedder(cfg)
            emb_ruta = carpeta / "embeddings.npy"
            embeddings = np.load(emb_ruta) if (embedder is not None and emb_ruta.exists()) else None
            print(f"[índice] Cargado desde disco: {len(fragmentos)} fragmentos.")
            return Indice(fragmentos, embeddings, embedder, cfg.rrf_k,
                  getattr(cfg, 'stemming', False), getattr(cfg, 'expandir_consulta', False))
        print("[índice] Los documentos o la configuración cambiaron: se reconstruye.")
    return construir_indice(cfg)
