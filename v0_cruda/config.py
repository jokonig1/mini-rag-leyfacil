"""
Configuración central del Mini-RAG.

Todos los valores pueden cambiarse aquí o, para los más usados, desde la línea
de comandos de cada script (ver `--help`).
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

RAIZ = Path(__file__).resolve().parent

# En Windows, Hugging Face muestra un aviso molesto sobre symlinks: lo silenciamos.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@dataclass
class Config:
    # --- Rutas ---------------------------------------------------------------
    carpeta_docs: Path = RAIZ / "docs"
    carpeta_indice: Path = RAIZ / "indice"
    carpeta_resultados: Path = RAIZ / "resultados"

    # --- Fragmentación (chunking) -------------------------------------------
    # "qa"      -> un fragmento por par pregunta/respuesta (recomendado para estas guías)
    # "ventana" -> ventana deslizante de tamaño fijo con solapamiento (línea base)
    estrategia_fragmentacion: str = "qa"
    max_caracteres: int = 1200          # tamaño máximo de cada fragmento
    solapamiento_caracteres: int = 200  # solo se usa con "ventana"

    # --- Embeddings (búsqueda semántica) ------------------------------------
    # "sentence-transformers" -> modelo local descargado de Hugging Face (1.ª vez)
    # "lmstudio"              -> endpoint /v1/embeddings de LM Studio
    # "ninguno"               -> solo BM25 (léxico), sin embeddings
    backend_embeddings: str = "sentence-transformers"
    modelo_embeddings: str = "intfloat/multilingual-e5-small"
    modelo_embeddings_lmstudio: str = "text-embedding-nomic-embed-text-v1.5"

    # --- Recuperación -------------------------------------------------------
    modo_busqueda: str = "hibrido"  # "bm25" | "denso" | "hibrido"
    top_k: int = 5                  # fragmentos que se entregan al LLM
    rrf_k: int = 60                 # constante de Reciprocal Rank Fusion

    # --- LLM (LM Studio, API compatible con OpenAI) -------------------------
    lmstudio_url: str = os.environ.get("LMSTUDIO_URL", "http://localhost:1234/v1")
    # "auto" usa el modelo de chat disponible en LM Studio (prefiere uno "gemma").
    # También puedes poner el identificador exacto, p. ej. "google/gemma-3-4b".
    modelo_llm: str = os.environ.get("LMSTUDIO_MODELO", "auto")
    temperatura: float = 0.1
    max_tokens: int = 400
    timeout_s: float = 180.0

    # --- Evaluación ---------------------------------------------------------
    ks_evaluacion: list[int] = field(default_factory=lambda: [1, 3, 5])
    usar_juez_llm: bool = True      # el LLM local juzga si la respuesta es correcta
    umbral_similitud: float = 0.85  # criterio de respaldo si no hay juez
    umbral_f1: float = 0.50


def agregar_argumentos_comunes(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("configuración común")
    g.add_argument("--docs", type=Path, help="Carpeta con los .txt (por defecto: docs/)")
    g.add_argument("--modo", choices=["bm25", "denso", "hibrido"], help="Tipo de búsqueda")
    g.add_argument("--k", type=int, help="Número de fragmentos que recibe el LLM")
    g.add_argument("--estrategia", choices=["qa", "ventana"], help="Estrategia de fragmentación")
    g.add_argument("--max-caracteres", type=int, help="Tamaño máximo de fragmento")
    g.add_argument("--embeddings", choices=["sentence-transformers", "lmstudio", "ninguno"],
                   help="Backend de embeddings")
    g.add_argument("--modelo-embeddings", help="Nombre del modelo de embeddings")
    g.add_argument("--modelo-llm", help="Identificador del modelo en LM Studio (o 'auto')")
    g.add_argument("--url", help="URL de LM Studio (por defecto http://localhost:1234/v1)")


def config_desde_args(args: argparse.Namespace) -> Config:
    cfg = Config()
    if getattr(args, "docs", None):
        cfg.carpeta_docs = args.docs.resolve()
    if getattr(args, "modo", None):
        cfg.modo_busqueda = args.modo
    if getattr(args, "k", None):
        cfg.top_k = args.k
    if getattr(args, "estrategia", None):
        cfg.estrategia_fragmentacion = args.estrategia
    if getattr(args, "max_caracteres", None):
        cfg.max_caracteres = args.max_caracteres
    if getattr(args, "embeddings", None):
        cfg.backend_embeddings = args.embeddings
    if getattr(args, "modelo_embeddings", None):
        if cfg.backend_embeddings == "lmstudio":
            cfg.modelo_embeddings_lmstudio = args.modelo_embeddings
        else:
            cfg.modelo_embeddings = args.modelo_embeddings
    if getattr(args, "modelo_llm", None):
        cfg.modelo_llm = args.modelo_llm
    if getattr(args, "url", None):
        cfg.lmstudio_url = args.url
    if cfg.backend_embeddings == "ninguno" and cfg.modo_busqueda != "bm25":
        print("[aviso] Sin embeddings solo es posible la búsqueda BM25; se usará modo 'bm25'.")
        cfg.modo_busqueda = "bm25"
    return cfg


def consola_utf8() -> None:
    """Evita errores con tildes/ñ en la consola de Windows."""
    for flujo in (sys.stdout, sys.stderr):
        try:
            flujo.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
