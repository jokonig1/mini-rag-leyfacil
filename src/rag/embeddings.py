"""Modelos de embeddings locales (sentence-transformers o LM Studio)."""
from __future__ import annotations

import numpy as np


def _prefijos(modelo: str) -> tuple[str, str]:
    """Algunos modelos requieren prefijos distintos para consultas y pasajes."""
    m = modelo.lower()
    if "e5" in m:
        return "query: ", "passage: "
    if "nomic" in m:
        return "search_query: ", "search_document: "
    return "", ""


def _normalizar(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    normas = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(normas, 1e-12, None)


class Embedder:
    nombre: str = ""

    def __init__(self, nombre: str):
        self.nombre = nombre
        self.pref_consulta, self.pref_pasaje = _prefijos(nombre)

    def _codificar(self, textos: list[str], progreso: bool = False) -> np.ndarray:
        raise NotImplementedError

    def codificar_pasajes(self, textos: list[str], progreso: bool = False) -> np.ndarray:
        return self._codificar([self.pref_pasaje + t for t in textos], progreso)

    def codificar_consulta(self, texto: str) -> np.ndarray:
        return self._codificar([self.pref_consulta + texto])[0]

    def similitud(self, a: str, b: str) -> float:
        """Similitud coseno simétrica (para comparar respuestas)."""
        v = self._codificar([self.pref_consulta + a, self.pref_consulta + b])
        return float(v[0] @ v[1])


class EmbedderSentenceTransformers(Embedder):
    def __init__(self, nombre: str):
        super().__init__(nombre)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError(
                "Falta 'sentence-transformers'. Instala requirements.txt o usa --embeddings ninguno"
            ) from e
        print(f"[embeddings] Cargando modelo local '{nombre}' (la 1.ª vez se descarga)...")
        self.modelo = SentenceTransformer(nombre, device="cpu")

    def _codificar(self, textos, progreso=False):
        v = self.modelo.encode(textos, batch_size=32, show_progress_bar=progreso,
                               normalize_embeddings=True, convert_to_numpy=True)
        return v.astype(np.float32)


class EmbedderLMStudio(Embedder):
    def __init__(self, nombre: str, url: str, timeout: float = 120):
        super().__init__(nombre)
        from openai import OpenAI
        self.cliente = OpenAI(base_url=url, api_key="lm-studio", timeout=timeout)

    def _codificar(self, textos, progreso=False):
        vectores = []
        for i in range(0, len(textos), 32):
            r = self.cliente.embeddings.create(model=self.nombre, input=textos[i:i + 32])
            vectores.extend(d.embedding for d in r.data)
        return _normalizar(np.array(vectores))


def crear_embedder(cfg) -> Embedder | None:
    if cfg.backend_embeddings == "ninguno":
        return None
    if cfg.backend_embeddings == "lmstudio":
        return EmbedderLMStudio(cfg.modelo_embeddings_lmstudio, cfg.lmstudio_url, cfg.timeout_s)
    return EmbedderSentenceTransformers(cfg.modelo_embeddings)


def nombre_modelo_embeddings(cfg) -> str:
    if cfg.backend_embeddings == "ninguno":
        return "ninguno"
    if cfg.backend_embeddings == "lmstudio":
        return f"lmstudio:{cfg.modelo_embeddings_lmstudio}"
    return cfg.modelo_embeddings
