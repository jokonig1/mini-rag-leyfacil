"""Cliente para el LLM local servido por LM Studio (API compatible con OpenAI)."""
from __future__ import annotations

import re

from openai import APIConnectionError, APIStatusError, OpenAI


class ClienteLLM:
    def __init__(self, cfg):
        self.cfg = cfg
        self.cliente = OpenAI(base_url=cfg.lmstudio_url, api_key="lm-studio", timeout=cfg.timeout_s)
        self._modelo = None if cfg.modelo_llm in ("", "auto") else cfg.modelo_llm

    @property
    def modelo(self) -> str:
        if self._modelo is None:
            self._modelo = self._resolver_modelo()
        return self._modelo

    def _resolver_modelo(self) -> str:
        try:
            ids = [m.id for m in self.cliente.models.list().data]
        except APIConnectionError as e:
            raise RuntimeError(
                f"No se pudo conectar a LM Studio en {self.cfg.lmstudio_url}.\n"
                "  -> Abre LM Studio, ve a la pestaña 'Developer' y pulsa 'Start Server' (puerto 1234)."
            ) from e
        chat = [i for i in ids if "embed" not in i.lower()]
        if not chat:
            raise RuntimeError("LM Studio no tiene ningún modelo de chat disponible. Carga gemma-3-4b.")
        preferidos = [i for i in chat if "gemma" in i.lower()]
        elegido = (preferidos or chat)[0]
        print(f"[llm] Usando el modelo '{elegido}' de LM Studio.")
        return elegido

    def generar(self, prompt: str, temperatura: float | None = None,
                max_tokens: int | None = None) -> str:
        # Gemma no tiene un rol "system" propio: todo va en el mensaje del usuario.
        try:
            r = self.cliente.chat.completions.create(
                model=self.modelo,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.cfg.temperatura if temperatura is None else temperatura,
                max_tokens=max_tokens or self.cfg.max_tokens,
            )
        except APIConnectionError as e:
            raise RuntimeError(f"Se perdió la conexión con LM Studio ({self.cfg.lmstudio_url}).") from e
        except APIStatusError as e:
            raise RuntimeError(f"LM Studio devolvió un error {e.status_code}: {e.message}") from e
        texto = r.choices[0].message.content or ""
        return re.sub(r"<end_of_turn>|<eos>", "", texto).strip()
