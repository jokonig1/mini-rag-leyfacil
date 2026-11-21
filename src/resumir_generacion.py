"""Une los historial_resumenes.csv de resultados/*/ en una tabla Markdown (resultados/tabla_generacion.md)."""
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent
frames = []
for h in sorted((RAIZ / "resultados").glob("*/historial_resumenes.csv")):
    d = pd.read_csv(h, encoding="utf-8-sig")
    d.insert(0, "set", h.parent.name)
    frames.append(d)
if not frames:
    raise SystemExit("No hay historiales: corre primero correr_experimentos.ps1")
t = pd.concat(frames, ignore_index=True)
cols = [c for c in ["set", "modo_busqueda", "top_k_llm", "prompt", "recall@1", "recall@3", "recall@5", "mrr",
                    "exactitud (solo con respuesta)", "abstencion_correcta (solo sin respuesta)",
                    "cita_correcta", "tiempo_generacion_promedio_s"] if c in t.columns]
md = t[cols].round(3).to_markdown(index=False)
(RAIZ / "resultados" / "tabla_generacion.md").write_text(md, encoding="utf-8")
print(md)
