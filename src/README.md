# src/: versión mejorada

Copia de `v0_cruda/` con los cambios del equipo. La explicación completa, las tablas y las conclusiones están en el `README.md` de la raíz.

Opciones agregadas respecto de la v0:

```powershell
python evaluar.py --modo tfidf_char          # tfidf | tfidf_char | hibrido_tfidf (sin descargas) | bm25 | denso | hibrido
python evaluar.py --stemming --expandir      # stemming Snowball y expansión de consulta
python evaluar.py --prompt mejorado          # plantilla de prompt alternativa
python evaluar.py --csv preguntas_prueba.csv --salida resultados\prueba
python comparar.py [--con-embeddings]        # tabla comparativa de recuperación
python probar_sin_lmstudio.py                # prueba del pipeline con LLM falso
```

En los CSV, `documento_esperado = NINGUNO` marca una pregunta que no tiene respuesta en los documentos.
Para esas preguntas el sistema acierta si responde que no encuentra la información.

Los documentos se leen de `../docs/` si no existe `src/docs/`.
