# Versión 0 – Código generado por IA (sin modificaciones)

- **IA utilizada:** Claude Opus 5.5
- **Fecha:** 19-11-2026
- **Link a la conversación:** 

## Prompt utilizado

```
Necesito un sistema Mini-RAG en Python para un proyecto universitario de Procesamiento del Lenguaje Natural.

Contexto:
- Tengo 37 documentos .txt en español en la carpeta docs/. Son guías legales chilenas sobre derecho laboral (Ley Fácil, Biblioteca del Congreso Nacional). Cada archivo empieza con "TÍTULO: ..." y "FUENTE: ...", seguido de preguntas y respuestas separadas por líneas en blanco.
- Solo puedo usar modelos locales. El LLM es gemma-3-4b corriendo en LM Studio, que expone una API compatible con OpenAI en http://localhost:1234/v1 google/gemma-3n-e4b.
- Trabajo en Windows con Python 3.11.

El sistema debe:
1. Cargar los documentos y dividirlos en fragmentos (chunks).
2. Indexar los fragmentos para poder recuperar los más relevantes ante una pregunta.
3. Responder preguntas usando solo los fragmentos recuperados, indicando de qué documento sale la respuesta.
4. Tener un script de evaluación que lea preguntas.csv con las columnas: pregunta, respuesta_esperada, documento_esperado (nombre del archivo .txt). Debe medir:
   - si el documento correcto aparece entre los fragmentos recuperados (Recall@k y MRR),
   - si la respuesta generada es correcta comparada con la respuesta esperada.
   Debe guardar los resultados en un CSV.

Entrégame el código completo, organizado en archivos, el requirements.txt y las instrucciones para ejecutarlo.
```