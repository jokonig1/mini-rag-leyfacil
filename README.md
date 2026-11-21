# Mini-RAG Ley Fácil: derecho laboral chileno

Proyecto de Procesamiento del Lenguaje Natural (UAI, 2026-1) para la start-up Lang-Chi.
Sistema que responde preguntas sobre derecho laboral chileno a partir de 37 guías de Ley Fácil
(Biblioteca del Congreso Nacional), recuperando primero los fragmentos relevantes y usando
solo un modelo local (gemma-3n-e4b en LM Studio).

## Estructura

```
docs/            37 guías de Ley Fácil en .txt (datos de entrada)
v0_cruda/        versión 0 generada por IA. NO SE MODIFICA (solo se le agrega preguntas.csv y resultados/)
src/             versión mejorada por el equipo (copia de v0 con cambios)
  comparar.py              compara configuraciones de recuperación en segundos (sin LLM)
  evaluar.py               evaluación completa (Recall@k, MRR, juez LLM, abstención)
  correr_experimentos.ps1  corre todos los experimentos con LLM
  preguntas*.csv           sets A, B y C de evaluación
```

## Etapas del enunciado

1. **Versión cruda con IA** (`v0_cruda/`): prompt y modelo en `v0_cruda/prompt.md`. Sin modificaciones.
2. **Explicar y simplificar** (sección siguiente): cada parte de la v0 y su comparación con alternativas simples.
3. **Repositorio con historial**: commits descriptivos por cada cambio.
4. **Presentación de 15 minutos**: qué entregó la IA, qué cambiamos y por qué.

## Qué hace cada parte de la v0 y qué probamos

| Parte de la v0 | Qué hace | Alternativa simple que comparamos |
|---|---|---|
| Fragmentación `qa` | Un fragmento por par pregunta-respuesta; si es muy largo, lo parte repitiendo la pregunta | Ventana fija de 400, 800 y 1200 caracteres con solapamiento |
| BM25 | Búsqueda léxica con tokens sin tildes ni stopwords | TF-IDF por palabras (1-2 gramas) y TF-IDF por caracteres (3-5 gramas) |
| Embeddings `multilingual-e5-small` | Búsqueda semántica (descarga ~470 MB de Hugging Face) | Reemplazo por TF-IDF, que no necesita descargas ni PyTorch |
| Híbrido RRF | Fusiona los rankings de BM25 y embeddings | `hibrido_tfidf`: fusiona BM25 y TF-IDF por caracteres |
| Prompt | Instrucciones de responder solo con el contexto y citar la fuente | `--prompt mejorado`: reglas más estrictas, ejemplo de abstención y máximo 3 oraciones |
| Evaluación | Recall@k, MRR y juez LLM | Se agrega el tratamiento de preguntas sin respuesta (el sistema acierta si se abstiene) |
| (nuevo) Normalización | No existía | Stemming Snowball en español (`--stemming`) y expansión de consulta con un léxico coloquial-legal (`--expandir`) |

## Preguntas de evaluación

- **Set A** (`preguntas.csv`): 24 preguntas, 21 con respuesta y 3 sin respuesta en los documentos. Redactadas cerca del texto, por lo que son fáciles.
- **Set B** (`preguntas_dificiles.csv`): las mismas 21 respuestas con paráfrasis y lenguaje coloquial, más 3 sin respuesta. **Es el set de desarrollo**: el léxico de sinónimos se diseñó mirando este tipo de preguntas.
- **Set C** (`preguntas_prueba.csv`): 12 preguntas con respuesta sobre otros documentos y 2 sin respuesta. **Es el set de prueba**: se escribió aparte y no se usó para diseñar nada. Al armarlo se detectó que dos entradas del léxico coincidían con palabras de sus preguntas; se quitaron y se volvió a medir.

Las métricas de recuperación se calculan solo sobre las preguntas que sí tienen respuesta (21, 21 y 12).

## Resultados de recuperación (sin LLM, reproducibles con `python comparar.py`)

### Tabla 1. Modo de búsqueda (fragmentación qa-1200, sin stemming ni expansión)

| Modo          |   R@1 A |   MRR A |   R@1 B |   MRR B |   R@1 C |   MRR C |
|:--------------|--------:|--------:|--------:|--------:|--------:|--------:|
| bm25          |   0.952 |   0.976 |   0.524 |   0.597 |   0.667 |   0.776 |
| hibrido_tfidf |   0.952 |   0.968 |   0.571 |   0.644 |   0.583 |   0.753 |
| tfidf         |   0.952 |   0.968 |   0.476 |   0.562 |   0.75  |   0.831 |
| tfidf_char    |   0.952 |   0.968 |   0.619 |   0.718 |   0.667 |   0.776 |

### Tabla 2. Estrategia de fragmentación (promedio de bm25, tfidf, tfidf_char e hibrido_tfidf)

| Fragmentación   |   n° fragmentos |   MRR A |   MRR B |   MRR C |
|:----------------|----------------:|--------:|--------:|--------:|
| qa-1200         |             428 |   0.97  |   0.63  |   0.784 |
| ventana-1200    |             263 |   0.976 |   0.589 |   0.851 |
| ventana-400     |             857 |   0.976 |   0.582 |   0.792 |
| ventana-800     |             430 |   0.952 |   0.56  |   0.84  |

### Tabla 3. Stemming y expansión de consulta (promedio de 4 modos, qa-1200)

| Variante       |   MRR A |   MRR B |   MRR C |
|:---------------|--------:|--------:|--------:|
| base           |   0.97  |   0.63  |   0.784 |
| expansion      |   0.97  |   0.66  |   0.784 |
| stem           |   0.992 |   0.676 |   0.763 |
| stem+expansion |   0.992 |   0.686 |   0.763 |

### Tabla 4. Recall@k (k = 1, 3, 5), qa-1200

| Set   | Modo          |   R@1 |   R@3 |   R@5 |
|:------|:--------------|------:|------:|------:|
| A     | bm25          | 0.952 | 1     | 1     |
| A     | hibrido_tfidf | 0.952 | 1     | 1     |
| A     | tfidf         | 0.952 | 1     | 1     |
| A     | tfidf_char    | 0.952 | 1     | 1     |
| B     | bm25          | 0.524 | 0.571 | 0.714 |
| B     | hibrido_tfidf | 0.571 | 0.714 | 0.762 |
| B     | tfidf         | 0.476 | 0.619 | 0.667 |
| B     | tfidf_char    | 0.619 | 0.81  | 0.857 |
| C     | bm25          | 0.667 | 0.917 | 0.917 |
| C     | hibrido_tfidf | 0.583 | 0.917 | 1     |
| C     | tfidf         | 0.75  | 0.917 | 0.917 |
| C     | tfidf_char    | 0.667 | 0.917 | 0.917 |

### Qué muestra

- **Con preguntas fáciles (Set A) todo funciona**: Recall@3 = 1,0 en todos los modos. Con ese set no se puede distinguir un método de otro, por eso se agregaron B y C.
- **Con paráfrasis (B) baja mucho**: MRR entre 0,56 y 0,72. TF-IDF por caracteres es el mejor modo en B (MRR 0,72 contra 0,60 de BM25 y 0,56 de TF-IDF por palabras), porque tolera plurales y conjugaciones.
- **En el set de prueba (C) no hay un ganador claro**. Con solo 12 preguntas, una pregunta cambia el MRR en unos 0,08, así que las diferencias entre modos están dentro del ruido.
- **Stemming y expansión mejoran B (+0,03 a +0,06 de MRR) pero no C**: la expansión no cambia nada y el stemming empeora un poco. La mejora en B es parcialmente un efecto de haber diseñado el léxico mirando ese set, y no se confirma en el set de prueba.
- **La fragmentación `qa` no gana de forma clara**: en B es la mejor, en C las ventanas de 800 y 1200 son mejores. Recall se mide a nivel de documento, lo que favorece a las ventanas grandes.
- **Embeddings (denso e híbrido)**: no se midieron en este entorno por no tener acceso a Hugging Face. Hay que correr `python comparar.py --con-embeddings` en el computador del equipo y agregar las filas aquí.

## Resultados con LLM (pendiente de correr en LM Studio)

`src/correr_experimentos.ps1` corre generación con k = 1, 3 y 5, prompt `v0` contra `mejorado`, y los modos principales sobre los sets A y C, y deja la tabla en `src/resultados/tabla_generacion.md`. Pegar aquí esa tabla.

Métricas: exactitud (juez LLM, solo preguntas con respuesta), abstención correcta (preguntas sin respuesta), cita correcta y tiempo por respuesta.

## Limitaciones

- El juez es el mismo modelo pequeño que genera, por lo que puede ser indulgente. Revisar a mano una muestra de `respuesta_generada` y `motivo_juez`.
- Los sets son pequeños (12 a 21 preguntas con respuesta), y la mayoría de las diferencias entre configuraciones son menores que el ruido.
- Los documentos se contradicen sobre la jornada máxima: 45 horas en `horas-extraordinarias`, 44 en `trabajo-a-distancia` y 40 en `reduccion-de-la-jornada` y `jornada-especial`. El sistema puede recuperar cualquiera.
- Detectar preguntas sin respuesta con el puntaje del mejor fragmento sirve a medias (AUC 0,76 a 0,81 con 9 preguntas): el puntaje no basta para filtrar antes de llamar al LLM.

## Cómo reproducir

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r src\requirements.txt
cd src
python comparar.py                       # recuperación sin LLM
python probar_sin_lmstudio.py            # prueba el pipeline con un LLM falso
.\correr_experimentos.ps1                # experimentos completos (LM Studio en Running)
```
