# Mini-RAG · Derecho laboral chileno (Ley Fácil, BCN)

Sistema de *Retrieval-Augmented Generation* 100 % local: recupera fragmentos de
37 guías legales en español y genera respuestas con **gemma-3-4b** en LM Studio,
indicando el documento de origen. Incluye un script de evaluación (Recall@k,
MRR y exactitud de la respuesta).

## Estructura

```
mini_rag/
├── docs/                    # tus 37 archivos .txt (TÍTULO:, FUENTE:, preguntas y respuestas)
├── preguntas.csv            # tu set de evaluación (pregunta, respuesta_esperada, documento_esperado)
├── preguntas_ejemplo.csv    # ejemplo del formato
├── config.py                # toda la configuración (rutas, chunking, modelos, k...)
├── construir_indice.py      # 1-2. carga, fragmenta e indexa
├── preguntar.py             # 3. responde preguntas citando el documento
├── evaluar.py               # 4. evaluación -> resultados/*.csv
├── requirements.txt
└── rag/
    ├── texto.py             # normalización, tokenización, stopwords en español
    ├── documentos.py        # carga de .txt y fragmentación
    ├── embeddings.py        # embeddings locales (sentence-transformers o LM Studio)
    ├── indice.py            # BM25 + embeddings + fusión RRF, persistencia en disco
    ├── llm.py               # cliente de LM Studio (API OpenAI)
    ├── sistema.py           # prompt, generación y extracción de la fuente
    └── metricas.py          # Recall@k, MRR, F1, juez LLM
```

## Instalación (Windows, Python 3.11)

En PowerShell, dentro de la carpeta `mini_rag`:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> Si PowerShell bloquea `Activate.ps1`, ejecuta una vez
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` (o usa `.venv\Scripts\activate.bat` en CMD).
>
> `sentence-transformers` instala PyTorch (versión CPU, unos cientos de MB). La primera
> vez que construyas el índice se descarga el modelo de embeddings
> `intfloat/multilingual-e5-small` (~470 MB) y después funciona sin internet.

Copia tus 37 `.txt` en `docs/` y tu `preguntas.csv` en la raíz del proyecto.

## Configurar LM Studio

1. Descarga y carga **gemma-3-4b** (o `gemma-3n-e4b`) en LM Studio.
2. Pestaña **Developer** → **Start Server** (puerto 1234). Debe responder en
   `http://localhost:1234/v1`.
3. Recomendado: longitud de contexto de 4096 tokens o más (5 fragmentos ≈ 1.500–2.000 tokens).

Por defecto el sistema usa el modelo de chat que LM Studio tenga disponible
(prefiere uno con "gemma" en el nombre). Para fijar uno concreto:
`--modelo-llm google/gemma-3-4b` o la variable de entorno `LMSTUDIO_MODELO`.

## Uso

```powershell
# 1-2. Construir el índice (se reconstruye solo si cambian docs/ o la configuración)
python construir_indice.py

# 3. Preguntar
python preguntar.py "¿Cuántos días de vacaciones me corresponden?"
python preguntar.py "¿Qué es el feriado progresivo?" --fragmentos   # muestra los fragmentos usados
python preguntar.py                                                 # modo interactivo
python preguntar.py "..." --solo-recuperar                          # sin LLM

# 4. Evaluar
python evaluar.py                          # usa preguntas.csv, Recall@1/3/5, MRR, juez LLM
python evaluar.py --solo-recuperacion      # solo Recall@k y MRR (segundos, sin LM Studio)
python evaluar.py --sin-juez               # exactitud por similitud semántica / F1
python evaluar.py --ks 1,3,5,10 --limite 10
```

Ejemplo de salida de `preguntar.py`:

```
Respuesta:
Los trabajadores con más de un año de servicio tienen derecho a un feriado anual de quince días hábiles, con remuneración íntegra.

Documento fuente: feriado_anual.txt
  - Feriado anual o vacaciones | Ley Fácil, Biblioteca del Congreso Nacional
```

### Formato de `preguntas.csv`

```csv
pregunta,respuesta_esperada,documento_esperado
"¿Cuántos días de vacaciones...?","Quince días hábiles...","feriado_anual.txt"
```

- Se aceptan `,` o `;` como separador y UTF-8 o la codificación de Excel (cp1252).
- `documento_esperado` puede llevar o no la extensión `.txt`. Si la respuesta
  está en varios documentos, sepáralos con `;` o `|`.

### Archivos de resultados (`resultados/`)

| Archivo | Contenido |
|---|---|
| `resultados_<modo>_<estrategia>_k<k>_<fecha>.csv` | Una fila por pregunta: documentos y fragmentos recuperados, rango del documento correcto, `acierto@k`, rango recíproco, respuesta generada, documentos citados, `cita_correcta`, F1, similitud, veredicto y motivo del juez, `respuesta_correcta`, tiempos |
| `resumen_..._<fecha>.json` | Métricas agregadas y configuración usada |
| `historial_resumenes.csv` | Una fila por ejecución, para comparar configuraciones en el informe |

Los CSV se guardan en UTF-8 con BOM, así que Excel muestra bien las tildes.

## Cómo funciona

**1. Carga y fragmentación.** Se leen los `.txt` (UTF-8 o Windows-1252), se
extraen `TÍTULO` y `FUENTE` y el cuerpo se divide por líneas en blanco. Con la
estrategia `qa` (por defecto) cada fragmento es **un par pregunta-respuesta**: un
bloque que termina en `?` o empieza con `¿` abre un fragmento nuevo, y los párrafos
siguientes son su respuesta. Si una respuesta supera `max_caracteres` (1200), se
divide por oraciones repitiendo la pregunta al inicio de cada parte. La estrategia
`ventana` (tamaño fijo con solapamiento) queda como línea base para comparar.

**2. Indexación.** Cada fragmento se indexa con el título del documento antepuesto:
- **BM25** (léxico) con tokens en minúsculas, sin tildes ni stopwords.
- **Embeddings densos** con `multilingual-e5-small` (prefijos `query:`/`passage:`), similitud coseno.
- **Híbrido** (por defecto): fusión de ambos rankings con *Reciprocal Rank Fusion*,
  `score = Σ 1/(60 + rango)`.

El índice se guarda en `indice/` con una firma (hash de los documentos y de la
configuración) y se reconstruye solo cuando algo cambia.

**3. Generación.** Los `k = 5` mejores fragmentos se pasan a Gemma con
instrucciones de responder **solo** con ese contexto, de decir *"No encuentro esa
información en los documentos disponibles."* cuando no esté y de cerrar con
`Fuente: <archivo.txt>`. La fuente citada se valida contra los documentos
recuperados (por nombre o por "Fragmento N"); si el modelo no cita nada, se usa el
documento del fragmento mejor rankeado y se marca como `cita_inferida`. Se usa
temperatura 0,1 y todo va en el mensaje de usuario, porque Gemma no tiene rol *system*.

**4. Evaluación.**
- **Recall@k:** proporción de preguntas en que algún fragmento del documento
  esperado aparece entre los *k* primeros recuperados.
- **MRR:** media de `1/rango` del primer fragmento del documento correcto (0 si no aparece).
- **Respuesta correcta:** el mismo LLM local actúa de juez y clasifica la respuesta
  como CORRECTA / PARCIAL / INCORRECTA frente a la esperada (`respuesta_correcta` = 1
  solo si es CORRECTA; `puntaje_juez` da 0,5 a las parciales). Como métricas
  complementarias se calculan el **F1 de palabras** (estilo SQuAD) y la **similitud
  semántica** con el modelo de embeddings. Con `--sin-juez`, una respuesta se considera
  correcta si la similitud es ≥ 0,85 o el F1 es ≥ 0,5.
- **cita_correcta:** si el documento que cita el sistema es el esperado.

## Experimentos sugeridos para el informe

```powershell
python evaluar.py --solo-recuperacion --modo bm25
python evaluar.py --solo-recuperacion --modo denso
python evaluar.py --solo-recuperacion --modo hibrido
python evaluar.py --solo-recuperacion --estrategia ventana --max-caracteres 800
python evaluar.py --k 3
python evaluar.py --k 8
```

Todas las ejecuciones quedan en `resultados/historial_resumenes.csv`, listas para
hacer una tabla comparativa.

## Opciones de embeddings

| Opción | Cuándo usarla |
|---|---|
| `--embeddings sentence-transformers` (por defecto) | Calidad recomendada; descarga el modelo una vez |
| `--embeddings lmstudio --modelo-embeddings text-embedding-nomic-embed-text-v1.5` | Si prefieres que LM Studio también sirva los embeddings (cárgalo en LM Studio) |
| `--embeddings ninguno` | Solo BM25, sin PyTorch ni descargas |

Otro modelo de sentence-transformers que funciona bien en español es
`--modelo-embeddings intfloat/multilingual-e5-base` (mejor, pero más lento).

## Limitaciones conocidas

- El juez LLM es el mismo modelo pequeño que genera las respuestas, así que puede
  ser indulgente o inconsistente. Conviene revisar a mano una muestra (las columnas
  `motivo_juez` y `respuesta_generada` ayudan) y reportar también F1 y similitud.
- Recall@k se mide a nivel de documento: si un documento tiene varios fragmentos,
  basta con que aparezca uno de ellos.
- gemma-3-4b en CPU puede tardar varios segundos por respuesta; la evaluación
  completa hace 2 llamadas por pregunta (respuesta + juez).

## Problemas frecuentes

| Problema | Solución |
|---|---|
| `No se pudo conectar a LM Studio` | Inicia el servidor en la pestaña Developer; revisa el puerto 1234 |
| Error descargando el modelo de embeddings | Revisa la conexión la primera vez, o usa `--embeddings ninguno` / `lmstudio` |
| `Faltan columnas en preguntas.csv` | Los encabezados deben ser `pregunta,respuesta_esperada,documento_esperado` |
| `[aviso] Fila N: 'x.txt' no está en docs/` | El nombre en `documento_esperado` no coincide con ningún archivo |
| Tildes raras en la consola | Usa Windows Terminal o ejecuta `chcp 65001` antes |
