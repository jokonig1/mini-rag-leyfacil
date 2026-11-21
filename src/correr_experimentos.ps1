$ErrorActionPreference = "Continue"
python comparar.py --con-embeddings
function Correr($set, $opciones) {
    $nombre = $set -replace '\.csv',''
    Write-Host "=== $set :: $opciones ===" -ForegroundColor Cyan
    $a = $opciones -split ' '
    python evaluar.py --csv $set --salida "resultados\$nombre" @a
}
Correr "preguntas.csv" "--modo hibrido --k 5"
Correr "preguntas.csv" "--modo hibrido --k 5 --prompt mejorado"
Correr "preguntas.csv" "--modo hibrido --k 3"
Correr "preguntas.csv" "--modo hibrido --k 1"
Correr "preguntas.csv" "--modo tfidf_char --k 5"
Correr "preguntas.csv" "--modo bm25 --k 5"
Correr "preguntas_prueba.csv" "--modo hibrido --k 5"
Correr "preguntas_prueba.csv" "--modo tfidf_char --k 5"
Correr "preguntas_prueba.csv" "--modo hibrido --k 5 --prompt mejorado"
python resumir_generacion.py
