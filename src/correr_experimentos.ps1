# Experimentos con LLM (requiere LM Studio en "Running", gemma-3n-e4b cargado).
# Ejecutar desde la carpeta src/ con el entorno virtual activado:   .\correr_experimentos.ps1
# Cada corrida agrega una fila a resultados\historial_resumenes.csv. Duración total: ~30-60 min en CPU.
$ErrorActionPreference = "Continue"

# 1) Recuperación sin LLM, incluyendo embeddings (denso e híbrido). Descarga el modelo HF la primera vez.
python comparar.py --con-embeddings

# 2) Generación con LLM. Set principal (preguntas.csv) y set de prueba con paráfrasis.
$sets = @("preguntas.csv", "preguntas_prueba.csv")
$configs = @(
    "--modo bm25 --k 5",
    "--modo tfidf_char --k 5",
    "--modo hibrido --k 5",
    "--modo hibrido --k 1",
    "--modo hibrido --k 3",
    "--modo hibrido --k 5 --prompt mejorado",
    "--modo hibrido_tfidf --stemming --expandir --k 5 --prompt mejorado"
)
foreach ($s in $sets) {
    foreach ($c in $configs) {
        Write-Host "=== $s :: $c ===" -ForegroundColor Cyan
        python evaluar.py --csv $s --salida "resultados\$($s -replace '\.csv','')" $c.Split(" ")
    }
}

# 3) Tabla resumen en Markdown
python resumir_generacion.py
