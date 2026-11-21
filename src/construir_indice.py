"""
Paso 1 y 2: carga los documentos de docs/, los fragmenta y construye el índice.

Uso:
    python construir_indice.py
    python construir_indice.py --estrategia ventana --max-caracteres 800
    python construir_indice.py --mostrar 5        # muestra 5 fragmentos de ejemplo
"""
import argparse
from statistics import mean

from config import agregar_argumentos_comunes, config_desde_args, consola_utf8
from rag.indice import construir_indice


def main():
    consola_utf8()
    parser = argparse.ArgumentParser(description="Construye el índice del Mini-RAG")
    agregar_argumentos_comunes(parser)
    parser.add_argument("--mostrar", type=int, default=3, help="Fragmentos de ejemplo a mostrar")
    args = parser.parse_args()
    cfg = config_desde_args(args)

    indice = construir_indice(cfg)
    largos = [len(f.texto) for f in indice.fragmentos]
    por_doc = {}
    for f in indice.fragmentos:
        por_doc[f.documento] = por_doc.get(f.documento, 0) + 1

    print("\n=== Estadísticas ===")
    print(f"Documentos:            {len(por_doc)}")
    print(f"Fragmentos:            {len(largos)}")
    print(f"Largo medio / mín / máx (caracteres): {mean(largos):.0f} / {min(largos)} / {max(largos)}")
    print(f"Fragmentos por documento (media): {mean(por_doc.values()):.1f}")

    for f in indice.fragmentos[:args.mostrar]:
        print(f"\n--- {f.id} ({f.documento}) ---\n{f.texto[:400]}{'...' if len(f.texto) > 400 else ''}")


if __name__ == "__main__":
    main()
