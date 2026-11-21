"""
Paso 3: responde preguntas usando solo los fragmentos recuperados.

Uso:
    python preguntar.py "¿Cuántos días de vacaciones me corresponden?"
    python preguntar.py                      # modo interactivo
    python preguntar.py "..." --fragmentos   # muestra también los fragmentos usados
    python preguntar.py "..." --solo-recuperar   # no llama al LLM
"""
import argparse

from config import agregar_argumentos_comunes, config_desde_args, consola_utf8
from rag.sistema import MiniRAG


def mostrar_fragmentos(resultados):
    print("\nFragmentos recuperados:")
    for r in resultados:
        f = r.fragmento
        detalle = ", ".join(f"{k}={v}" for k, v in r.detalle.items())
        texto = f.texto.replace("\n", " ")
        print(f"  {r.rango}. [{f.documento}] ({detalle})\n     {texto[:220]}{'...' if len(texto) > 220 else ''}")


def responder(rag: MiniRAG, pregunta: str, ver_fragmentos: bool, solo_recuperar: bool):
    if solo_recuperar:
        mostrar_fragmentos(rag.recuperar(pregunta))
        return
    r = rag.responder(pregunta)
    print("\nRespuesta:\n" + r.respuesta_limpia)
    if r.sin_informacion:
        print("\nDocumento fuente: (no se encontró la información)")
    else:
        nota = "  (inferida: el modelo no citó, se usa el fragmento más relevante)" if r.fuentes_inferidas else ""
        print(f"\nDocumento fuente: {', '.join(r.fuentes)}{nota}")
        titulos = {x.fragmento.documento: x.fragmento for x in r.resultados}
        for doc in r.fuentes:
            f = titulos.get(doc)
            if f:
                print(f"  - {f.titulo}" + (f" | {f.fuente}" if f.fuente else ""))
    print(f"\n(recuperación {r.tiempo_recuperacion_s:.2f} s, generación {r.tiempo_generacion_s:.1f} s)")
    if ver_fragmentos:
        mostrar_fragmentos(r.resultados)


def main():
    consola_utf8()
    parser = argparse.ArgumentParser(description="Pregunta al Mini-RAG")
    parser.add_argument("pregunta", nargs="?", help="Pregunta (si se omite, modo interactivo)")
    parser.add_argument("--fragmentos", action="store_true", help="Muestra los fragmentos recuperados")
    parser.add_argument("--solo-recuperar", action="store_true", help="Solo recuperación, sin LLM")
    parser.add_argument("--reconstruir", action="store_true", help="Fuerza reconstruir el índice")
    agregar_argumentos_comunes(parser)
    args = parser.parse_args()
    cfg = config_desde_args(args)

    rag = MiniRAG(cfg, reconstruir=args.reconstruir)

    if args.pregunta:
        responder(rag, args.pregunta, args.fragmentos, args.solo_recuperar)
        return

    print("\nMini-RAG de derecho laboral chileno. Escribe tu pregunta ('salir' para terminar).")
    while True:
        try:
            pregunta = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if pregunta.lower() in {"salir", "exit", "quit", ""}:
            break
        try:
            responder(rag, pregunta, args.fragmentos, args.solo_recuperar)
        except RuntimeError as e:
            print(f"[error] {e}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        raise SystemExit(f"[error] {e}")
