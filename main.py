"""Punto de entrada del proyecto Evaluador de Informes."""

from __future__ import annotations


def main() -> None:
    """Inicia la interfaz de línea de comandos del Evaluador CEPLAN."""

    from cli.ceplan_cli import app

    app()


if __name__ == "__main__":  # pragma: no cover - ejecución directa
    main()