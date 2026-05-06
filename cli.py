#!/usr/bin/env python3
"""
CaloFit CLI — Punto de entrada centralizado.

Uso dentro del contenedor Docker:
    python cli.py audit           # Auditoría de integridad
    python cli.py audit --json    # Auditoría en JSON
    python cli.py cleanup         # Limpieza (dry run)
    python cli.py cleanup --fix   # Limpieza real
    python cli.py health          # Health check
    python cli.py stats           # Estadísticas del sistema
    python cli.py version         # Versión del sistema

Desde Windows (fuera del contenedor):
    docker exec calofit_backend python cli.py audit
    docker exec calofit_backend python cli.py health
    docker exec calofit_backend python cli.py stats
"""
from __future__ import annotations

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import click
from rich.console import Console

_IS_TTY = sys.stdout.isatty()
console = Console(force_terminal=_IS_TTY, highlight=False)


def _get_db():
    """Obtiene sesión de BD de producción."""
    from app.core.database import SessionLocal
    return SessionLocal()


def _print_json(data: dict) -> None:
    """Imprime dict como JSON formateado."""
    print(json.dumps(data, indent=2, ensure_ascii=False, default=str))


# ─────────────────────────────────────────────────────────────────────────────
# CLI GROUP
# ─────────────────────────────────────────────────────────────────────────────

@click.group()
def cli():
    """CaloFit CLI — Herramientas de operación del backend."""
    pass


# ─────────────────────────────────────────────────────────────────────────────
# audit
# ─────────────────────────────────────────────────────────────────────────────

@cli.command("audit")
@click.option("--json", "use_json", is_flag=True, default=False, help="Exportar en JSON")
@click.option("--output", "-o", default=None, help="Guardar reporte en archivo")
def cmd_audit(use_json: bool, output: str):
    """
    Auditoria de integridad de la base de datos.

    Detecta: platos sin ingredientes, macros invalidas, cache expirado.
    Exit 0 = sin problemas. Exit 2 = problemas encontrados.
    """
    console.print("CaloFit Audit — iniciando...")
    db = _get_db()
    try:
        from scripts.audit import ejecutar_auditoria, imprimir_reporte
        reporte = ejecutar_auditoria(db)

        if use_json or output or not _IS_TTY:
            json_str = json.dumps(reporte, indent=2, ensure_ascii=False, default=str)
            if output:
                with open(output, "w", encoding="utf-8") as f:
                    f.write(json_str)
                console.print(f"Reporte guardado en: {output}")
            else:
                print(json_str)
        else:
            imprimir_reporte(reporte)

        if reporte["total_problemas"] > 0:
            sys.exit(2)

    except SystemExit:
        raise
    except Exception as e:
        console.print(f"Error en auditoria: {e}")
        sys.exit(1)
    finally:
        db.close()


# ─────────────────────────────────────────────────────────────────────────────
# cleanup
# ─────────────────────────────────────────────────────────────────────────────

@cli.command("cleanup")
@click.option("--fix", is_flag=True, default=False, help="Aplicar cambios reales (default = dry run)")
@click.option("--json", "use_json", is_flag=True, default=False, help="Exportar en JSON")
def cmd_cleanup(fix: bool, use_json: bool):
    """
    Limpieza de la base de datos.

    Por defecto hace dry run (solo reporta sin cambios).
    Usa --fix para aplicar cambios reales.
    """
    modo = "REAL" if fix else "DRY RUN"
    console.print(f"CaloFit Cleanup — modo {modo}")

    if fix and _IS_TTY:
        if not click.confirm("Esto modificara datos en la BD. Continuar?", default=False):
            console.print("Operacion cancelada.")
            sys.exit(0)

    db = _get_db()
    try:
        from scripts.cleanup import ejecutar_limpieza, imprimir_resumen_limpieza
        resumen = ejecutar_limpieza(db, dry_run=not fix)

        if use_json or not _IS_TTY:
            _print_json(resumen)
        else:
            imprimir_resumen_limpieza(resumen)

    except Exception as e:
        console.print(f"Error en limpieza: {e}")
        sys.exit(1)
    finally:
        db.close()


# ─────────────────────────────────────────────────────────────────────────────
# health
# ─────────────────────────────────────────────────────────────────────────────

@cli.command("health")
@click.option("--json", "use_json", is_flag=True, default=False, help="Exportar en JSON")
def cmd_health(use_json: bool):
    """
    Health check completo del sistema.

    Verifica: DB, modelos ML, LLM, validadores, servicios, API routes.
    Exit 0 = OK. Exit 1 = ERROR. Exit 2 = WARN.
    """
    console.print("CaloFit Health Check")
    db = _get_db()
    try:
        from scripts.check_health import verificar_sistema, imprimir_health_check
        reporte = verificar_sistema(db)

        if use_json or not _IS_TTY:
            _print_json(reporte)
        else:
            imprimir_health_check(reporte)

        st = reporte["status_global"]
        if st == "ERROR":
            sys.exit(1)
        elif st == "WARN":
            sys.exit(2)

    except SystemExit:
        raise
    except Exception as e:
        console.print(f"Error en health check: {e}")
        sys.exit(1)
    finally:
        db.close()


# ─────────────────────────────────────────────────────────────────────────────
# stats
# ─────────────────────────────────────────────────────────────────────────────

@cli.command("stats")
@click.option("--json", "use_json", is_flag=True, default=False, help="Exportar en JSON")
def cmd_stats(use_json: bool):
    """
    Estadisticas del sistema.

    Muestra: totales, promedios nutricionales (7d), top alimentos, ejercicio.
    """
    console.print("CaloFit Stats")
    db = _get_db()
    try:
        from scripts.stats import obtener_estadisticas, imprimir_estadisticas
        stats = obtener_estadisticas(db)

        if use_json or not _IS_TTY:
            _print_json(stats)
        else:
            imprimir_estadisticas(stats)

    except Exception as e:
        console.print(f"Error obteniendo stats: {e}")
        sys.exit(1)
    finally:
        db.close()


# ─────────────────────────────────────────────────────────────────────────────
# version
# ─────────────────────────────────────────────────────────────────────────────

@cli.command("version")
def cmd_version():
    """Muestra la version del sistema CaloFit."""
    info = {
        "nombre": "CaloFit Backend",
        "version": "1.0.0",
        "stack": "FastAPI + PostgreSQL + Groq + ML (RF + KNN)",
        "tests": "83 passing",
        "cobertura": "83%+ en servicios criticos",
    }
    if _IS_TTY:
        from rich.panel import Panel
        console.print(Panel(
            f"[bold cyan]{info['nombre']}[/bold cyan]\n"
            f"Version: [green]{info['version']}[/green]\n"
            f"Stack: {info['stack']}\n"
            f"Tests: {info['tests']} | Cobertura: {info['cobertura']}",
            title="CaloFit",
            border_style="cyan",
        ))
    else:
        _print_json(info)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cli()
