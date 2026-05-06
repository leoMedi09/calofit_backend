"""
Script de estadísticas del sistema CaloFit.
Muestra métricas de uso, distribución nutricional y rendimiento.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Dict, Any

from sqlalchemy import text
from sqlalchemy.orm import Session
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console(force_terminal=True)


def obtener_estadisticas(db: Session) -> Dict[str, Any]:
    """Obtiene estadísticas completas del sistema."""
    stats = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sistema": {},
        "nutricion": {},
        "ejercicio": {},
        "uso_reciente": {},
    }

    # ── Sistema ───────────────────────────────────────────────────────────────
    queries_sistema = {
        "total_clientes": "SELECT COUNT(*) FROM clients",
        "total_alimentos": "SELECT COUNT(*) FROM alimentos",
        "total_platos": "SELECT COUNT(*) FROM platos",
        "total_ejercicios": "SELECT COUNT(*) FROM ejercicios",
        "total_rutinas": "SELECT COUNT(*) FROM rutinas",
        "cache_hits": "SELECT COALESCE(SUM(hit_count), 0) FROM app_cache_alimentos",
        "cache_activo": "SELECT COUNT(*) FROM app_cache_alimentos WHERE expires_at > NOW()",
    }

    for k, q in queries_sistema.items():
        try:
            stats["sistema"][k] = db.execute(text(q)).scalar() or 0
        except Exception:
            stats["sistema"][k] = "N/A"

    # ── Nutrición ─────────────────────────────────────────────────────────────
    try:
        row = db.execute(text("""
            SELECT
                CAST(AVG(calorias_consumidas) AS FLOAT) as avg_kcal,
                CAST(AVG(proteinas_consumidas) AS FLOAT) as avg_prot,
                CAST(AVG(carbohidratos_consumidos) AS FLOAT) as avg_carb,
                CAST(AVG(grasas_consumidas) AS FLOAT) as avg_grasas,
                COUNT(*) as total_registros
            FROM progreso_calorias
            WHERE fecha >= CURRENT_DATE - INTERVAL '7 days'
        """)).fetchone()

        stats["nutricion"] = {
            "avg_kcal_7d": round(float(row[0] or 0), 1),
            "avg_proteinas_7d": round(float(row[1] or 0), 1),
            "avg_carbohidratos_7d": round(float(row[2] or 0), 1),
            "avg_grasas_7d": round(float(row[3] or 0), 1),
            "total_registros_7d": int(row[4] or 0),
        }
    except Exception as e:
        stats["nutricion"]["error"] = str(e)

    # ── Top alimentos más usados ──────────────────────────────────────────────
    try:
        top_alimentos = db.execute(text("""
            SELECT a.nombre, COUNT(*) as usos
            FROM plato_ingredientes pi
            JOIN alimentos a ON pi.alimento_id = a.id
            GROUP BY a.nombre
            ORDER BY usos DESC
            LIMIT 5
        """)).fetchall()

        stats["nutricion"]["top_alimentos"] = [
            {"nombre": r[0], "usos": r[1]} for r in top_alimentos
        ]
    except Exception:
        stats["nutricion"]["top_alimentos"] = []

    # ── Ejercicio ─────────────────────────────────────────────────────────────
    try:
        ej_stats = db.execute(text("""
            SELECT
                COUNT(*) as total_sesiones,
                COALESCE(AVG(calorias_quemadas), 0) as avg_kcal_quemadas,
                COALESCE(AVG(duracion_min), 0) as avg_duracion
            FROM workout_sessions
            WHERE fecha >= CURRENT_DATE - INTERVAL '7 days'
        """)).fetchone()

        stats["ejercicio"] = {
            "sesiones_7d": ej_stats[0] or 0,
            "avg_kcal_quemadas_7d": round(ej_stats[1] or 0, 1),
            "avg_duracion_min_7d": round(ej_stats[2] or 0, 1),
        }
    except Exception as e:
        stats["ejercicio"]["error"] = str(e)

    # ── Uso reciente (últimas 24h) ────────────────────────────────────────────
    try:
        uso_hoy = db.execute(text("""
            SELECT COUNT(*) FROM progreso_calorias
            WHERE fecha = CURRENT_DATE
        """)).scalar() or 0
        stats["uso_reciente"]["registros_hoy"] = uso_hoy
    except Exception:
        stats["uso_reciente"]["registros_hoy"] = "N/A"

    return stats


def imprimir_estadisticas(stats: Dict[str, Any]) -> None:
    """Imprime las estadísticas en formato Rich."""
    console.print()
    console.rule("[bold magenta]CaloFit — Estadísticas del Sistema[/bold magenta]")
    console.print(f"[dim]{stats['timestamp']}[/dim]")
    console.print()

    # Panel de sistema
    sistema = stats.get("sistema", {})
    s_table = Table(box=box.SIMPLE, show_header=False)
    s_table.add_column("Métrica", style="cyan")
    s_table.add_column("Valor", style="bold white", justify="right")
    s_table.add_row("Clientes", str(sistema.get("total_clientes", "N/A")))
    s_table.add_row("Alimentos en BD", str(sistema.get("total_alimentos", "N/A")))
    s_table.add_row("Platos registrados", str(sistema.get("total_platos", "N/A")))
    s_table.add_row("Ejercicios", str(sistema.get("total_ejercicios", "N/A")))
    s_table.add_row("Rutinas", str(sistema.get("total_rutinas", "N/A")))
    s_table.add_row("Cache hits totales", str(sistema.get("cache_hits", "N/A")))
    s_table.add_row("Entradas caché activas", str(sistema.get("cache_activo", "N/A")))
    console.print(Panel(s_table, title="🗃️  Sistema", border_style="cyan"))

    # Panel de nutrición (últimos 7 días)
    nutri = stats.get("nutricion", {})
    if "avg_kcal_7d" in nutri:
        n_table = Table(box=box.SIMPLE, show_header=False)
        n_table.add_column("Métrica", style="green")
        n_table.add_column("Valor", style="bold white", justify="right")
        n_table.add_row("Kcal promedio/día", f"{nutri['avg_kcal_7d']} kcal")
        n_table.add_row("Proteínas promedio", f"{nutri['avg_proteinas_7d']} g")
        n_table.add_row("Carbohidratos promedio", f"{nutri['avg_carbohidratos_7d']} g")
        n_table.add_row("Grasas promedio", f"{nutri['avg_grasas_7d']} g")
        n_table.add_row("Registros (7d)", str(nutri.get("total_registros_7d", 0)))

        top = nutri.get("top_alimentos", [])
        if top:
            n_table.add_section()
            n_table.add_row("[bold]Top 5 alimentos más usados[/bold]", "")
            for a in top:
                n_table.add_row(f"  {a['nombre']}", str(a['usos']))

        console.print(Panel(n_table, title="🥗 Nutrición (últimos 7 días)", border_style="green"))

    # Panel de ejercicio
    ej = stats.get("ejercicio", {})
    if "sesiones_7d" in ej:
        e_table = Table(box=box.SIMPLE, show_header=False)
        e_table.add_column("Métrica", style="yellow")
        e_table.add_column("Valor", style="bold white", justify="right")
        e_table.add_row("Sesiones (7d)", str(ej["sesiones_7d"]))
        e_table.add_row("Kcal quemadas promedio", f"{ej['avg_kcal_quemadas_7d']} kcal")
        e_table.add_row("Duración promedio", f"{ej['avg_duracion_min_7d']} min")
        console.print(Panel(e_table, title="🏋️  Ejercicio (últimos 7 días)", border_style="yellow"))

    uso = stats.get("uso_reciente", {})
    console.print(
        f"\n[bold]Actividad hoy:[/bold] {uso.get('registros_hoy', 'N/A')} registros\n"
    )
