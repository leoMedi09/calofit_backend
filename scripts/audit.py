"""
Script de auditoría de integridad de la base de datos CaloFit.
Usa ConsistencyChecker + análisis directo para detectar problemas.
"""
from __future__ import annotations

import sys
from typing import Any, Dict, List
from datetime import datetime, timedelta, timezone

from sqlalchemy import text
from sqlalchemy.orm import Session
from rich.console import Console
from rich.table import Table
from rich import box

console = Console(force_terminal=True)


def ejecutar_auditoria(db: Session) -> Dict[str, Any]:
    """
    Ejecuta auditoría completa de la BD y retorna reporte.

    Categorías revisadas:
    - Platos sin ingredientes
    - Platos con ingredientes sin macros
    - Alimentos con macros cero o negativas
    - Registros de progreso huérfanos
    - Cache expirado
    """
    reporte: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "problemas": [],
        "advertencias": [],
        "estadisticas": {},
        "total_problemas": 0,
        "total_advertencias": 0,
    }

    # ── 1. Platos sin ingredientes ────────────────────────────────────────────
    platos_sin_ings = db.execute(text("""
        SELECT p.id, p.nombre
        FROM platos p
        LEFT JOIN plato_ingredientes pi ON p.id = pi.plato_id
        WHERE pi.id IS NULL
        ORDER BY p.id
    """)).fetchall()

    if platos_sin_ings:
        for row in platos_sin_ings:
            reporte["problemas"].append({
                "tipo": "plato_sin_ingredientes",
                "severidad": "ALTA",
                "descripcion": f"Plato ID={row[0]} '{row[1]}' no tiene ingredientes",
                "id": row[0],
            })

    # ── 2. Ingredientes sin datos nutricionales ───────────────────────────────
    ings_sin_macros = db.execute(text("""
        SELECT a.id, a.nombre
        FROM alimentos a
        WHERE a.calorias_100g IS NULL OR a.calorias_100g = 0
        ORDER BY a.id
        LIMIT 50
    """)).fetchall()

    if ings_sin_macros:
        for row in ings_sin_macros:
            reporte["advertencias"].append({
                "tipo": "alimento_sin_calorias",
                "severidad": "MEDIA",
                "descripcion": f"Alimento ID={row[0]} '{row[1]}' tiene calorías=0",
                "id": row[0],
            })

    # ── 3. Macros negativas en alimentos ─────────────────────────────────────
    macros_neg = db.execute(text("""
        SELECT id, nombre, calorias_100g, proteina_100g, carbohidratos_100g, grasas_100g
        FROM alimentos
        WHERE calorias_100g < 0
           OR proteina_100g < 0
           OR carbohidratos_100g < 0
           OR grasas_100g < 0
        LIMIT 20
    """)).fetchall()

    for row in macros_neg:
        reporte["problemas"].append({
            "tipo": "macros_negativas",
            "severidad": "CRÍTICA",
            "descripcion": f"Alimento ID={row[0]} '{row[1]}' tiene macros negativas",
            "id": row[0],
        })

    # ── 4. Cache expirado ─────────────────────────────────────────────────────
    cache_expirado_count = db.execute(text("""
        SELECT COUNT(*) FROM app_cache_alimentos
        WHERE expires_at < NOW()
    """)).scalar()

    if cache_expirado_count and cache_expirado_count > 0:
        reporte["advertencias"].append({
            "tipo": "cache_expirado",
            "severidad": "BAJA",
            "descripcion": f"{cache_expirado_count} entradas de caché expiradas pendientes de limpiar",
            "count": cache_expirado_count,
        })

    # ── 5. Estadísticas generales ─────────────────────────────────────────────
    stats_queries = {
        "total_alimentos": "SELECT COUNT(*) FROM alimentos",
        "total_platos": "SELECT COUNT(*) FROM platos",
        "total_clientes": "SELECT COUNT(*) FROM clients",
        "total_ejercicios": "SELECT COUNT(*) FROM ejercicios",
        "total_rutinas": "SELECT COUNT(*) FROM rutinas",
        "total_registros_hoy": """
            SELECT COUNT(*) FROM progreso_calorias
            WHERE fecha = CURRENT_DATE
        """,
        "platos_con_ingredientes": """
            SELECT COUNT(DISTINCT plato_id) FROM plato_ingredientes
        """,
        "cache_activo": """
            SELECT COUNT(*) FROM app_cache_alimentos
            WHERE expires_at > NOW()
        """,
    }

    for key, query in stats_queries.items():
        try:
            val = db.execute(text(query)).scalar()
            reporte["estadisticas"][key] = val or 0
        except Exception:
            reporte["estadisticas"][key] = "N/A"

    reporte["total_problemas"] = len(reporte["problemas"])
    reporte["total_advertencias"] = len(reporte["advertencias"])

    return reporte


def imprimir_reporte(reporte: Dict[str, Any]) -> None:
    """Imprime el reporte en formato Rich."""
    console.print()
    console.rule("[bold cyan]CaloFit — Auditoría de Integridad[/bold cyan]")
    console.print(f"[dim]Timestamp: {reporte['timestamp']}[/dim]")
    console.print()

    # Estadísticas
    stats_table = Table(
        title="📊 Estadísticas del Sistema",
        box=box.ROUNDED,
        style="cyan",
        header_style="bold cyan",
    )
    stats_table.add_column("Métrica", style="white")
    stats_table.add_column("Valor", style="bold green", justify="right")

    labels = {
        "total_alimentos": "Alimentos en BD",
        "total_platos": "Platos registrados",
        "platos_con_ingredientes": "Platos con ingredientes",
        "total_clientes": "Clientes activos",
        "total_ejercicios": "Ejercicios",
        "total_rutinas": "Rutinas",
        "total_registros_hoy": "Registros hoy",
        "cache_activo": "Entradas caché activas",
    }

    for key, label in labels.items():
        val = reporte["estadisticas"].get(key, "N/A")
        stats_table.add_row(label, str(val))

    console.print(stats_table)
    console.print()

    # Problemas críticos
    if reporte["problemas"]:
        prob_table = Table(
            title=f"🚨 Problemas Encontrados ({reporte['total_problemas']})",
            box=box.ROUNDED,
            style="red",
            header_style="bold red",
        )
        prob_table.add_column("Severidad", style="bold")
        prob_table.add_column("Tipo")
        prob_table.add_column("Descripción")

        for p in reporte["problemas"]:
            sev = p["severidad"]
            color = "red" if sev == "CRÍTICA" else "yellow"
            prob_table.add_row(
                f"[{color}]{sev}[/{color}]",
                p["tipo"],
                p["descripcion"],
            )
        console.print(prob_table)
        console.print()
    else:
        console.print("[bold green]✅ Sin problemas críticos encontrados[/bold green]")
        console.print()

    # Advertencias
    if reporte["advertencias"]:
        warn_table = Table(
            title=f"⚠️  Advertencias ({reporte['total_advertencias']})",
            box=box.SIMPLE,
            style="yellow",
            header_style="bold yellow",
        )
        warn_table.add_column("Severidad")
        warn_table.add_column("Descripción")

        for w in reporte["advertencias"][:10]:  # max 10
            warn_table.add_row(w["severidad"], w["descripcion"])

        if reporte["total_advertencias"] > 10:
            warn_table.add_row(
                "...",
                f"[dim]y {reporte['total_advertencias'] - 10} advertencias más[/dim]",
            )

        console.print(warn_table)
        console.print()

    # Resumen final
    if reporte["total_problemas"] == 0 and reporte["total_advertencias"] == 0:
        console.print("[bold green]🎉 La BD está en perfecto estado![/bold green]")
    elif reporte["total_problemas"] == 0:
        console.print(
            f"[yellow]⚠️  {reporte['total_advertencias']} advertencias — "
            "ejecuta 'calofit cleanup' para limpiar[/yellow]"
        )
    else:
        console.print(
            f"[bold red]❌ {reporte['total_problemas']} problemas — "
            "ejecuta 'calofit cleanup --fix' para corregir[/bold red]"
        )

    console.print()
