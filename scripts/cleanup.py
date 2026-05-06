"""
Script de limpieza automática de la BD CaloFit.
Elimina datos huérfanos, caché expirado y corrige inconsistencias.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Any

from sqlalchemy import text
from sqlalchemy.orm import Session
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console(force_terminal=True)


def ejecutar_limpieza(db: Session, dry_run: bool = True) -> Dict[str, Any]:
    """
    Ejecuta limpieza de la BD.

    Args:
        db: Sesión de SQLAlchemy
        dry_run: Si True, solo reporta sin ejecutar cambios reales

    Returns:
        Dict con resumen de operaciones realizadas
    """
    modo = "[yellow]DRY RUN[/yellow]" if dry_run else "[green]REAL[/green]"
    console.print(f"\n[bold]Modo:[/bold] {modo}")
    console.print()

    resumen: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "operaciones": [],
        "total_eliminados": 0,
        "total_corregidos": 0,
    }

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:

        # ── 1. Limpiar caché expirado ─────────────────────────────────────────
        task = progress.add_task("Limpiando caché expirado...", total=None)
        count_cache = db.execute(text("""
            SELECT COUNT(*) FROM app_cache_alimentos
            WHERE expires_at < NOW()
        """)).scalar() or 0

        if count_cache > 0:
            if not dry_run:
                db.execute(text("""
                    DELETE FROM app_cache_alimentos
                    WHERE expires_at < NOW()
                """))
                db.commit()
            resumen["operaciones"].append({
                "operacion": "limpiar_cache_expirado",
                "descripcion": f"Eliminadas {count_cache} entradas de caché expiradas",
                "cantidad": count_cache,
                "ejecutado": not dry_run,
            })
            resumen["total_eliminados"] += count_cache

        progress.update(task, description=f"✅ Caché: {count_cache} entradas expiradas")

        # ── 2. Limpiar alimentos sin resolver sin hit_count ───────────────────
        task2 = progress.add_task("Revisando alimentos sin resolver...", total=None)
        try:
            count_unresolved = db.execute(text("""
                SELECT COUNT(*) FROM alimento_sin_resolver
                WHERE created_at < NOW() - INTERVAL '30 days'
            """)).scalar() or 0

            if count_unresolved > 0:
                if not dry_run:
                    db.execute(text("""
                        DELETE FROM alimento_sin_resolver
                        WHERE created_at < NOW() - INTERVAL '30 days'
                    """))
                    db.commit()
                resumen["operaciones"].append({
                    "operacion": "limpiar_sin_resolver_antiguos",
                    "descripcion": f"Eliminados {count_unresolved} alimentos sin resolver con >30 días",
                    "cantidad": count_unresolved,
                    "ejecutado": not dry_run,
                })
                resumen["total_eliminados"] += count_unresolved
            progress.update(task2, description=f"✅ Sin resolver: {count_unresolved} entradas antiguas")
        except Exception:
            db.rollback()
            progress.update(task2, description="⚠️  Sin resolver: tabla no disponible")

        # ── 3. Platos sin ingredientes (marcar o eliminar) ────────────────────
        task3 = progress.add_task("Revisando platos huérfanos...", total=None)
        platos_huerfanos = db.execute(text("""
            SELECT p.id, p.nombre
            FROM platos p
            LEFT JOIN plato_ingredientes pi ON p.id = pi.plato_id
            WHERE pi.id IS NULL
              AND p.origen = 'manual'
        """)).fetchall()

        count_huerfanos = len(platos_huerfanos)
        if count_huerfanos > 0:
            resumen["operaciones"].append({
                "operacion": "detectar_platos_huerfanos",
                "descripcion": f"{count_huerfanos} platos sin ingredientes detectados",
                "cantidad": count_huerfanos,
                "detalle": [{"id": r[0], "nombre": r[1]} for r in platos_huerfanos[:5]],
                "ejecutado": False,  # Solo reportamos, no eliminamos automáticamente
            })

        progress.update(task3, description=f"✅ Huérfanos: {count_huerfanos} platos detectados")

        # ── 4. Normalizar nombres de alimentos (lowercase) ────────────────────
        task4 = progress.add_task("Normalizando nombres...", total=None)
        count_no_norm = db.execute(text("""
            SELECT COUNT(*) FROM alimentos
            WHERE nombre_normalizado IS NULL
               OR nombre_normalizado = ''
        """)).scalar() or 0

        if count_no_norm > 0:
            if not dry_run:
                db.execute(text("""
                    UPDATE alimentos
                    SET nombre_normalizado = LOWER(TRIM(nombre))
                    WHERE nombre_normalizado IS NULL OR nombre_normalizado = ''
                """))
                db.commit()
            resumen["operaciones"].append({
                "operacion": "normalizar_nombres",
                "descripcion": f"Normalizados {count_no_norm} nombres de alimentos",
                "cantidad": count_no_norm,
                "ejecutado": not dry_run,
            })
            resumen["total_corregidos"] += count_no_norm

        progress.update(task4, description=f"✅ Nombres: {count_no_norm} normalizados")

        # ── 5. Limpiar historial de auditoría antiguo (>90 días) ──────────────
        task5 = progress.add_task("Limpiando auditoría antigua...", total=None)
        try:
            count_auditoria = db.execute(text("""
                SELECT COUNT(*) FROM auditoria_admin
                WHERE created_at < NOW() - INTERVAL '90 days'
            """)).scalar() or 0

            if count_auditoria > 0 and not dry_run:
                db.execute(text("""
                    DELETE FROM auditoria_admin
                    WHERE created_at < NOW() - INTERVAL '90 days'
                """))
                db.commit()

            resumen["operaciones"].append({
                "operacion": "limpiar_auditoria_antigua",
                "descripcion": f"{count_auditoria} registros de auditoría con >90 días",
                "cantidad": count_auditoria,
                "ejecutado": not dry_run,
            })
            resumen["total_eliminados"] += count_auditoria if not dry_run else 0

        except Exception:
            pass  # Tabla puede no existir

        progress.update(task5, description="✅ Auditoría antigua revisada")

    return resumen


def imprimir_resumen_limpieza(resumen: Dict[str, Any]) -> None:
    """Imprime el resumen de limpieza en formato Rich."""
    from rich.table import Table
    from rich import box

    console.print()
    console.rule("[bold green]Resumen de Limpieza[/bold green]")

    modo_str = "🔍 DRY RUN (sin cambios)" if resumen["dry_run"] else "✅ REAL (cambios aplicados)"
    console.print(f"Modo: [bold]{modo_str}[/bold]")
    console.print()

    tabla = Table(box=box.ROUNDED, header_style="bold cyan")
    tabla.add_column("Operación")
    tabla.add_column("Descripción")
    tabla.add_column("Ejecutado", justify="center")

    for op in resumen["operaciones"]:
        ejecutado_icon = "✅" if op.get("ejecutado") else "📋"
        tabla.add_row(
            op["operacion"],
            op["descripcion"],
            ejecutado_icon,
        )

    console.print(tabla)
    console.print()
    console.print(f"[bold]Total eliminados:[/bold] {resumen['total_eliminados']}")
    console.print(f"[bold]Total corregidos:[/bold] {resumen['total_corregidos']}")

    if resumen["dry_run"]:
        console.print(
            "\n[yellow]Ejecuta con --no-dry-run para aplicar los cambios reales.[/yellow]"
        )
    console.print()
