"""
Script de health check completo del sistema CaloFit.
Verifica DB, modelos ML, servicios IA y APIs externas.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Any

from rich.console import Console
from rich.table import Table
from rich import box

console = Console(force_terminal=True)


def verificar_sistema(db) -> Dict[str, Any]:
    """
    Verifica el estado de todos los componentes del sistema.

    Returns:
        Dict con estado de cada componente: OK | ERROR | WARN
    """
    reporte = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "componentes": {},
        "status_global": "OK",
    }

    # ── 1. Base de datos ──────────────────────────────────────────────────────
    try:
        from sqlalchemy import text
        result = db.execute(text("SELECT 1")).scalar()
        reporte["componentes"]["base_de_datos"] = {
            "status": "OK",
            "mensaje": "Conexión activa",
        }
    except Exception as e:
        reporte["componentes"]["base_de_datos"] = {
            "status": "ERROR",
            "mensaje": str(e),
        }
        reporte["status_global"] = "ERROR"

    # ── 2. Modelos ML ─────────────────────────────────────────────────────────
    try:
        from app.services.ai.ml_service import MLServiceWrapper
        ml = MLServiceWrapper()
        test_features = {
            "Weight (kg)": 70, "Height (m)": 1.75, "Age": 30,
            "Gender": 1, "Session_Duration (hours)": 1.0,
            "Workout_Frequency (days/week)": 3,
            "Workout_Type": "Cardio", "avg_kcal_7d": 2000,
        }
        perfil, confianza = ml.predecir_perfil(test_features)
        reporte["componentes"]["modelo_ml_perfil"] = {
            "status": "OK",
            "mensaje": f"Predicción activa (confianza {confianza:.0%})",
            "perfil_test": perfil,
        }
    except Exception as e:
        reporte["componentes"]["modelo_ml_perfil"] = {
            "status": "WARN",
            "mensaje": f"ML no disponible: {e}",
        }
        if reporte["status_global"] == "OK":
            reporte["status_global"] = "WARN"

    # ── 3. Servicio LLM (Groq) ────────────────────────────────────────────────
    try:
        from app.services.ai.llm_service import LLMService
        llm = LLMService()
        reporte["componentes"]["llm_groq"] = {
            "status": "OK",
            "mensaje": "Groq inicializado",
        }
    except Exception as e:
        reporte["componentes"]["llm_groq"] = {
            "status": "WARN",
            "mensaje": f"LLM offline: {e}",
        }

    # ── 4. Validadores ────────────────────────────────────────────────────────
    try:
        from app.services.validators import SemanticValidator, NutritionalValidator
        sv = SemanticValidator(db)
        nv = NutritionalValidator()
        # Test rápido
        r = nv.validar({
            "nombre_plato": "test", "peso_total_gramos": 100,
            "calorias_total": 100, "proteina_total": 5,
            "carbohidratos_total": 10, "grasas_total": 3,
        })
        reporte["componentes"]["validadores"] = {
            "status": "OK",
            "mensaje": "SemanticValidator + NutritionalValidator OK",
        }
    except Exception as e:
        reporte["componentes"]["validadores"] = {
            "status": "ERROR",
            "mensaje": str(e),
        }
        reporte["status_global"] = "ERROR"

    # ── 5. NutricionService (alimentos en memoria) ────────────────────────────
    try:
        from app.services.nutricion_service import NutricionService
        ns = NutricionService()
        total = len(ns.alimentos_df) if hasattr(ns, 'alimentos_df') else "N/A"
        reporte["componentes"]["nutricion_service"] = {
            "status": "OK",
            "mensaje": f"{total} alimentos cargados en memoria",
        }
    except Exception as e:
        reporte["componentes"]["nutricion_service"] = {
            "status": "WARN",
            "mensaje": f"NutricionService: {e}",
        }

    # ── 6. EjerciciosService ──────────────────────────────────────────────────
    try:
        from app.services.ejercicios_service import EjerciciosService
        es = EjerciciosService()
        reporte["componentes"]["ejercicios_service"] = {
            "status": "OK",
            "mensaje": "EjerciciosService OK",
        }
    except Exception as e:
        reporte["componentes"]["ejercicios_service"] = {
            "status": "WARN",
            "mensaje": str(e),
        }

    # ── 7. API Routes (estructura) ────────────────────────────────────────────
    try:
        from app.api.v1 import router
        rutas = [r.path for r in router.routes]
        reporte["componentes"]["api_v1_routes"] = {
            "status": "OK",
            "mensaje": f"{len(rutas)} rutas registradas en /api/v1/",
        }
    except Exception as e:
        reporte["componentes"]["api_v1_routes"] = {
            "status": "WARN",
            "mensaje": str(e),
        }

    return reporte


def imprimir_health_check(reporte: Dict[str, Any]) -> None:
    """Imprime el health check en formato Rich."""
    console.print()
    console.rule("[bold cyan]CaloFit — Health Check[/bold cyan]")
    console.print(f"[dim]{reporte['timestamp']}[/dim]")
    console.print()

    tabla = Table(box=box.ROUNDED, header_style="bold white")
    tabla.add_column("Componente", style="white", min_width=25)
    tabla.add_column("Estado", justify="center", min_width=8)
    tabla.add_column("Detalles")

    status_icons = {"OK": "✅", "WARN": "⚠️ ", "ERROR": "❌"}
    status_colors = {"OK": "green", "WARN": "yellow", "ERROR": "red"}

    for nombre, info in reporte["componentes"].items():
        st = info["status"]
        icon = status_icons.get(st, "?")
        color = status_colors.get(st, "white")
        tabla.add_row(
            nombre.replace("_", " ").title(),
            f"[{color}]{icon} {st}[/{color}]",
            info["mensaje"],
        )

    console.print(tabla)
    console.print()

    # Resultado global
    global_st = reporte["status_global"]
    global_color = status_colors.get(global_st, "white")
    global_icon = status_icons.get(global_st, "?")

    console.print(
        f"Estado global: [{global_color}]{global_icon} {global_st}[/{global_color}]"
    )
    console.print()
