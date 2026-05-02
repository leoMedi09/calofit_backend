"""
Endpoints de ejercicios — CaloFit.

  GET  /ejercicios/                  — Catálogo con filtros
  POST /ejercicios/rutina            — Generar rutina adaptativa
  POST /ejercicios/log-series        — Registrar series/reps directamente
  GET  /ejercicios/logs/{client_id}  — Historial de workout_logs
"""
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import text as _sql
from sqlalchemy.orm import Session

from app.api.routes.auth import get_current_user
from app.core.database import get_db
from app.services.asistente_registro_ejercicio import registro_ejercicio_handler
from app.services.rutina_service import generar_rutina_inteligente

router = APIRouter()


# ── Schemas ───────────────────────────────────────────────────────────────────

class RutinaRequest(BaseModel):
    zonas_objetivo: List[str] = Field(
        ...,
        description="Grupos musculares objetivo: Pecho, Espalda, Piernas, Hombros, "
                    "Bíceps, Tríceps, Core, Glúteos, Cardio, Cuerpo Completo",
        example=["Piernas", "Glúteos"],
    )
    tiempo_min: int = Field(default=60, ge=15, le=180, description="Tiempo disponible en minutos")


class LogSeriesRequest(BaseModel):
    ejercicio:       str   = Field(..., max_length=200, description="Nombre del ejercicio")
    series:          int   = Field(..., ge=1, le=20)
    reps:            int   = Field(..., ge=1, le=100)
    peso_kg:         Optional[float] = Field(default=None, ge=0, le=500)
    met:             float = Field(default=5.0, ge=1.0, le=20.0)
    duracion_min:    float = Field(default=45.0, ge=1.0, le=240.0)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/")
async def listar_ejercicios(
    grupo: Optional[str] = Query(default=None, description="Filtrar por grupo_padre"),
    nivel: Optional[str] = Query(default=None, description="Filtrar por nivel"),
    metrica: Optional[str] = Query(default=None, description="Filtrar por tipo_metrica"),
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """Catálogo de ejercicios con filtros opcionales."""
    conditions = []
    params: dict = {"lim": limit}

    if grupo:
        conditions.append("grupo_padre = :grupo")
        params["grupo"] = grupo
    if nivel:
        conditions.append("nivel = :nivel")
        params["nivel"] = nivel
    if metrica:
        conditions.append("tipo_metrica = :metrica")
        params["metrica"] = metrica

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    rows = db.execute(_sql(f"""
        SELECT id, nombre, musculo_principal, tipo, nivel, met,
               tipo_metrica, grupo_padre, es_cardio
        FROM ejercicios
        {where}
        ORDER BY nombre
        LIMIT :lim
    """), params).fetchall()

    return {
        "total": len(rows),
        "ejercicios": [
            {
                "id": r[0], "nombre": r[1], "musculo_principal": r[2],
                "tipo": r[3], "nivel": r[4], "met": r[5],
                "tipo_metrica": r[6], "grupo_padre": r[7], "es_cardio": r[8],
            }
            for r in rows
        ],
    }


@router.get("/grupos")
async def listar_grupos(db: Session = Depends(get_db)):
    """Lista todos los grupos_padre disponibles con conteo de ejercicios."""
    rows = db.execute(_sql("""
        SELECT grupo_padre, COUNT(*) as total
        FROM ejercicios
        WHERE grupo_padre IS NOT NULL
        GROUP BY grupo_padre
        ORDER BY total DESC
    """)).fetchall()
    return {"grupos": [{"nombre": r[0], "total": r[1]} for r in rows]}


@router.post("/rutina")
async def generar_rutina(
    body: RutinaRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Genera una rutina personalizada basada en perfil ML (A/B/C),
    lesiones del usuario y zonas objetivo.
    """
    from app.models.client import Client
    perfil = db.query(Client).filter(Client.email.ilike(current_user.email)).first()
    if not perfil:
        raise HTTPException(status_code=404, detail="Perfil de cliente no encontrado")

    rutina = await generar_rutina_inteligente(
        user_id=perfil.id,
        zonas_objetivo=body.zonas_objetivo,
        tiempo_min=body.tiempo_min,
        db=db,
    )
    if "error" in rutina:
        raise HTTPException(status_code=400, detail=rutina["error"])
    return rutina


@router.post("/log-series")
async def registrar_series(
    body: LogSeriesRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Registra una sesión detallada (series × reps × peso) en workout_logs
    y sincroniza con progreso_calorias y Random Forest features.
    """
    from app.models.client import Client
    perfil = db.query(Client).filter(Client.email.ilike(current_user.email)).first()
    if not perfil:
        raise HTTPException(status_code=404, detail="Perfil de cliente no encontrado")

    peso_corporal = float(getattr(perfil, "weight", None) or 70.0)
    return registro_ejercicio_handler.registrar_workout_log(
        client_id=perfil.id,
        ejercicio=body.ejercicio,
        series=body.series,
        reps=body.reps,
        peso_kg=body.peso_kg,
        db=db,
        met=body.met,
        duracion_min=body.duracion_min,
        peso_corporal_kg=peso_corporal,
    )


@router.get("/logs")
async def historial_logs(
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Historial de workout_logs del usuario autenticado."""
    from app.models.client import Client
    perfil = db.query(Client).filter(Client.email.ilike(current_user.email)).first()
    if not perfil:
        raise HTTPException(status_code=404, detail="Perfil no encontrado")

    return {
        "logs": registro_ejercicio_handler.get_workout_logs(perfil.id, db, limit),
    }
