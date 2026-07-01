from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.database import get_db

router = APIRouter()

# PostgreSQL (Neon en producción / contenedor `db` en local)
_SQL = """
    SELECT id, first_name, last_name_paternal, email, gender, goal, activity_level
    FROM clients
    WHERE (:gender IS NULL OR gender = :gender)
      AND (:goal IS NULL OR goal = :goal)
      AND (:activity_level IS NULL OR activity_level = :activity_level)
      AND (:nombre IS NULL OR first_name ILIKE :nombre_like)
    ORDER BY id
    LIMIT :limit OFFSET :offset
"""


@router.get("/")
async def nueva_funcion(
    gender: Optional[str] = Query(None),
    goal: Optional[str] = Query(None),
    activity_level: Optional[str] = Query(None),
    nombre: Optional[str] = Query(None),
    limit: int = Query(20, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    rows = db.execute(
        text(_SQL),
        {
            "gender": gender,
            "goal": goal,
            "activity_level": activity_level,
            "nombre": nombre,
            "nombre_like": f"%{nombre}%" if nombre else None,
            "limit": limit,
            "offset": offset,
        },
    ).mappings().all()

    return {
        "status": "ok",
        "filtros_aplicados": {
            "gender": gender,
            "goal": goal,
            "activity_level": activity_level,
            "nombre": nombre,
            "limit": limit,
            "offset": offset,
        },
        "total": len(rows),
        "clientes": [dict(r) for r in rows],
    }
