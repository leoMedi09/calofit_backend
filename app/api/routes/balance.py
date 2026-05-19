from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.api.routes.auth import get_current_user
from app.models.client import Client
from app.models.historial import ProgresoCalorias
from app.models.nutricion import PlanNutricional, PlanDiario
from datetime import datetime, date, timezone as _tz
from typing import Optional
from zoneinfo import ZoneInfo

_LIMA = ZoneInfo("America/Lima")

def _utc_naive_to_lima(dt: datetime) -> datetime:
    """Convierte un datetime naive almacenado como UTC a hora Lima."""
    if dt is None:
        return datetime.now(_LIMA)
    return dt.replace(tzinfo=_tz.utc).astimezone(_LIMA)

router = APIRouter()


@router.get("/hoy")
async def obtener_balance_hoy(
    fecha: Optional[str] = Query(None, description="Fecha opcional YYYY-MM-DD para historial"),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    📊 MI BALANCE DIARIO: Ver todos los registros de una fecha

    
    Devuelve:
    - Resumen de calorías (consumidas, quemadas, restantes)
    - Lista de alimentos registrados
    - Lista de ejercicios registrados
    """
    # Obtener cliente
    cliente = db.query(Client).filter(Client.email == current_user.email).first()
    if not cliente:
        raise HTTPException(status_code=404, detail="Cliente no encontrado")
    
    # Obtener plan activo
    # Obtener plan activo (Lógica alineada con Dashboard)
    plan_activo = db.query(PlanNutricional).filter(
        PlanNutricional.client_id == cliente.id
    ).order_by(PlanNutricional.fecha_creacion.desc()).first()
    
    objetivo_diario = 2000 # Default
    if plan_activo:
        # Intentar obtener meta especifica del dia
        from app.core.utils import get_peru_now
        dia_semana = get_peru_now().isoweekday()
        plan_hoy = db.query(PlanDiario).filter(
            PlanDiario.plan_id == plan_activo.id,
            PlanDiario.dia_numero == dia_semana
        ).first()
        
        if not plan_hoy:
             # Si no hay para hoy, tomar el primero disponible (Lógica Dashboard)
             plan_hoy = db.query(PlanDiario).filter(
                 PlanDiario.plan_id == plan_activo.id
             ).first()

        if plan_hoy:
             objetivo_diario = plan_hoy.calorias_dia
        else:
             objetivo_diario = plan_activo.calorias_ia_base or 2000
    else:
        # 🆕 FALLBACK IA (Lógica Dashboard): Calcular si no hay plan
        from app.services.ia_service import ia_engine
        
        genero_map = {"M": 1, "F": 2}
        genero = genero_map.get(cliente.gender, 1)
        edad = (date.today().year - cliente.birth_date.year) if cliente.birth_date else 25
        nivel_map = {"Sedentario": 1.20, "Ligero": 1.375, "Moderado": 1.55, "Activo": 1.725, "Muy activo": 1.90}
        nivel_actividad = nivel_map.get(cliente.activity_level, 1.20)
        objetivo_map = {"Perder peso": "perder", "Mantener peso": "mantener", "Ganar masa": "ganar"}
        objetivo = objetivo_map.get(cliente.goal, "mantener")
        
        objetivo_diario, proteinas_objetivo, carbohidratos_objetivo, grasas_objetivo = ia_engine.calcular_macros_completos(
            genero=genero, edad=edad, peso=cliente.weight, talla=cliente.height,
            nivel_actividad=nivel_actividad, objetivo=objetivo
        )
    
    # Obtener progreso de hoy
    
    from app.core.utils import get_peru_date
    if fecha:
        try:
            hoy = date.fromisoformat(fecha)
        except ValueError:
            hoy = get_peru_date()
    else:
        hoy = get_peru_date()
        
    progreso_hoy = db.query(ProgresoCalorias).filter(
        ProgresoCalorias.client_id == cliente.id,
        ProgresoCalorias.fecha == hoy
    ).first()
    
    calorias_consumidas = progreso_hoy.calorias_consumidas if progreso_hoy else 0

    # Calorías quemadas: fuente autoritativa = workout_logs (cubre todos los paths de registro)
    from sqlalchemy import text as _sql_wl
    _dialect = getattr(getattr(db, "bind", None), "dialect", None)
    _dname = getattr(_dialect, "name", "") or ""
    if _dname == "postgresql":
        calorias_quemadas = float(db.execute(_sql_wl(
            "SELECT COALESCE(SUM(calorias_quemadas), 0) FROM workout_logs "
            "WHERE client_id = :cid "
            "  AND (created_at AT TIME ZONE 'UTC' AT TIME ZONE 'America/Lima')::date = :hoy"
        ), {"cid": cliente.id, "hoy": hoy}).scalar() or 0)
    else:
        calorias_quemadas = float(db.execute(_sql_wl(
            "SELECT COALESCE(SUM(calorias_quemadas), 0) FROM workout_logs "
            "WHERE client_id = :cid AND date(created_at) = :hoy"
        ), {"cid": cliente.id, "hoy": hoy}).scalar() or 0)

    calorias_restantes = objetivo_diario - calorias_consumidas + calorias_quemadas
    
    # Obtener preferencias de alimentos registrados hoy (como proxy de registros)
    from app.models.preferencias import PreferenciaAlimento, PreferenciaEjercicio
    from sqlalchemy import func
    
    from sqlalchemy import text as _text
    from app.models.comida_registro import ComidaRegistro

    dialect = getattr(getattr(db, "bind", None), "dialect", None)
    dialect_name = getattr(dialect, "name", "") or ""

    # Fuente de verdad: comida_registros (macros correctos por ítem, ordenados por hora)
    if dialect_name == "postgresql":
        registros_hoy = db.query(ComidaRegistro).filter(
            ComidaRegistro.client_id == cliente.id,
            func.date(
                func.timezone("America/Lima", func.timezone("UTC", ComidaRegistro.created_at))
            ) == hoy,
        ).order_by(ComidaRegistro.created_at.desc()).all()

        ejercicios_hoy_rows = db.execute(_text(
            "SELECT id, ejercicio, series, reps, peso_kg, calorias_quemadas, session_duration_min, intensity, "
            "  created_at AT TIME ZONE 'UTC' AT TIME ZONE 'America/Lima' AS hora_lima "
            "FROM workout_logs "
            "WHERE client_id = :cid "
            "  AND (created_at AT TIME ZONE 'UTC' AT TIME ZONE 'America/Lima')::date = :hoy "
            "ORDER BY created_at DESC"
        ), {"cid": cliente.id, "hoy": hoy}).fetchall()
    else:
        registros_hoy = db.query(ComidaRegistro).filter(
            ComidaRegistro.client_id == cliente.id,
            ComidaRegistro.fecha == hoy,
        ).order_by(ComidaRegistro.created_at.desc()).all()

        ejercicios_hoy_rows = db.execute(_text(
            "SELECT id, ejercicio, series, reps, peso_kg, calorias_quemadas, session_duration_min, intensity, created_at AS hora_lima "
            "FROM workout_logs WHERE client_id = :cid AND date(created_at) = :hoy ORDER BY created_at DESC"
        ), {"cid": cliente.id, "hoy": hoy}).fetchall()

    # Preferencias para es_favorito / puntuacion
    prefs_idx: dict[str, PreferenciaAlimento] = {
        p.alimento.lower(): p
        for p in db.query(PreferenciaAlimento).filter(
            PreferenciaAlimento.client_id == cliente.id
        ).all()
    }

    # Agrupar registros por nombre (mismo alimento registrado varias veces → sumar macros, mostrar ×N)
    _grupos: dict[str, dict] = {}
    for reg in registros_hoy:  # ya ordenados DESC por created_at
        _key = reg.nombre_alimento.lower()
        if _key not in _grupos:
            _grupos[_key] = {
                "id":     reg.id,
                "nombre": reg.nombre_alimento,
                "hora":   reg.created_at,
                "kcal": 0.0, "prot": 0.0, "carb": 0.0, "gras": 0.0,
                "cantidad": 0,
            }
        _grupos[_key]["kcal"]     += float(reg.kcal or 0)
        _grupos[_key]["prot"]     += float(reg.proteina_g or 0)
        _grupos[_key]["carb"]     += float(reg.carbohidratos_g or 0)
        _grupos[_key]["gras"]     += float(reg.grasas_g or 0)
        _grupos[_key]["cantidad"] += 1

    return {
        "fecha": hoy.isoformat(),
        "resumen": {
            "calorias_consumidas": calorias_consumidas or 0,
            "calorias_quemadas": calorias_quemadas or 0,
            "calorias_restantes": calorias_restantes,
            "objetivo_diario": objetivo_diario,
            "proteinas_g": progreso_hoy.proteinas_consumidas if progreso_hoy else 0.0,
            "carbohidratos_g": progreso_hoy.carbohidratos_consumidos if progreso_hoy else 0.0,
            "grasas_g": progreso_hoy.grasas_consumidas if progreso_hoy else 0.0,
            "proteinas_objetivo": proteinas_objetivo if 'proteinas_objetivo' in locals() else (plan_hoy.proteinas_g if 'plan_hoy' in locals() and plan_hoy else 150.0),
            "carbohidratos_objetivo": carbohidratos_objetivo if 'carbohidratos_objetivo' in locals() else (plan_hoy.carbohidratos_g if 'plan_hoy' in locals() and plan_hoy else 250.0),
            "grasas_objetivo": grasas_objetivo if 'grasas_objetivo' in locals() else (plan_hoy.grasas_g if 'plan_hoy' in locals() and plan_hoy else 60.0)
        },
        "alimentos_registrados": [
            {
                "id":       g["id"],
                "nombre":   g["nombre"].capitalize(),
                "cantidad": g["cantidad"],
                "frecuencia_total": prefs_idx[_k].frecuencia if _k in prefs_idx else g["cantidad"],
                "puntuacion":       round(prefs_idx[_k].puntuacion, 2) if _k in prefs_idx else 1.0,
                "es_favorito":      bool(prefs_idx[_k].es_favorito) if _k in prefs_idx else False,
                "hora_registro":    _utc_naive_to_lima(g["hora"]).strftime("%H:%M"),
                "macros": {
                    "calorias":      round(g["kcal"], 1),
                    "proteinas":     round(g["prot"], 1),
                    "carbohidratos": round(g["carb"], 1),
                    "grasas":        round(g["gras"], 1),
                },
            }
            for _k, g in _grupos.items()
        ],
        "ejercicios_registrados": [
            {
                "id": row.id,
                "nombre": (row.ejercicio or "").capitalize(),
                "series": row.series or 0,
                "reps": row.reps or 0,
                "peso_kg": row.peso_kg,
                "calorias_quemadas": float(row.calorias_quemadas or 0),
                "duracion_min": float(row.session_duration_min or 0),
                "intensidad": row.intensity or "",
                "hora_registro": row.hora_lima.strftime("%H:%M") if row.hora_lima else "",
            }
            for row in ejercicios_hoy_rows
        ]
    }


@router.post("/favorito/{registro_id}")
async def toggle_favorito(
    registro_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user),
):
    """Marca o desmarca un alimento como favorito (toggle)."""
    from app.models.preferencias import PreferenciaAlimento

    cliente = db.query(Client).filter(Client.email == current_user.email).first()
    if not cliente:
        raise HTTPException(status_code=404, detail="Cliente no encontrado")

    registro = db.query(PreferenciaAlimento).filter(
        PreferenciaAlimento.id == registro_id,
        PreferenciaAlimento.client_id == cliente.id,
    ).first()
    if not registro:
        raise HTTPException(status_code=404, detail="Registro no encontrado")

    registro.es_favorito = 0 if registro.es_favorito else 1
    db.commit()
    return {"es_favorito": bool(registro.es_favorito), "id": registro_id}


@router.get("/favoritos")
async def listar_favoritos(
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user),
):
    """Devuelve todos los alimentos marcados como favoritos del usuario."""
    from app.models.preferencias import PreferenciaAlimento

    cliente = db.query(Client).filter(Client.email == current_user.email).first()
    if not cliente:
        raise HTTPException(status_code=404, detail="Cliente no encontrado")

    favs = db.query(PreferenciaAlimento).filter(
        PreferenciaAlimento.client_id == cliente.id,
        PreferenciaAlimento.es_favorito == 1,
    ).order_by(PreferenciaAlimento.frecuencia.desc()).all()

    return [
        {
            "id": f.id,
            "nombre": f.alimento.capitalize(),
            "frecuencia_total": f.frecuencia,
            "macros": {
                "calorias": f.calorias or 0,
                "proteinas": f.proteinas or 0,
                "carbohidratos": f.carbohidratos or 0,
                "grasas": f.grasas or 0,
            },
        }
        for f in favs
    ]


@router.delete("/registro/{registro_id}")
async def eliminar_registro(
    registro_id: int,
    tipo: str,  # "alimento" o "ejercicio"
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    🗑️ ELIMINAR REGISTRO: Elimina un alimento o ejercicio registrado
    
    Parámetros:
    - registro_id: ID del registro a eliminar
    - tipo: "alimento" o "ejercicio"
    
    Recalcula automáticamente el balance después de eliminar.
    """
    # Obtener cliente
    cliente = db.query(Client).filter(Client.email == current_user.email).first()
    if not cliente:
        raise HTTPException(status_code=404, detail="Cliente no encontrado")
    
    from app.models.comida_registro import ComidaRegistro
    from app.services.trazabilidad import recalcular_progreso_diario
    from sqlalchemy import text as _text2

    if tipo == "alimento":
        registro = db.query(ComidaRegistro).filter(
            ComidaRegistro.id == registro_id,
            ComidaRegistro.client_id == cliente.id
        ).first()
        if not registro:
            raise HTTPException(status_code=404, detail="Registro no encontrado")
        nombre_registro = registro.nombre_alimento
        fecha_alim = registro.fecha
        db.delete(registro)
        db.flush()
        recalcular_progreso_diario(cliente.id, fecha_alim, db)
        db.commit()
    elif tipo == "ejercicio":
        # Ejercicios viven en workout_logs
        row = db.execute(_text2(
            "SELECT id, ejercicio, calorias_quemadas FROM workout_logs "
            "WHERE id = :rid AND client_id = :cid"
        ), {"rid": registro_id, "cid": cliente.id}).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Registro no encontrado")
        nombre_registro = row.ejercicio
        cal_a_restar = float(row.calorias_quemadas or 0)
        db.execute(_text2("DELETE FROM workout_logs WHERE id = :rid"), {"rid": registro_id})
        # Restar calorías del progreso del día
        from app.core.utils import get_peru_date as _gpd
        db.execute(_text2(
            "UPDATE progreso_calorias SET calorias_quemadas = GREATEST(0, calorias_quemadas - :cal) "
            "WHERE client_id = :cid AND fecha = :hoy"
        ), {"cal": cal_a_restar, "cid": cliente.id, "hoy": _gpd()})
        db.commit()
    else:
        raise HTTPException(status_code=400, detail="Tipo debe ser 'alimento' o 'ejercicio'")
    
    # Recalcular balance
    
    from app.core.utils import get_peru_date
    hoy = get_peru_date()
    progreso_hoy = db.query(ProgresoCalorias).filter(
        ProgresoCalorias.client_id == cliente.id,
        ProgresoCalorias.fecha == hoy
    ).first()
    
    if progreso_hoy:
        # Obtener plan para calcular restantes
        # Obtener plan para calcular restantes (Lógica alineada)
        plan_activo = db.query(PlanNutricional).filter(
            PlanNutricional.client_id == cliente.id
        ).order_by(PlanNutricional.fecha_creacion.desc()).first()
        
        objetivo = 2000
        if plan_activo:
            from app.core.utils import get_peru_now
            dia_semana = get_peru_now().isoweekday()
            plan_hoy = db.query(PlanDiario).filter(
                PlanDiario.plan_id == plan_activo.id,
                PlanDiario.dia_numero == dia_semana
            ).first()
            if plan_hoy:
                objetivo = plan_hoy.calorias_dia
            else:
                objetivo = plan_activo.calorias_ia_base or 2000
        calorias_restantes = objetivo - (progreso_hoy.calorias_consumidas or 0) + (progreso_hoy.calorias_quemadas or 0)
        
        nuevo_balance = {
            "calorias_consumidas": progreso_hoy.calorias_consumidas or 0,
            "calorias_quemadas": progreso_hoy.calorias_quemadas or 0,
            "calorias_restantes": calorias_restantes,
            "proteinas_g": progreso_hoy.proteinas_consumidas or 0.0,
            "carbohidratos_g": progreso_hoy.carbohidratos_consumidos or 0.0,
            "grasas_g": progreso_hoy.grasas_consumidas or 0.0
        }
    else:
        nuevo_balance = {
            "calorias_consumidas": 0,
            "calorias_quemadas": 0,
            "calorias_restantes": 2000,
            "proteinas_g": 0.0,
            "carbohidratos_g": 0.0,
            "grasas_g": 0.0
        }
    
    return {
        "success": True,
        "mensaje": f"'{nombre_registro.capitalize()}' eliminado exitosamente",
        "nuevo_balance": nuevo_balance
    }
