from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.api.routes.auth import get_current_user
from app.models.client import Client
from app.models.historial import ProgresoCalorias
from app.models.nutricion import PlanNutricional, PlanDiario
from datetime import datetime, date
from typing import List, Dict, Any, Optional
from fastapi import Query

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
    calorias_quemadas = progreso_hoy.calorias_quemadas if progreso_hoy else 0
    calorias_restantes = objetivo_diario - calorias_consumidas + calorias_quemadas
    
    # Obtener preferencias de alimentos registrados hoy (como proxy de registros)
    from app.models.preferencias import PreferenciaAlimento, PreferenciaEjercicio
    from sqlalchemy import func
    
    # Importante: en Postgres, timestamptz se normaliza a UTC; si filtramos por date()
    # sin convertir a Perú, los registros cerca de medianoche se "mueven" de día.
    # Ajustamos el filtro a fecha Perú para Postgres.
    dialect = getattr(getattr(db, "bind", None), "dialect", None)
    dialect_name = getattr(dialect, "name", "") or ""

    if dialect_name == "postgresql":
        # `ultima_vez` es TIMESTAMP (sin tz). Asumimos que DB lo guarda en UTC (func.now()) y
        # lo convertimos a hora Perú antes de extraer `date`.
        def _date_peru(ts_col):
            # ts (sin tz) -> asumir UTC -> convertir a America/Lima -> date
            return func.date(func.timezone("America/Lima", func.timezone("UTC", ts_col)))

        alimentos_hoy = db.query(PreferenciaAlimento).filter(
            PreferenciaAlimento.client_id == cliente.id,
            _date_peru(PreferenciaAlimento.ultima_vez) == hoy,
        ).all()
        ejercicios_hoy = db.query(PreferenciaEjercicio).filter(
            PreferenciaEjercicio.client_id == cliente.id,
            _date_peru(PreferenciaEjercicio.ultima_vez) == hoy,
        ).all()
    else:
        # SQLite / otros: date() directo es suficiente.
        alimentos_hoy = db.query(PreferenciaAlimento).filter(
            PreferenciaAlimento.client_id == cliente.id,
            func.date(PreferenciaAlimento.ultima_vez) == hoy,
        ).all()
        ejercicios_hoy = db.query(PreferenciaEjercicio).filter(
            PreferenciaEjercicio.client_id == cliente.id,
            func.date(PreferenciaEjercicio.ultima_vez) == hoy,
        ).all()
    
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
                "id": alimento.id,
                "nombre": alimento.alimento.capitalize(),
                "frecuencia_total": alimento.frecuencia,
                "puntuacion": round(alimento.puntuacion, 2),
                "es_favorito": bool(alimento.es_favorito),
                "hora_registro": alimento.ultima_vez.strftime("%H:%M:%S"),
                "macros": {
                    "calorias": alimento.calorias or 0,
                    "proteinas": alimento.proteinas or 0,
                    "carbohidratos": alimento.carbohidratos or 0,
                    "grasas": alimento.grasas or 0
                }
            }
            for alimento in alimentos_hoy
        ],
        "ejercicios_registrados": [
            {
                "id": ejercicio.id,
                "nombre": ejercicio.ejercicio.capitalize(),
                "frecuencia_total": ejercicio.frecuencia,
                "hora_registro": ejercicio.ultima_vez.strftime("%H:%M:%S"),
                "calorias_quemadas": ejercicio.calorias_quemadas or 0
            }
            for ejercicio in ejercicios_hoy
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
    
    from app.models.preferencias import PreferenciaAlimento, PreferenciaEjercicio
    
    if tipo == "alimento":
        registro = db.query(PreferenciaAlimento).filter(
            PreferenciaAlimento.id == registro_id,
            PreferenciaAlimento.client_id == cliente.id
        ).first()
    elif tipo == "ejercicio":
        registro = db.query(PreferenciaEjercicio).filter(
            PreferenciaEjercicio.id == registro_id,
            PreferenciaEjercicio.client_id == cliente.id
        ).first()
    else:
        raise HTTPException(status_code=400, detail="Tipo debe ser 'alimento' o 'ejercicio'")
    
    if not registro:
        raise HTTPException(status_code=404, detail="Registro no encontrado")
    
    # Guardar nombre para el mensaje
    nombre_registro = registro.alimento if tipo == "alimento" else registro.ejercicio
    
    # Eliminar registro
    db.delete(registro)
    db.commit()
    
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
