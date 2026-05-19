from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime, timedelta, date
from typing import List, Dict
import math

from app.core.database import get_db
from app.models.client import Client
from app.models.nutricion import PlanDiario, PlanNutricional
from app.api.routes.auth import get_current_user
from app.models.historial import ProgresoCalorias, HistorialPeso, HistorialIMC
from app.core.utils import get_peru_date, get_peru_now, calcular_metabolismo_basal, obtener_macros_desglosados

router = APIRouter()

@router.get("/clientes/{cliente_id}/resumen-diario")
async def get_daily_summary(
    cliente_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    cliente = db.query(Client).filter(Client.id == cliente_id).first()
    if not cliente:
        raise HTTPException(status_code=404, detail="Cliente no encontrado")
    
    # 1. Obtener datos de consumo real del historial
    hoy = get_peru_date()
    progreso_hoy = db.query(ProgresoCalorias).filter(
        ProgresoCalorias.client_id == cliente_id,
        ProgresoCalorias.fecha == hoy
    ).first()

    consumo_actual = {
        "calorias": progreso_hoy.calorias_consumidas if progreso_hoy else 0,
        "proteinas": progreso_hoy.proteinas_consumidas if progreso_hoy else 0.0,
        "carbohidratos": progreso_hoy.carbohidratos_consumidos if progreso_hoy else 0.0,
        "grasas": progreso_hoy.grasas_consumidas if progreso_hoy else 0.0
    }
    
    # 2. Obtener el plan nutricional activo
    plan_maestro = db.query(PlanNutricional).filter(
        PlanNutricional.client_id == cliente_id
    ).order_by(PlanNutricional.fecha_creacion.desc()).first()
    
    plan_objetivo = None
    if plan_maestro:
        dia_semana = get_peru_now().isoweekday()
        plan_hoy = db.query(PlanDiario).filter(
            PlanDiario.plan_id == plan_maestro.id,
            PlanDiario.dia_numero == dia_semana
        ).first() or db.query(PlanDiario).filter(PlanDiario.plan_id == plan_maestro.id).first()
        
        if plan_hoy:
            total_macros_kcal = (plan_hoy.proteinas_g * 4) + (plan_hoy.carbohidratos_g * 4) + (plan_hoy.grasas_g * 9)
            pct_p = round((plan_hoy.proteinas_g * 4 / total_macros_kcal) * 100) if total_macros_kcal > 0 else 0
            pct_c = round((plan_hoy.carbohidratos_g * 4 / total_macros_kcal) * 100) if total_macros_kcal > 0 else 0
            pct_g = round((plan_hoy.grasas_g * 9 / total_macros_kcal) * 100) if total_macros_kcal > 0 else 0
            
            estado_api = plan_maestro.status
            estado_frontend = "validado" if estado_api == "validado" else "provisional_ia"
            
            es_condicion_critica = False
            condiciones_list = [c.lower() for c in (cliente.medical_conditions or [])]
            from app.services.ia_service import CONDICIONES_CRITICAS
            for condicion_critica in CONDICIONES_CRITICAS:
                if any(condicion_critica in c for c in condiciones_list):
                    es_condicion_critica = True
                    break
            
            mensaje_cliente = ""
            if estado_frontend == "validado":
                mensaje_cliente = "✅ Tu nutricionista ha validado tu plan. ¡Sigue así!"
            elif es_condicion_critica:
                mensaje_cliente = "⚠️ Tu plan es provisional y ultra-conservador por tu condición médica. Contacta a tu nutricionista."
                estado_frontend = "en_revision"
            else:
                mensaje_cliente = "🤖 Plan generado por IA. Tu nutricionista lo revisará pronto."

            plan_objetivo = {
                "calorias_objetivo": plan_hoy.calorias_dia,
                "proteinas_objetivo_g": plan_hoy.proteinas_g,
                "carbohidratos_objetivo_g": plan_hoy.carbohidratos_g,
                "grasas_objetivo_g": plan_hoy.grasas_g,
                "distribucion": {
                    "proteina_pct": pct_p,
                    "carbohidratos_pct": pct_c,
                    "grasas_pct": pct_g
                },
                "validado": plan_maestro.status == "validado",
                "plan_id": plan_maestro.id,
                "estado_plan": estado_frontend,
                "requiere_validacion": estado_api == "draft_ia" or es_condicion_critica,
                "es_condicion_critica": es_condicion_critica,
                "mensaje_cliente": mensaje_cliente,
                "descripcion_estado": "Plan validado" if estado_frontend == "validado" else "Pendiente de validación",
                "generado_automaticamente": plan_maestro.nutricionista_id is None,
                "ai_strategic_focus": cliente.ai_strategic_focus
            }
    else:
        # FALLBACK: Usar el cálculo centralizado de utils.py
        calorias_fallback = calcular_metabolismo_basal(cliente)
        if cliente.goal == "Perder peso":
            calorias_fallback *= 0.85
        elif cliente.goal == "Ganar masa":
            calorias_fallback *= 1.1

        peso = float(cliente.weight or 70.0)
        macros = obtener_macros_desglosados(calorias_fallback, cliente.goal, peso)
        
        es_condicion_critica = False
        condiciones_list = [c.lower() for c in (cliente.medical_conditions or [])]
        from app.services.ia_service import CONDICIONES_CRITICAS
        for condicion_critica in CONDICIONES_CRITICAS:
            if any(condicion_critica in c for c in condiciones_list):
                es_condicion_critica = True
                break

        plan_objetivo = {
            "calorias_objetivo": macros["calorias"],
            "proteinas_objetivo_g": macros["proteinas_g"],
            "carbohidratos_objetivo_g": macros["carbohidratos_g"],
            "grasas_objetivo_g": macros["grasas_g"],
            "distribucion": {
                "proteina_pct": macros["pct"]["p"],
                "carbohidratos_pct": macros["pct"]["c"],
                "grasas_pct": macros["pct"]["g"]
            },
            "validado": False,
            "plan_id": None,
            "es_fallback": True,
            "estado_plan": "en_revision" if es_condicion_critica else "provisional_ia",
            "requiere_validacion": True,
            "es_condicion_critica": es_condicion_critica,
            "mensaje_cliente": "⚠️ Por tu seguridad, este plan es ultra-conservador hasta que el nutri lo valide." if es_condicion_critica else "🤖 No tienes un plan activo aún. Hemos generado uno temporal con IA.",
            "descripcion_estado": "Cálculo IA Temporal",
            "generado_automaticamente": True,
            "ai_strategic_focus": cliente.ai_strategic_focus
        }

    # 3. Generar Insight
    meta_calorias = plan_objetivo["calorias_objetivo"]
    consumido = consumo_actual["calorias"]
    if consumido == 0:
        ai_insight = f"{cliente.first_name}, aún no has registrado calorías hoy. Tu meta es {meta_calorias:.0f} kcal."
    elif consumido < meta_calorias * 0.5:
        ai_insight = f"{cliente.first_name}, te faltan {(meta_calorias - consumido):.0f} kcal para tu meta."
    elif consumido >= meta_calorias * 0.9 and consumido <= meta_calorias * 1.1:
        ai_insight = f"¡Excelente, {cliente.first_name}! Estás en tu meta de {meta_calorias:.0f} kcal."
    elif consumido > meta_calorias * 1.1:
        ai_insight = f"{cliente.first_name}, has superado tu meta en {(consumido - meta_calorias):.0f} kcal."
    else:
        ai_insight = f"Llevas {((consumido / meta_calorias)*100):.0f}% de tu meta. Te quedan {(meta_calorias - consumido):.0f} kcal."

    return {
        "dieta_recomendada": {
            "calorias_diarias": consumo_actual["calorias"],
            "proteinas_g": consumo_actual["proteinas"],
            "carbohidratos_g": consumo_actual["carbohidratos"],
            "grasas_g": consumo_actual["grasas"],
            "gasto_metabolico_basal": round(calcular_metabolismo_basal(cliente), 1),
            "imc": round(cliente.weight / ((cliente.height/100)**2), 1) if cliente.height and cliente.weight else 0,
        },
        "calorias_quemadas": progreso_hoy.calorias_quemadas if progreso_hoy else 0,
        "resumen": {
            "calorias_consumidas": consumo_actual["calorias"],
            "calorias_quemadas": progreso_hoy.calorias_quemadas if progreso_hoy else 0,
        },
        "plan_nutricional": plan_objetivo,
        "ai_insight": ai_insight,
        "ai_strategic_focus": cliente.ai_strategic_focus,
        "nutri_weekly_note": cliente.nutri_weekly_note,
        "is_strategy_validated": cliente.is_strategic_guide_validated
    }

@router.get("/clientes/{cliente_id}/calorias-tendencia")
async def get_calories_trend(cliente_id: int, db: Session = Depends(get_db)):
    dias_semana = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
    resultado = []
    hoy = get_peru_date()
    for i in range(7):
        fecha = hoy - timedelta(days=6-i)
        p = db.query(ProgresoCalorias).filter(ProgresoCalorias.client_id == cliente_id, ProgresoCalorias.fecha == fecha).first()
        resultado.append({
            "dia": dias_semana[fecha.weekday()],
            "consumidas": p.calorias_consumidas if p else 0,
            "quemadas": p.calorias_quemadas if p else 0
        })
    return resultado

@router.get("/clientes/{cliente_id}/peso-historial")
async def get_weight_history(cliente_id: int, db: Session = Depends(get_db)):
    registros = db.query(HistorialPeso).filter(HistorialPeso.client_id == cliente_id).order_by(HistorialPeso.fecha_registro.desc()).limit(10).all()
    return [{"fecha": r.fecha_registro.isoformat(), "peso": r.peso_kg} for r in registros]

@router.get("/clientes/{cliente_id}/imc-historial")
async def get_imc_history(cliente_id: int, db: Session = Depends(get_db)):
    registros = db.query(HistorialIMC).filter(HistorialIMC.client_id == cliente_id).order_by(HistorialIMC.fecha_registro.desc()).limit(10).all()
    return [{"fecha": r.fecha_registro.isoformat(), "imc": r.imc, "categoria": r.categoria} for r in registros]