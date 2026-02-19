from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime, timedelta, date
from typing import List, Dict
import math
from app.services.ia_service import ia_engine

from app.core.database import get_db
from app.models.client import Client
from app.models.nutricion import PlanDiario, PlanNutricional
from app.api.routes.auth import get_current_user

router = APIRouter()

@router.get("/clientes/{cliente_id}/resumen-diario")
async def get_daily_summary(
    cliente_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    from app.models.historial import ProgresoCalorias
    
    cliente = db.query(Client).filter(Client.id == cliente_id).first()
    
    if not cliente:
        raise HTTPException(status_code=404, detail="Cliente no encontrado")
    
    # 1. Obtener datos de consumo real del historial (ProgresoCalorias)
    hoy = date.today()
    progreso_hoy = db.query(ProgresoCalorias).filter(
        ProgresoCalorias.client_id == cliente_id,
        ProgresoCalorias.fecha == hoy
    ).first()

    # Preparamos los valores de consumo (reales o 0)
    consumo_actual = {
        "calorias": progreso_hoy.calorias_consumidas if progreso_hoy else 0,
        "proteinas": progreso_hoy.proteinas_consumidas if progreso_hoy else 0.0,
        "carbohidratos": progreso_hoy.carbohidratos_consumidos if progreso_hoy else 0.0,
        "grasas": progreso_hoy.grasas_consumidas if progreso_hoy else 0.0
    }
    
    # 🆕 2. Obtener el plan nutricional activo para mostrar OBJETIVO de macros
    plan_maestro = db.query(PlanNutricional).filter(
        PlanNutricional.client_id == cliente_id
    ).order_by(PlanNutricional.fecha_creacion.desc()).first()
    
    plan_objetivo = None
    if plan_maestro:
        # Obtener el plan del día actual
        dia_semana = datetime.now().isoweekday()
        plan_hoy = db.query(PlanDiario).filter(
            PlanDiario.plan_id == plan_maestro.id,
            PlanDiario.dia_numero == dia_semana
        ).first()
        
        if not plan_hoy:
            # Si no hay para hoy, tomar el primero disponible
            plan_hoy = db.query(PlanDiario).filter(
                PlanDiario.plan_id == plan_maestro.id
            ).first()
        
        if plan_hoy:
            # Calcular porcentajes de distribución
            total_macros_kcal = (plan_hoy.proteinas_g * 4) + (plan_hoy.carbohidratos_g * 4) + (plan_hoy.grasas_g * 9)
            pct_proteina = round((plan_hoy.proteinas_g * 4 / total_macros_kcal) * 100) if total_macros_kcal > 0 else 0
            pct_carbohidratos = round((plan_hoy.carbohidratos_g * 4 / total_macros_kcal) * 100) if total_macros_kcal > 0 else 0
            pct_grasas = round((plan_hoy.grasas_g * 9 / total_macros_kcal) * 100) if total_macros_kcal > 0 else 0
            
            plan_objetivo = {
                "calorias_objetivo": plan_hoy.calorias_dia,
                "proteinas_objetivo_g": plan_hoy.proteinas_g,
                "carbohidratos_objetivo_g": plan_hoy.carbohidratos_g,
                "grasas_objetivo_g": plan_hoy.grasas_g,
                "distribucion": {
                    "proteina_pct": pct_proteina,
                    "carbohidratos_pct": pct_carbohidratos,
                    "grasas_pct": pct_grasas
                },
                "validado": plan_maestro.status == "validado",
                "plan_id": plan_maestro.id
            }
    else:
        # 🆕 FALLBACK: Usar el mismo modelo ML que generación automática
        print(f"⚠️ Usuario {cliente_id} sin plan. Calculando con IA...")
        
        from app.services.ia_service import ia_engine
        
        # Mapear datos
        genero_map = {"M": 1, "F": 2}
        genero = genero_map.get(cliente.gender, 1)
        edad = (date.today().year - cliente.birth_date.year) if cliente.birth_date else 25
        
        nivel_map = {
            "Sedentario": 1.20,
            "Ligero": 1.375,
            "Moderado": 1.55,
            "Activo": 1.725,
            "Muy activo": 1.90
        }
        nivel_actividad = nivel_map.get(cliente.activity_level, 1.20)
        
        objetivo_map = {
            "Perder peso": "perder",
            "Mantener peso": "mantener",
            "Ganar masa": "ganar"
        }
        objetivo = objetivo_map.get(cliente.goal, "mantener")
        
        # Calcular calorías con el modelo ML
        calorias_fallback = ia_engine.calcular_requerimiento(
            genero=genero,
            edad=edad,
            peso=cliente.weight,
            talla=cliente.height,
            nivel_actividad=nivel_actividad,
            objetivo=objetivo
        )
        
        # Calcular macros dinámicos según objetivo
        if objetivo == "perder":
            pct_proteina, pct_carbohidratos, pct_grasas = 0.35, 0.30, 0.35
        elif objetivo == "ganar":
            pct_proteina, pct_carbohidratos, pct_grasas = 0.30, 0.45, 0.25
        else:
            pct_proteina, pct_carbohidratos, pct_grasas = 0.30, 0.40, 0.30
        
        proteinas_g = round((calorias_fallback * pct_proteina) / 4, 1)
        carbohidratos_g = round((calorias_fallback * pct_carbohidratos) / 4, 1)
        grasas_g = round((calorias_fallback * pct_grasas) / 9, 1)
        
        plan_objetivo = {
            "calorias_objetivo": calorias_fallback,
            "proteinas_objetivo_g": proteinas_g,
            "carbohidratos_objetivo_g": carbohidratos_g,
            "grasas_objetivo_g": grasas_g,
            "distribucion": {
                "proteina_pct": int(pct_proteina * 100),
                "carbohidratos_pct": int(pct_carbohidratos * 100),
                "grasas_pct": int(pct_grasas * 100)
            },
            "validado": False,
            "plan_id": None,
            "es_fallback": True  # 🆕 Indica que es calculado temporalmente
        }
        
        print(f"✅ Fallback IA: {calorias_fallback:.0f} kcal | P:{proteinas_g}g C:{carbohidratos_g}g G:{grasas_g}g")

    # 3. Generar el Insight Dinámico usando la meta correcta del plan
    meta_calorias = plan_objetivo["calorias_objetivo"] if plan_objetivo else round(calcular_tmb(cliente), 1)
    consumido = consumo_actual["calorias"]
    
    # Generar insight basado en el progreso real vs meta del plan
    if consumido == 0:
        ai_insight = f"{cliente.first_name}, aún no has registrado calorías hoy. Tu meta es {meta_calorias:.0f} kcal."
    elif consumido < meta_calorias * 0.5:
        restante = meta_calorias - consumido
        ai_insight = f"{cliente.first_name}, debes aumentar tu ingesta calórica para alcanzar tu meta de {meta_calorias:.0f} kcal hoy. Te faltan {restante:.0f} kcal."
    elif consumido >= meta_calorias * 0.9 and consumido <= meta_calorias * 1.1:
        ai_insight = f"¡Excelente, {cliente.first_name}! Estás en tu meta de {meta_calorias:.0f} kcal. Sigue así."
    elif consumido > meta_calorias * 1.1:
        exceso = consumido - meta_calorias
        ai_insight = f"{cliente.first_name}, has superado tu meta de {meta_calorias:.0f} kcal en {exceso:.0f} kcal. Modera tu ingesta."
    else:
        restante = meta_calorias - consumido
        porcentaje = (consumido / meta_calorias * 100) if meta_calorias > 0 else 0
        ai_insight = f"Vas bien, {cliente.first_name}. Llevas {porcentaje:.0f}% de tu meta ({meta_calorias:.0f} kcal). Te quedan {restante:.0f} kcal."


    # 4. Respuesta final con macros objetivo incluidos
    return {
        "dieta_recomendada": {
            "calorias_diarias": consumo_actual["calorias"],
            "proteinas_g": consumo_actual["proteinas"],
            "carbohidratos_g": consumo_actual["carbohidratos"],
            "grasas_g": consumo_actual["grasas"],
            "gasto_metabolico_basal": round(calcular_tmb(cliente), 1),
            "imc": round(calcular_imc(cliente.weight, cliente.height), 1),
        },
        "plan_nutricional": plan_objetivo,  # 🆕 Macros del plan asignado
        "ai_insight": ai_insight
    }


@router.get("/clientes/{cliente_id}/calorias-tendencia")
async def get_calories_trend(
    cliente_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Obtiene la tendencia de calorías de los últimos 7 días usando datos reales
    """
    from app.models.user import User
    from app.models.historial import ProgresoCalorias
    if isinstance(current_user, User) and hasattr(current_user, 'role_name') and current_user.role_name not in ['admin', 'nutritionist', 'coach']:
        if current_user.id != cliente_id:
            raise HTTPException(status_code=403, detail="No autorizado")

    dias_semana = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
    resultado = []

    # Obtener progreso real de los últimos 7 días
    fecha_limite = datetime.now().date() - timedelta(days=6)

    progreso_calorias = db.query(ProgresoCalorias).filter(
        ProgresoCalorias.client_id == cliente_id,
        ProgresoCalorias.fecha >= fecha_limite
    ).order_by(ProgresoCalorias.fecha).all()

    # Crear mapa de fecha -> progreso
    progreso_por_fecha = {p.fecha: p for p in progreso_calorias}

    # Generar datos para los últimos 7 días
    for i in range(7):
        fecha = datetime.now().date() - timedelta(days=6-i)
        dia_semana_idx = fecha.weekday()

        # Buscar datos reales para esta fecha
        progreso_dia = progreso_por_fecha.get(fecha)

        if progreso_dia:
            consumidas = progreso_dia.calorias_consumidas or 0
            quemadas = progreso_dia.calorias_quemadas or 0
        else:
            # Si no hay datos reales, buscar en planes nutricionales
            plan_dia = db.query(PlanDiario).join(PlanNutricional).filter(
                PlanNutricional.client_id == cliente_id,
                PlanDiario.dia_numero == (dia_semana_idx + 1)
            ).first()
            consumidas = plan_dia.calorias_dia if plan_dia else 0
            quemadas = 0

        resultado.append({
            "dia": dias_semana[dia_semana_idx],
            "consumidas": consumidas,
            "quemadas": quemadas
        })

    return resultado


@router.get("/clientes/{cliente_id}/peso-historial")
async def get_weight_history(
    cliente_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Obtiene el historial de peso real de los últimos 6 meses
    """
    from app.models.user import User
    from app.models.historial import HistorialPeso
    if isinstance(current_user, User) and hasattr(current_user, 'role_name') and current_user.role_name not in ['admin', 'nutritionist', 'coach']:
        if current_user.id != cliente_id:
            raise HTTPException(status_code=403, detail="No autorizado")

    cliente = db.query(Client).filter(Client.id == cliente_id).first()
    if not cliente:
        raise HTTPException(status_code=404, detail="Cliente no encontrado")

    # Obtener historial de peso real de los últimos 6 meses
    from datetime import datetime, timedelta
    fecha_limite = datetime.now().date() - timedelta(days=180)  # 6 meses atrás

    registros_peso = db.query(HistorialPeso).filter(
        HistorialPeso.client_id == cliente_id,
        HistorialPeso.fecha_registro >= fecha_limite
    ).order_by(HistorialPeso.fecha_registro).all()

    # Si no hay registros históricos, devolver al menos el peso actual
    if not registros_peso:
        return [{
            "mes": 0,
            "peso": cliente.weight or 70.0,
            "fecha": datetime.now().date().isoformat()
        }]

    # Convertir a formato mensual para el frontend
    resultado = []
    for registro in registros_peso:
        # Calcular meses atrás desde hoy
        meses_atras = (datetime.now().date() - registro.fecha_registro).days // 30
        resultado.append({
            "mes": meses_atras,
            "peso": registro.peso_kg,
            "fecha": registro.fecha_registro.isoformat()
        })

    return resultado


@router.get("/clientes/{cliente_id}/imc-historial")
async def get_imc_history(
    cliente_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Obtiene el historial de IMC real de los últimos 6 meses
    """
    from app.models.user import User
    from app.models.historial import HistorialIMC
    if isinstance(current_user, User) and hasattr(current_user, 'role_name') and current_user.role_name not in ['admin', 'nutritionist', 'coach']:
        if current_user.id != cliente_id:
            raise HTTPException(status_code=403, detail="No autorizado")

    cliente = db.query(Client).filter(Client.id == cliente_id).first()
    if not cliente:
        raise HTTPException(status_code=404, detail="Cliente no encontrado")

    # Obtener historial de IMC real de los últimos 6 meses
    from datetime import datetime, timedelta
    fecha_limite = datetime.now().date() - timedelta(days=180)  # 6 meses atrás

    registros_imc = db.query(HistorialIMC).filter(
        HistorialIMC.client_id == cliente_id,
        HistorialIMC.fecha_registro >= fecha_limite
    ).order_by(HistorialIMC.fecha_registro).all()

    # Si no hay registros históricos, calcular IMC actual
    if not registros_imc:
        imc_actual = calcular_imc(cliente.weight or 70.0, cliente.height or 170.0)
        categoria_actual = obtener_categoria_imc(imc_actual)
        return [{
            "mes": 0,
            "imc": round(imc_actual, 1),
            "categoria": categoria_actual,
            "fecha": datetime.now().date().isoformat()
        }]

    # Convertir a formato mensual para el frontend
    resultado = []
    for registro in registros_imc:
        # Calcular meses atrás desde hoy
        meses_atras = (datetime.now().date() - registro.fecha_registro).days // 30
        resultado.append({
            "mes": meses_atras,
            "imc": registro.imc,
            "categoria": registro.categoria,
            "fecha": registro.fecha_registro.isoformat()
        })

    return resultado


@router.get("/clientes/{cliente_id}/analisis-ia")
async def get_ai_analysis(
    cliente_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Obtiene el análisis de IA del cliente (TMB, IMC, categoría)
    """
    from app.models.user import User
    if isinstance(current_user, User) and hasattr(current_user, 'role_name') and current_user.role_name not in ['admin', 'nutritionist', 'coach']:
        if current_user.id != cliente_id:
            raise HTTPException(status_code=403, detail="No autorizado")
    
    cliente = db.query(Client).filter(Client.id == cliente_id).first()
    if not cliente:
        raise HTTPException(status_code=404, detail="Cliente no encontrado")
    
    tmb = calcular_tmb(cliente)
    imc = calcular_imc(cliente.weight, cliente.height)
    categoria_imc = obtener_categoria_imc(imc)
    
    return {
        "gasto_estimado": round(tmb, 1),
        "imc_actual": round(imc, 1),
        "imc_categoria": categoria_imc
    }


# ============ FUNCIONES AUXILIARES ============

def calcular_tmb(cliente: Client) -> float:
    """
    Calcula la Tasa Metabólica Basal usando la fórmula de Harris-Benedict revisada
    """
    # Calcular edad a partir de birth_date
    if cliente.birth_date:
        today = date.today()
        edad = today.year - cliente.birth_date.year - ((today.month, today.day) < (cliente.birth_date.month, cliente.birth_date.day))
    else:
        edad = 30  # Valor por defecto si no hay birth_date
    
    # Determinar género basado en el campo gender del cliente
    # Asumir masculino por defecto si no hay información de género
    genero = getattr(cliente, 'gender', 'M')  # Usar M o F según el modelo
    genero_num = 1 if genero == 'M' else 2
    
    peso = cliente.weight or 75
    estatura = cliente.height or 170
    
    if genero_num == 1:  # Masculino
        tmb = 88.362 + (13.397 * peso) + (4.799 * estatura) - (5.677 * edad)
    else:  # Femenino
        tmb = 447.593 + (9.247 * peso) + (3.098 * estatura) - (4.330 * edad)
    
    # Factor de actividad por defecto (sedentario)
    nivel_map = {
        "Sedentario": 1.20,
        "Ligero": 1.375,
        "Moderado": 1.55,
        "Activo": 1.725,
        "Muy activo": 1.90
    }
    factor = nivel_map.get(getattr(cliente, 'activity_level', 'Sedentario'), 1.20)
    return tmb * factor


def calcular_imc(peso: float, estatura: float) -> float:
    """
    Calcula el Índice de Masa Corporal (IMC)
    IMC = peso (kg) / (estatura (cm))²
    """
    if not estatura or estatura <= 0:
        return 0
    
    estatura_metros = estatura / 100  # Convertir de cm a metros
    return peso / (estatura_metros ** 2)


def obtener_categoria_imc(imc: float) -> str:
    """
    Determina la categoría del IMC según los estándares de la OMS
    """
    if imc < 18.5:
        return "Bajo peso"
    elif 18.5 <= imc < 25:
        return "Normal"
    elif 25 <= imc < 30:
        return "Sobrepeso"
    elif 30 <= imc < 35:
        return "Obesidad Grado I"
    elif 35 <= imc < 40:
        return "Obesidad Grado II"
    else:
        return "Obesidad Grado III"


# ===== ENDPOINTS PARA REGISTRO MANUAL DE DATOS HISTÓRICOS =====

from pydantic import BaseModel
from typing import Optional
from app.models.historial import HistorialPeso, HistorialIMC, ProgresoCalorias
from datetime import date

class PesoRegistro(BaseModel):
    peso_kg: float
    notas: Optional[str] = None

class CaloriasDiariasRegistro(BaseModel):
    calorias_consumidas: int
    calorias_quemadas: int

@router.post("/clientes/{cliente_id}/peso")
async def registrar_peso(
    cliente_id: int,
    peso_data: PesoRegistro,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Registra un nuevo peso para el cliente y calcula automáticamente el IMC
    """
    from app.models.user import User
    if isinstance(current_user, User) and hasattr(current_user, 'role_name') and current_user.role_name not in ['admin', 'nutritionist', 'coach']:
        if current_user.id != cliente_id:
            raise HTTPException(status_code=403, detail="No autorizado")

    cliente = db.query(Client).filter(Client.id == cliente_id).first()
    if not cliente:
        raise HTTPException(status_code=404, detail="Cliente no encontrado")

    # Validar rango de peso realista
    if peso_data.peso_kg < 30 or peso_data.peso_kg > 300:
        raise HTTPException(status_code=400, detail="Peso fuera de rango realista (30-300 kg)")

    # Verificar si ya existe registro para hoy
    hoy = date.today()
    registro_existente = db.query(HistorialPeso).filter(
        HistorialPeso.client_id == cliente_id,
        HistorialPeso.fecha_registro == hoy
    ).first()

    if registro_existente:
        # Actualizar registro existente
        registro_existente.peso_kg = peso_data.peso_kg
        registro_existente.notas = peso_data.notas
    else:
        # Crear nuevo registro
        nuevo_peso = HistorialPeso(
            client_id=cliente_id,
            peso_kg=peso_data.peso_kg,
            fecha_registro=hoy,
            notas=peso_data.notas
        )
        db.add(nuevo_peso)

    # Calcular y registrar IMC automáticamente
    if cliente.height and cliente.height > 0:
        imc = peso_data.peso_kg / ((cliente.height / 100) ** 2)
        categoria = obtener_categoria_imc(imc)

        # Verificar si ya existe IMC para hoy
        imc_existente = db.query(HistorialIMC).filter(
            HistorialIMC.client_id == cliente_id,
            HistorialIMC.fecha_registro == hoy
        ).first()

        if imc_existente:
            imc_existente.imc = imc
            imc_existente.categoria = categoria
        else:
            nuevo_imc = HistorialIMC(
                client_id=cliente_id,
                imc=imc,
                categoria=categoria,
                fecha_registro=hoy
            )
            db.add(nuevo_imc)

    db.commit()

    return {
        "message": "Peso registrado exitosamente",
        "peso_kg": peso_data.peso_kg,
        "fecha": hoy.isoformat(),
        "imc_calculado": imc if cliente.height else None,
        "categoria_imc": categoria if cliente.height else None
    }

@router.post("/clientes/{cliente_id}/calorias-diarias")
async def registrar_calorias_diarias(
    cliente_id: int,
    calorias_data: CaloriasDiariasRegistro,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Registra o actualiza las calorías consumidas y quemadas para el día actual
    """
    from app.models.user import User
    if isinstance(current_user, User) and hasattr(current_user, 'role_name') and current_user.role_name not in ['admin', 'nutritionist', 'coach']:
        if current_user.id != cliente_id:
            raise HTTPException(status_code=403, detail="No autorizado")

    cliente = db.query(Client).filter(Client.id == cliente_id).first()
    if not cliente:
        raise HTTPException(status_code=404, detail="Cliente no encontrado")

    # Validar valores
    if calorias_data.calorias_consumidas < 0 or calorias_data.calorias_quemadas < 0:
        raise HTTPException(status_code=400, detail="Las calorías no pueden ser negativas")

    hoy = date.today()

    # Verificar si ya existe registro para hoy
    registro_existente = db.query(ProgresoCalorias).filter(
        ProgresoCalorias.client_id == cliente_id,
        ProgresoCalorias.fecha == hoy
    ).first()

    if registro_existente:
        # Actualizar registro existente
        registro_existente.calorias_consumidas = calorias_data.calorias_consumidas
        registro_existente.calorias_quemadas = calorias_data.calorias_quemadas
        message = "Calorías actualizadas exitosamente"
    else:
        # Crear nuevo registro
        nuevo_progreso = ProgresoCalorias(
            client_id=cliente_id,
            fecha=hoy,
            calorias_consumidas=calorias_data.calorias_consumidas,
            calorias_quemadas=calorias_data.calorias_quemadas
        )
        db.add(nuevo_progreso)
        message = "Calorías registradas exitosamente"

    db.commit()

    return {
        "message": message,
        "calorias_consumidas": calorias_data.calorias_consumidas,
        "calorias_quemadas": calorias_data.calorias_quemadas,
        "fecha": hoy.isoformat()
    }