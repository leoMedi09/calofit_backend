from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.api.routes.auth import get_current_user
from app.models.client import Client
from app.models.historial import ProgresoCalorias
from app.models.nutricion import PlanNutricional, PlanDiario
from datetime import datetime, date
from typing import List, Dict, Any

router = APIRouter()


@router.get("/hoy")
async def obtener_balance_hoy(
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    📊 MI BALANCE DIARIO: Ver todos los registros de hoy
    
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
    plan_activo = db.query(PlanNutricional).filter(
        PlanNutricional.client_id == cliente.id,
        PlanNutricional.status == "validado"
    ).first()
    
    objetivo_diario = plan_activo.calorias_ia_base if plan_activo else 2000
    
    # Obtener progreso de hoy
    hoy = date.today()
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
    
    alimentos_hoy = db.query(PreferenciaAlimento).filter(
        PreferenciaAlimento.client_id == cliente.id,
        func.date(PreferenciaAlimento.ultima_vez) == hoy
    ).all()
    
    ejercicios_hoy = db.query(PreferenciaEjercicio).filter(
        PreferenciaEjercicio.client_id == cliente.id,
        func.date(PreferenciaEjercicio.ultima_vez) == hoy
    ).all()
    
    return {
        "fecha": hoy.isoformat(),
        "resumen": {
            "calorias_consumidas": calorias_consumidas or 0,
            "calorias_quemadas": calorias_quemadas or 0,
            "calorias_restantes": calorias_restantes,
            "objetivo_diario": objetivo_diario
        },
        "alimentos_registrados": [
            {
                "id": alimento.id,
                "nombre": alimento.alimento.capitalize(),
                "frecuencia_total": alimento.frecuencia,
                "puntuacion": round(alimento.puntuacion, 2),
                "hora_registro": alimento.ultima_vez.strftime("%H:%M:%S")
            }
            for alimento in alimentos_hoy
        ],
        "ejercicios_registrados": [
            {
                "id": ejercicio.id,
                "nombre": ejercicio.ejercicio.capitalize(),
                "frecuencia_total": ejercicio.frecuencia,
                "hora_registro": ejercicio.ultima_vez.strftime("%H:%M:%S")
            }
            for ejercicio in ejercicios_hoy
        ]
    }


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
    hoy = date.today()
    progreso_hoy = db.query(ProgresoCalorias).filter(
        ProgresoCalorias.client_id == cliente.id,
        ProgresoCalorias.fecha == hoy
    ).first()
    
    if progreso_hoy:
        # Obtener plan para calcular restantes
        plan_activo = db.query(PlanNutricional).filter(
            PlanNutricional.client_id == cliente.id,
            PlanNutricional.status == "validado"
        ).first()
        
        objetivo = plan_activo.calorias_ia_base if plan_activo else 2000
        calorias_restantes = objetivo - (progreso_hoy.calorias_consumidas or 0) + (progreso_hoy.calorias_quemadas or 0)
        
        nuevo_balance = {
            "calorias_consumidas": progreso_hoy.calorias_consumidas or 0,
            "calorias_quemadas": progreso_hoy.calorias_quemadas or 0,
            "calorias_restantes": calorias_restantes
        }
    else:
        nuevo_balance = {
            "calorias_consumidas": 0,
            "calorias_quemadas": 0,
            "calorias_restantes": 2000
        }
    
    return {
        "success": True,
        "mensaje": f"'{nombre_registro.capitalize()}' eliminado exitosamente",
        "nuevo_balance": nuevo_balance
    }
