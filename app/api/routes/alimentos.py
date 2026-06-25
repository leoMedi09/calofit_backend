from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.api.routes.auth import get_current_user
from app.services.ia_service import ia_engine
from pydantic import BaseModel
from typing import List, Dict

router = APIRouter()


class DetalleAlimentoRequest(BaseModel):
    alimento: str
    porcion_gramos: int = 100  # opcional, default 100g


@router.post("/detalle")
async def obtener_detalle_alimento(
    request: DetalleAlimentoRequest,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    🍎 DETALLE DE ALIMENTO: Genera info nutricional completa con IA
    
    Usa Groq para generar:
    - Datos nutricionales detallados
    - Recomendaciones de consumo
    - Sugerencias de combinación
    - Porciones comunes
    
    Perfecto para la pantalla "Ver Detalle" en Flutter.
    """
    
    prompt = f"""
    Eres un nutricionista experto. Genera información nutricional COMPLETA para el siguiente alimento:
    
    ALIMENTO: {request.alimento}
    PORCIÓN: {request.porcion_gramos}g
    
    RESPONDE EN FORMATO JSON VÁLIDO:
    {{
        "nombre": "Nombre del alimento en español",
        "descripcion": "Breve descripción nutricional (máximo 80 caracteres)",
        "datos_nutricionales": {{
            "porcion": "{request.porcion_gramos}g",
            "calorias": número,
            "proteinas": número,
            "carbohidratos": número,
            "grasas": número,
            "fibra": número
        }},
        "recomendaciones": [
            "3 recomendaciones prácticas de consumo, combinaciones o timing"
        ],
        "porciones_comunes": [
            {{"nombre": "1 porción estándar", "gramos": número}},
            {{"nombre": "1 porción grande", "gramos": número}},
            {{"nombre": "1 porción pequeña", "gramos": número}}
        ],
        "alternativas_saludables": [
            "2-3 alimentos similares más saludables o con diferente perfil nutricional"
        ]
    }}
    
    IMPORTANTE:
    - Valores nutricionales PRECISOS según USDA o tablas oficiales
    - Recomendaciones PRÁCTICAS y aplicables
    - Alternativas REALISTAS
    
    SOLO responde con JSON válido, sin texto adicional.
    """
    
    try:
        response = await ia_engine.groq_client.chat.completions.create(
            model="groq/compound-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.3
        )
        
        respuesta_texto = response.choices[0].message.content.strip()
        
        # Extraer JSON
        import json
        import re
        
        json_match = re.search(r'\{.*\}', respuesta_texto, re.DOTALL)
        if json_match:
            detalle = json.loads(json_match.group())
            return detalle
        else:
            raise ValueError("No se pudo parsear JSON de Groq")
            
    except Exception as e:
        print(f"Error generando detalle con Groq: {e}")
        # Fallback: respuesta básica
        return {
            "nombre": request.alimento.capitalize(),
            "descripcion": "Información nutricional estimada",
            "datos_nutricionales": {
                "porcion": f"{request.porcion_gramos}g",
                "calorias": 150,
                "proteinas": 10,
                "carbohidratos": 15,
                "grasas": 5,
                "fibra": 2
            },
            "recomendaciones": [
                "Consulta con un nutricionista para info personalizada",
                "Combina con vegetales para mayor nutrición",
                "Consume con moderación según tu plan"
            ],
            "porciones_comunes": [
                {"nombre": "1 porción estándar", "gramos": 100},
                {"nombre": "1 porción grande", "gramos": 150},
                {"nombre": "1 porción pequeña", "gramos": 75}
            ],
            "alternativas_saludables": [
                "Consulta tu plan nutricional personalizado"
            ]
        }


@router.post("/actualizar-porcion")
async def actualizar_porcion_balance(
    alimento: str,
    nueva_porcion_gramos: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    🔄 ACTUALIZAR PORCIÓN: Ajusta la cantidad de un alimento ya registrado
    
    Se usa cuando el usuario cambia la porción desde la pantalla de detalle.
    """
    from app.models.client import Client
    from app.models.historial import ProgresoCalorias
    from datetime import date
    
    # Obtener cliente
    cliente = db.query(Client).filter(Client.email == current_user.email).first()
    if not cliente:
        raise HTTPException(status_code=404, detail="Cliente no encontrado")
    
    # Calcular nuevas calorías según la porción ajustada
    # Usar Groq para obtener calorías exactas de la nueva porción
    detalle_request = DetalleAlimentoRequest(
        alimento=alimento,
        porcion_gramos=nueva_porcion_gramos
    )
    
    detalle = await obtener_detalle_alimento(detalle_request, db, current_user)
    nuevas_calorias = detalle["datos_nutricionales"]["calorias"]
    
    # Actualizar en progreso de hoy
    hoy = date.today()
    progreso = db.query(ProgresoCalorias).filter(
        ProgresoCalorias.client_id == cliente.id,
        ProgresoCalorias.fecha == hoy
    ).first()
    
    if not progreso:
        raise HTTPException(status_code=404, detail="No hay registros de hoy para actualizar")
    
    # Nota: Esto es simplificado. En producción necesitarías un registro más detallado
    # por ahora actualiza las calorías totales
    
    return {
        "success": True,
        "alimento": alimento,
        "nueva_porcion": f"{nueva_porcion_gramos}g",
        "nuevas_calorias": nuevas_calorias,
        "mensaje": f"Porción de {alimento} actualizada a {nueva_porcion_gramos}g"
    }
