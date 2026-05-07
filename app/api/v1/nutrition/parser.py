from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional

from app.core.database import get_db
from app.api.routes.auth import get_current_user
from app.services.nlp_food_extractor import NLPFoodExtractor
from app.services.ia_service import ia_engine

router = APIRouter(tags=["Nutrition Parser"])

class ParseIngredientsRequest(BaseModel):
    texto: str

class ParsedIngredient(BaseModel):
    nombre: str
    cantidad: float
    unidad: str
    gramos_totales: float
    calorias: float
    proteinas_g: float
    carbohidratos_g: float
    grasas_g: float

class ParseIngredientsResponse(BaseModel):
    ingredientes: List[ParsedIngredient]
    calorias_total: float
    proteinas_total: float
    carbohidratos_total: float
    grasas_total: float
    advertencia: Optional[str] = None

@router.post("/parse_ingredients", response_model=ParseIngredientsResponse)
async def parse_ingredients(
    request: ParseIngredientsRequest,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Parsea una cadena de texto (ej: '200g de arroz y 50g de pollo') y devuelve 
    los ingredientes con sus macronutrientes escalados matemáticamente, 
    sin guardarlos en el historial del usuario.
    """
    if not request.texto or not request.texto.strip():
        raise HTTPException(status_code=400, detail="El texto está vacío")
        
    try:
        extractor = NLPFoodExtractor(ia_service=ia_engine, db=db)
        
        # extraer asume que devuelve un objeto ResultadoExtraccion
        resultado = await extractor.extraer(request.texto)
        
        if not resultado:
            return ParseIngredientsResponse(
                ingredientes=[],
                calorias_total=0,
                proteinas_total=0,
                carbohidratos_total=0,
                grasas_total=0,
                advertencia="No pude entender los ingredientes."
            )
            
        ingredientes_list = []
        for item in resultado.items:
            ingredientes_list.append(ParsedIngredient(
                nombre=item.alimento,
                cantidad=item.cantidad,
                unidad=item.unidad,
                gramos_totales=item.gramos_totales,
                calorias=item.calorias,
                proteinas_g=item.proteinas_g,
                carbohidratos_g=item.carbohidratos_g,
                grasas_g=item.grasas_g,
            ))
            
        return ParseIngredientsResponse(
            ingredientes=ingredientes_list,
            calorias_total=resultado.calorias_total,
            proteinas_total=resultado.proteinas_total,
            carbohidratos_total=resultado.carbohidratos_total,
            grasas_total=resultado.grasas_total,
            advertencia=resultado.advertencia
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
