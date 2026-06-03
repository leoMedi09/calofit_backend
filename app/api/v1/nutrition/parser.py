from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional

from app.core.database import get_db
from app.api.routes.auth import get_current_user
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
    current_user=Depends(get_current_user),
):
    """
    Estima macros de un texto de alimento usando LLM (USDA/INS-CENAN).
    Funciona con CUALQUIER alimento — sin lookup de BD.
    No guarda nada; solo devuelve los macros estimados para preview.
    """
    if not request.texto or not request.texto.strip():
        raise HTTPException(status_code=400, detail="El texto está vacío")

    from app.services.llm_registro import _PROMPT_COMIDA, _parse_json

    try:
        prompt = _PROMPT_COMIDA.format(mensaje=request.texto)
        raw = await ia_engine._llamar_groq(prompt, max_tokens=500, temp=0.0)
        datos = _parse_json(raw)

        if not datos or not datos.get("alimentos"):
            return ParseIngredientsResponse(
                ingredientes=[],
                calorias_total=0, proteinas_total=0,
                carbohidratos_total=0, grasas_total=0,
                advertencia="No identifiqué ese alimento. Intenta con más detalle.",
            )

        # Filtrar ficticios (es_real: false)
        items_reales = [a for a in datos["alimentos"] if a.get("es_real", True) is not False]
        if not items_reales:
            return ParseIngredientsResponse(
                ingredientes=[],
                calorias_total=0, proteinas_total=0,
                carbohidratos_total=0, grasas_total=0,
                advertencia="Ese alimento no existe en ninguna base de datos nutricional.",
            )

        ingredientes_list = []
        kcal_total = prot_total = carb_total = grasa_total = 0.0

        for item in items_reales:
            p   = float(item.get("prot_g",  0) or 0)
            c   = float(item.get("carb_g",  0) or 0)
            g   = float(item.get("grasa_g", 0) or 0)
            # Calcular kcal desde macros (consistente con el resto del sistema)
            k   = round(4 * p + 4 * c + 9 * g, 1) or float(item.get("kcal", 0) or 0)
            grm = float(item.get("porcion_g", 100) or 100)

            ingredientes_list.append(ParsedIngredient(
                nombre=item.get("nombre", "Alimento"),
                cantidad=grm,
                unidad="g",
                gramos_totales=grm,
                calorias=k,
                proteinas_g=p,
                carbohidratos_g=c,
                grasas_g=g,
            ))
            kcal_total  += k
            prot_total  += p
            carb_total  += c
            grasa_total += g

        return ParseIngredientsResponse(
            ingredientes=ingredientes_list,
            calorias_total=round(kcal_total, 1),
            proteinas_total=round(prot_total, 1),
            carbohidratos_total=round(carb_total, 1),
            grasas_total=round(grasa_total, 1),
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
