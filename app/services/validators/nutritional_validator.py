"""
Validador nutricional: coherencia de macros y densidad calórica.
"""
from typing import Dict, Any, List, Optional
from app.services.validators.base_validator import BaseValidator, ValidationResult

# Atwater factors
_KCAL_PROT  = 4.0
_KCAL_CARB  = 4.0
_KCAL_GRAS  = 9.0
_TOLERANCIA = 0.15   # ±15%

# Densidad calórica kcal/100g
_DENSIDAD_MIN  =  20.0
_DENSIDAD_MAX  = 800.0

# Porcentajes máximos por momento del día (fracción del TDEE)
_TDEE_PCT: Dict[str, float] = {
    "desayuno": 0.30,
    "almuerzo": 0.40,
    "cena":     0.30,
    "snack":    0.15,
}


class NutritionalValidator(BaseValidator):
    """
    Valida coherencia nutricional de un plato.

    Entrada (datos):
        nombre_plato         : str
        peso_total_gramos    : float
        calorias_total       : float
        proteina_total       : float
        carbohidratos_total  : float
        grasas_total         : float
        tdee_usuario         : float  (opcional)
        momento_dia          : str    (desayuno|almuerzo|cena|snack, opcional)
    """

    def __init__(self) -> None:
        super().__init__("NutritionalValidator")

    def validar(self, datos: Dict[str, Any]) -> ValidationResult:
        nombre  = datos.get("nombre_plato", "")
        peso    = float(datos.get("peso_total_gramos", 0) or 0)
        kcal    = float(datos.get("calorias_total", 0) or 0)
        prot    = float(datos.get("proteina_total", 0) or 0)
        carb    = float(datos.get("carbohidratos_total", 0) or 0)
        gras    = float(datos.get("grasas_total", 0) or 0)
        tdee    = float(datos.get("tdee_usuario", 0) or 0)
        momento = (datos.get("momento_dia") or "almuerzo").lower()

        errores:     List[str] = []
        advertencias: List[str] = []
        sugerencias:  List[str] = []
        confianza = 100

        # 1. Valores negativos
        for campo, val in [("Calorías", kcal), ("Proteína", prot),
                           ("Carbohidratos", carb), ("Grasas", gras)]:
            if val < 0:
                errores.append(f"{campo} negativo: {val}")
        if errores:
            return self._crear_resultado(False, 0, errores=errores)

        # 2. Todos en cero
        if kcal == 0 and prot == 0 and carb == 0 and gras == 0:
            return self._crear_resultado(False, 0, errores=["Todos los macros son 0"])

        # 3. Atwater
        kcal_calc = prot * _KCAL_PROT + carb * _KCAL_CARB + gras * _KCAL_GRAS
        if kcal_calc > 0:
            desv = abs(kcal - kcal_calc) / kcal_calc
            if desv > _TOLERANCIA:
                advertencias.append(
                    f"Atwater: declarado {kcal:.0f} kcal vs calculado {kcal_calc:.0f} kcal "
                    f"(desviación {desv*100:.1f}%, tolerancia {_TOLERANCIA*100:.0f}%)"
                )
                confianza -= 10

        # 4. Densidad calórica
        if peso > 0:
            densidad = kcal / peso * 100
            if densidad < _DENSIDAD_MIN:
                advertencias.append(
                    f"Densidad calórica muy baja: {densidad:.1f} kcal/100g (mín {_DENSIDAD_MIN:.0f})"
                )
                confianza -= 5
            elif densidad > _DENSIDAD_MAX:
                advertencias.append(
                    f"Densidad calórica muy alta: {densidad:.1f} kcal/100g (máx {_DENSIDAD_MAX:.0f})"
                )
                confianza -= 10
        else:
            advertencias.append("Peso total no informado — no se puede calcular densidad")

        # 5. Proporciones de macros
        if kcal > 0:
            pct_prot = prot * _KCAL_PROT / kcal * 100
            pct_carb = carb * _KCAL_CARB / kcal * 100
            pct_gras = gras * _KCAL_GRAS / kcal * 100
            if pct_gras > 50:
                advertencias.append(f"Grasas representan {pct_gras:.1f}% de las kcal (típico ≤35%)")
                confianza -= 5
            if pct_carb > 70:
                advertencias.append(f"Carbohidratos representan {pct_carb:.1f}% de las kcal (típico ≤65%)")
        else:
            pct_prot = pct_carb = pct_gras = 0.0

        # 6. Contexto TDEE
        if tdee > 0:
            pct_tdee = kcal / tdee * 100
            limite   = _TDEE_PCT.get(momento, 0.40) * 100
            if pct_tdee > limite:
                advertencias.append(
                    f"{momento.capitalize()}: {kcal:.0f} kcal = {pct_tdee:.1f}% del TDEE "
                    f"(límite {limite:.0f}% → {tdee * limite / 100:.0f} kcal)"
                )

        densidad_final = round(kcal / peso * 100, 1) if peso > 0 else 0.0

        return self._crear_resultado(
            es_valido=len(errores) == 0,
            confianza=max(0, confianza),
            errores=errores,
            advertencias=advertencias,
            sugerencias=sugerencias,
            metadata={
                "nombre_plato":      nombre,
                "densidad_calorica": densidad_final,
                "proporciones": {
                    "proteina_pct":      round(pct_prot, 1),
                    "carbohidratos_pct": round(pct_carb, 1),
                    "grasas_pct":        round(pct_gras, 1),
                },
            },
        )
