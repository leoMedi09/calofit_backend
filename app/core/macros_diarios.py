"""
Macros — fuentes unificadas (Problema 1).

1) ``macros_desde_calorias_peso_objetivo``: día completo con peso + objetivo
   (IA, dashboard, nutricionista, calculador_requerimientos).

2) ``macros_desde_calorias_pct_clasico``: reparto % sobre un total kcal
   (CalculadorDietaAutomatica, ``parsear_macros_de_texto`` cuando solo hay kcal).
"""
from __future__ import annotations

from typing import Dict, Optional


def macros_desde_calorias_peso_objetivo(
    calorias: float,
    objetivo: str,
    peso_kg: float = 70.0,
) -> Dict[str, float]:
    """
    Retorna gramos de P/C/G y calorías_totales (redondeo coherente con el resto del backend).
    """
    obj = (objetivo or "mantener").lower()
    cal = float(calorias)
    peso = float(peso_kg or 70.0)

    if "perder" in obj:
        prot_g, gras_ratio = round(peso * 2.1, 1), 0.25
    elif "ganar" in obj:
        prot_g, gras_ratio = round(peso * 1.8, 1), 0.25
    else:
        prot_g, gras_ratio = round(peso * 1.6, 1), 0.25

    cal_gras = cal * gras_ratio
    cal_carb = cal - (prot_g * 4) - cal_gras
    carb_g = round(max(0.0, cal_carb) / 4, 1)
    fat_g = round(cal_gras / 9, 1)

    return {
        "calorias_totales": round(cal, 1),
        "proteinas_g": prot_g,
        "carbohidratos_g": carb_g,
        "grasas_g": fat_g,
    }


def macros_desde_calorias_pct_clasico(
    calorias: float,
    objetivo: Optional[str] = None,
) -> Dict[str, float]:
    """
    Reparto % P/C/G sobre un total de calorías (sin peso corporal).
    - *ganar*: 25 / 50 / 25 (como el antiguo ``obtener_macros_desglosados``).
    - *perder*: 35 / 35 / 30.
    - resto (mantener u omitido): 30 / 40 / 30 (``CalculadorDietaAutomatica`` histórico).
    """
    cal = float(calorias)
    obj = (objetivo or "mantener").lower()
    if "ganar" in obj:
        pct_p, pct_c, pct_g = 0.25, 0.50, 0.25
    elif "perder" in obj:
        pct_p, pct_c, pct_g = 0.35, 0.35, 0.30
    else:
        pct_p, pct_c, pct_g = 0.30, 0.40, 0.30

    proteinas_g = round((cal * pct_p) / 4, 1)
    carbohidratos_g = round((cal * pct_c) / 4, 1)
    grasas_g = round((cal * pct_g) / 9, 1)
    return {
        "calorias_totales": round(cal, 1),
        "proteinas_g": proteinas_g,
        "carbohidratos_g": carbohidratos_g,
        "grasas_g": grasas_g,
    }


def porcentajes_aprox_desde_gramos(
    calorias: float,
    proteinas_g: float,
    carbohidratos_g: float,
    grasas_g: float,
) -> Dict[str, int]:
    """Porcentajes de energía a partir de gramos (enteros que suman ~100)."""
    cal = float(calorias)
    if cal <= 0:
        return {"p": 0, "c": 0, "g": 0}
    kcal_p = proteinas_g * 4
    kcal_c = carbohidratos_g * 4
    kcal_g = grasas_g * 9
    total = kcal_p + kcal_c + kcal_g
    if total <= 0:
        return {"p": 0, "c": 0, "g": 0}
    p = int(round(100 * kcal_p / total))
    c = int(round(100 * kcal_c / total))
    g = int(round(100 * kcal_g / total))
    diff = 100 - (p + c + g)
    if diff != 0:
        c = max(0, c + diff)
    return {"p": p, "c": c, "g": g}
