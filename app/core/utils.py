import re
from datetime import datetime, date, timedelta, timezone
from typing import Any, Dict, Optional, Sequence

def get_peru_now() -> datetime:
    """Retorna la fecha y hora actual en zona horaria de Perú (UTC-5)"""
    # UTC now
    utc_now = datetime.now(timezone.utc)
    # Peru is UTC-5
    peru_time = utc_now - timedelta(hours=5)
    return peru_time

def get_peru_date() -> date:
    """Retorna la fecha actual en Perú"""
    return get_peru_now().date()


def inferir_momento_dia_peru() -> str:
    """
    Devuelve el momento del día según la hora actual en Perú (UTC-5).
    Fuente canónica usada por todos los módulos del backend.

    Rangos:
      05–09 → desayuno
      10–14 → almuerzo
      15–17 → merienda
      18–21 → cena
      22–04 → snack
    """
    hora = get_peru_now().hour
    if  5 <= hora <=  9: return "desayuno"
    if 10 <= hora <= 14: return "almuerzo"
    if 15 <= hora <= 17: return "merienda"
    if 18 <= hora <= 21: return "cena"
    return "snack"


def parsear_macros_de_texto(
    macros_str: str,
    objetivo_plato: Optional[str] = None,
) -> Optional[Dict[str, float]]:
    """
    Parsea string de macros en múltiples formatos que el LLM puede generar:
      - "P: 30g | C: 20g | G: 10g | Cal: 380kcal"
      - "653 kcal, 51g de proteína, 28g de grasa y 40g de carbohidratos"
      - "380 kcal | prot: 35g | carb: 45g | gras: 12g"
      - "Calorías: 500 | Proteínas: 40g | Carbohidratos: 50g | Grasas: 15g"

    Si solo hay calorías, estima P/C/G con ``macros_desde_calorias_pct_clasico``
    (opcional ``objetivo_plato`` para % ganar/perder/mantener).
    """
    from app.core.macros_diarios import macros_desde_calorias_pct_clasico
    if not macros_str or not macros_str.strip():
        return None
    s = macros_str.strip()

    # ── Calorías ────────────────────────────────────────────────────
    cal = (
        re.search(r'Cal(?:or[ií]as?)?:\s*([\d.,]+)', s, re.IGNORECASE) or
        re.search(r'([\d.,]+)\s*kcal', s, re.IGNORECASE) or
        re.search(r'([\d.,]+)\s*cal\b', s, re.IGNORECASE)
    )

    # ── Proteínas ───────────────────────────────────────────────────
    p = (
        re.search(r'P(?:rot(?:eína?s?)?)?\s*:\s*([\d.,]+)', s, re.IGNORECASE)
        or re.search(r"(?<![A-Za-z0-9])P\s+([\d.,]+)\s*g\b", s, re.IGNORECASE)
        or re.search(r'([\d.,]+)\s*g\s*(?:de\s+)?prot(?:eína?s?)?', s, re.IGNORECASE)
        or re.search(r'prot(?:eína?s?)?\s*[:\-]\s*([\d.,]+)', s, re.IGNORECASE)
        or re.search(r'prot(?:eína?s?)?\s+([\d.,]+)\s*g', s, re.IGNORECASE)
    )

    # ── Carbohidratos ────────────────────────────────────────────────
    c = (
        re.search(r'C(?:arb(?:ohidrat[eo]s?)?)?\s*:\s*([\d.,]+)', s, re.IGNORECASE) or
        re.search(r'([\d.,]+)\s*g\s*(?:de\s+)?carb(?:ohidrat[eo]s?)?', s, re.IGNORECASE) or
        re.search(r'carb(?:ohidrat[eo]s?)?\s*[:\-]\s*([\d.,]+)', s, re.IGNORECASE) or
        re.search(r'carb(?:ohidrat[eo]s?)?\s+([\d.,]+)\s*g', s, re.IGNORECASE)
    )

    # ── Grasas ───────────────────────────────────────────────────────
    g = (
        re.search(r'G(?:ras(?:as?)?)?\s*:\s*([\d.,]+)', s, re.IGNORECASE) or
        re.search(r'([\d.,]+)\s*g\s*(?:de\s+)?gras(?:as?)?', s, re.IGNORECASE) or
        re.search(r'gras(?:as?)?\s*[:\-]\s*([\d.,]+)', s, re.IGNORECASE) or
        re.search(r'gras(?:as?)?\s+([\d.,]+)\s*g', s, re.IGNORECASE)
    )

    try:
        def to_float(m):
            if m is None:
                return 0.0
            return float(m.group(1).replace(',', '.'))

        cal_val  = to_float(cal)
        prot_val = to_float(p)
        carb_val = to_float(c)
        gras_val = to_float(g)

        # Solo kcal: mismo reparto % que CalculadorDietaAutomatica (app.core.macros_diarios)
        if cal_val > 0 and prot_val == 0 and carb_val == 0 and gras_val == 0:
            est = macros_desde_calorias_pct_clasico(cal_val, objetivo_plato)
            prot_val = est["proteinas_g"]
            carb_val = est["carbohidratos_g"]
            gras_val = est["grasas_g"]

        if cal_val == 0 and prot_val == 0:
            return None

        # Si el modelo dio P/C/G pero olvidó Cal, derivar kcal (Atwater aprox.).
        atwater = 4.0 * prot_val + 4.0 * carb_val + 9.0 * gras_val
        if atwater > 0 and cal_val <= 0:
            cal_val = round(atwater, 1)

        return {
            "proteinas_g":    prot_val,
            "carbohidratos_g": carb_val,
            "grasas_g":       gras_val,
            "calorias":       cal_val,
        }
    except (ValueError, AttributeError):
        return None


# Palabras en nombre/ingredientes que implican aporte proteico relevante (plato típico).
_PALABRAS_FUENTE_PROTEINA = frozenset(
    (
        "pollo", "pechuga", "muslo", "pavo", "pato", "huevo", "huevos",
        "carne", "res", "ternera", "cerdo", "chancho", "chicharron", "chicharrón",
        "lomo", "bistec", "brocheta",
        "pescado", "pez", "trucha", "tilapia", "corvina", "chita", "caballa",
        "atún", "atun", "marisc", "langost", "camar", "pulpo", "calamar",
        "queso", "requesón", "requeson", "yogurt", "yoghurt", "leche", "suero",
        "lenteja", "lentejas", "garbanzo", "garbanzos", "frejol", "frejoles",
        "poroto", "porotos", "arveja", "arvejas", "haba", "habas",
        "quinua", "quinoa", "soya", "tofu", "sibayo",
        "tocino", "jamón", "jamon", "tocineta", "sardina", "caballa",
    )
)


def _texto_plato_sugiere_proteina(nombre_plato: str, ingredientes: Optional[Sequence[str]]) -> bool:
    txt = f"{nombre_plato or ''} {' '.join(ingredientes or ())}".lower()
    return any(p in txt for p in _PALABRAS_FUENTE_PROTEINA)


def coherenciar_macros_tarjeta(
    parsed: Optional[Dict[str, Any]],
    nombre_plato: str = "",
    ingredientes: Optional[Sequence[str]] = None,
) -> Optional[Dict[str, float]]:
    """
    Si el modelo devuelve P: 0g pero el plato claramente lleva proteínas, o P/C/G y Cal no cierran
    por Atwater, reequilibra gramos para que la tarjeta no muestre 0g de proteína de forma absurda.

    Caso típico: Cal ≈ 4C + 9G y P omitido/0 con «arroz con pollo» o «sopa de lentejas».
    """
    if not parsed:
        return parsed
    try:
        cal = float(parsed.get("calorias") or 0)
        p = float(parsed.get("proteinas_g") or 0)
        c = float(parsed.get("carbohidratos_g") or 0)
        g = float(parsed.get("grasas_g") or 0)
    except (TypeError, ValueError):
        return parsed
    if cal <= 0:
        return parsed

    atw = 4.0 * p + 4.0 * c + 9.0 * g
    residual = cal - atw
    sugiere = _texto_plato_sugiere_proteina(nombre_plato, ingredientes)

    if p < 2.0 and sugiere:
        # Energía ya «llenada» solo con C y G pero el nombre insiste en proteína: reparto mínimo creíble.
        if abs(residual) <= max(8.0, cal * 0.06):
            p_tgt = max(12.0, min(48.0, 0.22 * cal / 4.0))
            carb_kcal = max(0.0, cal - 4.0 * p_tgt - 9.0 * g)
            c_adj = max(0.0, carb_kcal / 4.0)
            p, c = p_tgt, c_adj
        elif residual > 3.0 * 4.0:
            p = max(p, residual / 4.0)
    elif p < 1.0 and residual > 4.0 * 3.0:
        p = max(p, residual / 4.0)

    return {
        "proteinas_g": round(p, 1),
        "carbohidratos_g": round(c, 1),
        "grasas_g": round(g, 1),
        "calorias": round(cal, 1),
    }


def calcular_metabolismo_basal(cliente) -> float:
    """
    Calcula la Tasa Metabólica Basal usando la fórmula de Harris-Benedict revisada (Mifflin-St Jeor es otra opción, pero Harris-Benedict es la estándar en el proyecto).
    """
    from datetime import date
    # Calcular edad a partir de birth_date
    if cliente.birth_date:
        today = date.today()
        edad = today.year - cliente.birth_date.year - ((today.month, today.day) < (cliente.birth_date.month, cliente.birth_date.day))
    else:
        edad = 30  # Valor por defecto si no hay birth_date
    
    # Determinar género basado en el campo gender del cliente
    genero = getattr(cliente, 'gender', 'M')
    peso = cliente.weight or 75
    estatura = cliente.height or 170
    
    if genero == 'M':  # Masculino
        tmb = 88.362 + (13.397 * peso) + (4.799 * estatura) - (5.677 * edad)
    else:  # Femenino
        tmb = 447.593 + (9.247 * peso) + (3.098 * estatura) - (4.330 * edad)
    
    # Factor de actividad
    nivel_map = {
        "Sedentario": 1.20,
        "Ligero": 1.375,
        "Moderado": 1.55,
        "Activo": 1.725,
        "Muy activo": 1.90
    }
    factor = nivel_map.get(getattr(cliente, 'activity_level', 'Sedentario'), 1.20)
    return tmb * factor

def obtener_macros_desglosados(
    calorias: float,
    objetivo: str = "Mantener peso",
    peso_kg: float = 70.0,
):
    """
    Desglose P/C/G a partir de calorías diarias y objetivo.
    Usa la misma regla que IAService.calcular_macros_optimizados (app.core.macros_diarios).
    """
    from app.core.macros_diarios import (
        macros_desde_calorias_peso_objetivo,
        porcentajes_aprox_desde_gramos,
    )

    m = macros_desde_calorias_peso_objetivo(calorias, objetivo, peso_kg)
    cal = m["calorias_totales"]
    pct = porcentajes_aprox_desde_gramos(
        cal, m["proteinas_g"], m["carbohidratos_g"], m["grasas_g"]
    )
    return {
        "calorias": int(round(cal)),
        "proteinas_g": m["proteinas_g"],
        "carbohidratos_g": m["carbohidratos_g"],
        "grasas_g": m["grasas_g"],
        "pct": {"p": pct["p"], "c": pct["c"], "g": pct["g"]},
    }
