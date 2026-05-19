"""
Trazabilidad de ingesta.

comida_registros  → fuente de verdad auditada por evento
progreso_calorias → derivado: recalcular_progreso_diario() lo mantiene sincronizado
"""
from __future__ import annotations

from datetime import date
from typing import Optional

from sqlalchemy import func as sqlfunc
from sqlalchemy.orm import Session

from app.core.logging_config import get_logger
from app.models.comida_registro import ComidaRegistro
from app.models.historial import ProgresoCalorias

logger = get_logger("trazabilidad")

_TIPO_RESOLUCION_MAP: dict[str, str] = {
    "platos":         "bd_plato",
    "plato_dinamico": "plato_dinamico",
    "manual":         "manual",
    "llm":            "llm_estimado",
    "bd":             "bd_alimento",
    "nlp_extractor":  "bd_alimento",
    "usda":           "bd_alimento",
    "fatsecret":      "bd_alimento",
    "estimado":       "llm_estimado",
    "postgres":       "bd_alimento",
}

_CONFIANZA_MAP: dict[str, float] = {
    "bd_plato":       1.0,
    "bd_alimento":    1.0,
    "plato_dinamico": 0.85,
    "llm_estimado":   0.5,
    "manual":         1.0,
}


def recalcular_progreso_diario(
    client_id: int,
    fecha: date,
    db: Session,
) -> ProgresoCalorias:
    """
    Suma los ComidaRegistro del día y actualiza (o crea) la fila en progreso_calorias.
    Retorna el objeto actualizado sin hacer commit.
    """
    totales = db.query(
        sqlfunc.coalesce(sqlfunc.sum(ComidaRegistro.kcal), 0.0),
        sqlfunc.coalesce(sqlfunc.sum(ComidaRegistro.proteina_g), 0.0),
        sqlfunc.coalesce(sqlfunc.sum(ComidaRegistro.carbohidratos_g), 0.0),
        sqlfunc.coalesce(sqlfunc.sum(ComidaRegistro.grasas_g), 0.0),
    ).filter(
        ComidaRegistro.client_id == client_id,
        ComidaRegistro.fecha == fecha,
    ).first()

    progreso = db.query(ProgresoCalorias).filter(
        ProgresoCalorias.client_id == client_id,
        ProgresoCalorias.fecha == fecha,
    ).first()
    if not progreso:
        progreso = ProgresoCalorias(client_id=client_id, fecha=fecha)
        db.add(progreso)

    progreso.calorias_consumidas      = int(round(float(totales[0])))
    progreso.proteinas_consumidas     = round(float(totales[1]), 1)
    progreso.carbohidratos_consumidos = round(float(totales[2]), 1)
    progreso.grasas_consumidas        = round(float(totales[3]), 1)

    return progreso


def crear_comida_registros(
    client_id: int,
    fecha: date,
    extraccion: dict,
    texto_original: str,
    db: Session,
    momento: Optional[str] = None,
) -> list[ComidaRegistro]:
    """
    Crea filas en comida_registros desde una extracción NLP.
    Un registro por alimento detectado; macros distribuidas equitativamente.
    Retorna objetos añadidos a la sesión (sin flush/commit).
    """
    origen = extraccion.get("origen", "bd")
    tipo_resolucion = _TIPO_RESOLUCION_MAP.get(origen, "bd_alimento")
    confianza = _CONFIANZA_MAP.get(tipo_resolucion, 1.0)

    nombres = extraccion.get("alimentos_detectados") or []
    if not nombres:
        logger.warning("crear_comida_registros: sin alimentos_detectados — omitido")
        return []

    texto_trunc = (texto_original or "")[:500] or None
    registros: list[ComidaRegistro] = []

    # Usar macros individuales cuando están disponibles (plato + extras)
    alimentos_con_macros = extraccion.get("alimentos_con_macros")
    if alimentos_con_macros:
        for item in alimentos_con_macros:
            reg = ComidaRegistro(
                client_id=client_id,
                fecha=fecha,
                nombre_alimento=item["nombre"][:255],
                kcal=round(float(item.get("kcal", 0)), 2),
                proteina_g=round(float(item.get("prot_g", 0)), 2),
                carbohidratos_g=round(float(item.get("carb_g", 0)), 2),
                grasas_g=round(float(item.get("gras_g", 0)), 2),
                tipo_resolucion=tipo_resolucion,
                confianza=confianza,
                texto_original=texto_trunc,
                momento=momento,
            )
            db.add(reg)
            registros.append(reg)
    else:
        # Fallback: distribución equitativa para alimentos simples (1 solo alimento)
        kcal_t = float(extraccion.get("calorias", 0) or 0)
        prot_t = float(extraccion.get("proteinas_g", 0) or 0)
        carb_t = float(extraccion.get("carbohidratos_g", 0) or 0)
        gras_t = float(extraccion.get("grasas_g", 0) or 0)
        n = len(nombres)
        for nombre in nombres:
            reg = ComidaRegistro(
                client_id=client_id,
                fecha=fecha,
                nombre_alimento=nombre[:255],
                kcal=round(kcal_t / n, 2),
                proteina_g=round(prot_t / n, 2),
                carbohidratos_g=round(carb_t / n, 2),
                grasas_g=round(gras_t / n, 2),
                tipo_resolucion=tipo_resolucion,
                confianza=confianza,
                texto_original=texto_trunc,
                momento=momento,
            )
            db.add(reg)
            registros.append(reg)

    logger.info(
        "Creados %d ComidaRegistro para client=%d fecha=%s (tipo=%s confianza=%.2f)",
        len(registros), client_id, fecha, tipo_resolucion, confianza,
    )
    return registros
