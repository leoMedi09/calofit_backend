"""
Paso 2 — respuesta estructurada más estable para el cliente.

- ``schema_version``: versión del contrato en ``respuesta_estructurada``.
- ``macros_normalizados`` por sección comida: kcal y gramos P/C/G parseados del mismo ``macros`` que ve la tarjeta.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.core.utils import coherenciar_macros_tarjeta, parsear_macros_de_texto


class MacrosNormalizadosModel(BaseModel):
    """Valores numéricos listos para UI (evita depender solo del string)."""

    model_config = ConfigDict(extra="ignore")

    kcal: float = Field(0, ge=0)
    proteinas_g: float = Field(0, ge=0)
    carbohidratos_g: float = Field(0, ge=0)
    grasas_g: float = Field(0, ge=0)


def _macros_dict_desde_parse(parsed: Optional[Dict[str, float]]) -> Dict[str, float]:
    if not parsed:
        return {"kcal": 0.0, "proteinas_g": 0.0, "carbohidratos_g": 0.0, "grasas_g": 0.0}
    return {
        "kcal": float(parsed.get("calorias") or 0),
        "proteinas_g": float(parsed.get("proteinas_g") or 0),
        "carbohidratos_g": float(parsed.get("carbohidratos_g") or 0),
        "grasas_g": float(parsed.get("grasas_g") or 0),
    }


def enriquecer_respuesta_estructurada(
    respuesta_estructurada: Dict[str, Any],
    objetivo_cliente: Optional[str] = None,
) -> None:
    """
    Mutación in-place: añade schema_version y macros_normalizados en secciones comida.
    No elimina campos existentes (compatibilidad Flutter actual).
    """
    if not isinstance(respuesta_estructurada, dict):
        return

    respuesta_estructurada["schema_version"] = 2
    secciones: List[Dict[str, Any]] = respuesta_estructurada.get("secciones") or []
    if not isinstance(secciones, list):
        return

    for sec in secciones:
        if not isinstance(sec, dict):
            continue
        if sec.get("tipo") != "comida":
            continue
        raw = (sec.get("macros") or "").strip()
        if not raw:
            sec["macros_normalizados"] = _macros_dict_desde_parse(None)
            continue
        parsed = parsear_macros_de_texto(raw, objetivo_cliente)
        ing_list = sec.get("ingredientes")
        if isinstance(ing_list, list):
            ing_seq = [str(x) for x in ing_list if x]
        else:
            ing_seq = []
        parsed = coherenciar_macros_tarjeta(
            parsed,
            nombre_plato=str(sec.get("nombre") or ""),
            ingredientes=ing_seq,
        )
        blob = _macros_dict_desde_parse(parsed)
        try:
            m = MacrosNormalizadosModel.model_validate(blob)
            sec["macros_normalizados"] = m.model_dump()
        except Exception:
            sec["macros_normalizados"] = blob
