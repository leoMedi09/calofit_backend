"""
Modelo de resultado nutricional con estado, confianza y modo de resolución.

Usado por plato_constructor.py para reportar la calidad del plato construido,
y como vocabulario compartido para trazabilidad en el pipeline de 5 capas.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


# ─── Rangos calóricos esperados por tipo/momento ────────────────────────────
KCAL_RANGES: dict[str, tuple[int, int]] = {
    "cebiche":            (150,  450),
    "tiradito":           (150,  400),
    "causa ferreñafana":  (400,  650),
    "arroz con pato":     (700, 1000),
    "jalea":              (500,  900),
    "ensalada":           ( 80,  450),
    "sopa":               (100,  400),
    "snack":              ( 80,  300),
    "merienda":           ( 80,  300),
    "desayuno":           (300,  600),
    "almuerzo":           (600, 1000),
    "cena":               (300,  600),
    "default":            ( 80, 1100),
}

# Palabras clave que implican proteína animal en el nombre del plato
_PROTEIN_KEYWORDS: frozenset[str] = frozenset({
    "pollo", "res", "carne", "cerdo", "chancho", "pato", "cabrito",
    "pavo", "cordero", "pescado", "atun", "atún", "salmon", "salmón",
    "caballa", "trucha", "lenguado", "salpreso", "camarones", "langostino",
    "pulpo", "calamar", "jalea", "chicharron", "chicharrón",
})

# Score de confianza por fuente de datos del alimento
_FUENTE_SCORE: dict[str, float] = {
    "USDA (auto-aprendido)":      0.8,
    "FatSecret (auto-aprendido)": 0.8,
    "Groq (estimado)":            0.5,
    "manual":                     1.0,
    # Todo lo no listado (INS/CENAN, catálogo) → 1.0 (default)
}


# ─── Dataclass principal ─────────────────────────────────────────────────────

@dataclass
class ResultadoNutricional:
    """
    Resultado del proceso de construcción/resolución de un plato.

    estado:
        "ok"         — plato construido correctamente, macros verificados
        "incompleto" — ingrediente principal no resuelto o kcal fuera de rango
        "invalido"   — plato rechazado (no-alimento, ficción, transacción fallida)

    confianza:
        0.0–1.0 basado en fuentes de los ingredientes resueltos.
        1.0 = todos desde BD local/INS/CENAN.
        0.5 = estimado por LLM.

    modo_resolucion:
        "receta_bd"     — encontrado en catálogo platos (CAPA 1)
        "reconstruido"  — construido por plato_constructor (CAPA 1.5)
        "estimado_llm"  — estimado directamente por Groq (CAPA 5)
    """
    estado: Literal["ok", "incompleto", "invalido"]
    kcal: float
    proteina: float
    carbohidratos: float
    grasas: float
    confianza: float
    modo_resolucion: Literal["receta_bd", "reconstruido", "estimado_llm"]
    nombre_plato: str = ""
    advertencias: list[str] = field(default_factory=list)
    plato_id: Optional[int] = None

    @property
    def es_confiable(self) -> bool:
        return self.estado == "ok" and self.confianza >= 0.7

    def __str__(self) -> str:
        return (
            f"ResultadoNutricional('{self.nombre_plato}' "
            f"estado={self.estado} kcal={self.kcal:.0f} "
            f"confianza={self.confianza:.2f} modo={self.modo_resolucion})"
        )


# ─── Funciones helper ────────────────────────────────────────────────────────

def confidence_score(fuentes: list[str]) -> float:
    """
    Calcula la confianza media del plato en base a las fuentes de sus ingredientes.

    - BD local / INS/CENAN / USDA   → 0.8-1.0
    - FatSecret                     → 0.8
    - Groq (estimado)               → 0.5
    - Ingrediente faltante (None)   → 0.0

    Args:
        fuentes: lista de strings con el campo `fuente` de cada Alimento resuelto.
                 Puede incluir None para ingredientes que no se resolvieron.
    Returns:
        Float entre 0.0 y 1.0, redondeado a 2 decimales.
    """
    if not fuentes:
        return 0.0
    scores = [_FUENTE_SCORE.get(f or "bd", 1.0) for f in fuentes]
    return round(sum(scores) / len(scores), 2)


def validar_plato_nutricional(
    nombre: str,
    kcal: float,
    proteina: float,
    tipo_plato: str = "default",
) -> list[str]:
    """
    Valida que el plato cumpla los rangos calóricos esperados y tenga
    la proteína mínima si el nombre implica proteína animal.

    Returns:
        Lista de strings con advertencias (lista vacía = válido).
    """
    advertencias: list[str] = []
    min_k, max_k = KCAL_RANGES.get(tipo_plato, KCAL_RANGES["default"])

    if kcal <= 0:
        advertencias.append(f"kcal=0 — plato sin macros resueltos (posible fallo de transacción)")
    elif kcal < min_k:
        advertencias.append(
            f"kcal {kcal:.0f} < mínimo esperado {min_k} para '{tipo_plato}' "
            f"(posible ingrediente principal omitido)"
        )
    elif kcal > max_k:
        advertencias.append(
            f"kcal {kcal:.0f} > máximo esperado {max_k} para '{tipo_plato}' "
            f"(verificar gramajes)"
        )

    nombre_lower = (nombre or "").lower()
    if any(kw in nombre_lower for kw in _PROTEIN_KEYWORDS) and proteina < 10.0:
        advertencias.append(
            f"proteína {proteina:.1f}g insuficiente para plato con proteína animal "
            f"('{nombre}') — ¿ingrediente principal resuelto?"
        )

    return advertencias


def validar_macros_atwater(
    kcal: float,
    proteina: float,
    carbohidratos: float,
    grasas: float,
) -> tuple[bool, str]:
    """
    Verifica que kcal sea coherente con macros usando factores Atwater (±15%).
    Retorna (True, "") si válido; (False, motivo) si inconsistente.
    Omite ítems con kcal < 5 (agua, condimentos de traza).
    """
    if kcal < 5.0:
        return True, ""
    kcal_calculada = proteina * 4.0 + carbohidratos * 4.0 + grasas * 9.0
    if kcal_calculada < 1.0:
        return False, (
            f"macros insuficientes para kcal={kcal:.1f} "
            f"(Atwater produce {kcal_calculada:.1f} kcal)"
        )
    desviacion = abs(kcal - kcal_calculada) / kcal
    # Tolerancia ampliada al 30%: la fórmula Atwater es una aproximación.
    # Mariscos, legumbres, fibra, alcohol y alimentos procesados pueden superar
    # el 15% sin ser datos inválidos. Rechazar al 30% cubre solo errores reales.
    if desviacion > 0.30:
        return False, (
            f"inconsistencia nutricional: kcal_bd={kcal:.1f} "
            f"vs kcal_atwater={kcal_calculada:.1f} "
            f"(desviación {desviacion:.0%})"
        )
    return True, ""
