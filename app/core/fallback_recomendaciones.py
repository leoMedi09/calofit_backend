"""Opciones de respaldo de las recomendaciones de comida, etiquetadas por alérgeno.

Se usan cuando el modelo de lenguaje no responde o responde algo inservible. Cada opción lleva etiquetas
(frutos_secos, huevo, gluten, lacteos, azucar...) y se filtra por etiquetas, no por palabras sueltas del plato.
Las restricciones salen del perfil: `Client.medical_conditions` (texto libre: "Diabetes", "Celíaco", "Alergia al maní"...)
y `Client.forbidden_foods` (lista del nutricionista), más lo que el usuario pida en el mensaje ("no quiero huevo").

Criterio conservador: una alergia o intolerancia que no se pueda interpretar (p. ej. "alergia a la fresa") deja
sin recomendación específica y se deriva al nutricionista. No sustituye el criterio del nutricionista.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

from app.core.knn_filtros import _palabras, _tokens_de_prohibidos, norm


@dataclass(frozen=True)
class OpcionFallback:
    texto: str
    etiquetas: frozenset = frozenset()


def _o(texto: str, *etiquetas: str) -> OpcionFallback:
    return OpcionFallback(texto, frozenset(etiquetas))


OPCIONES: Dict[str, List[OpcionFallback]] = {
    "DESAYUNO": [
        _o("- Fruta picada con chía (~90 kcal, P:2g C:20g G:1g)"),
        _o("- Avena tibia con agua (~150 kcal, P:5g C:27g G:3g)", "gluten"),
        _o("- Tostada integral con palta (~130 kcal, P:3g C:16g G:7g)", "gluten"),
        _o("- Plátano con frutos secos (~180 kcal, P:4g C:28g G:7g)", "frutos_secos", "azucar"),
        _o("- Batido de frutas naturales (~110 kcal, P:2g C:25g G:1g)", "azucar"),
    ],
    "ALMUERZO": [
        _o("- Lentejas guisadas con verduras (~350 kcal, P:18g C:55g G:5g)"),
        _o("- Sopa de verduras con quinua (~300 kcal, P:10g C:50g G:5g)"),
        _o("- Ensalada de garbanzos y palta (~280 kcal, P:8g C:35g G:12g)"),
        _o("- Tofu a la plancha con brócoli y arroz (~380 kcal, P:22g C:45g G:10g)", "soya"),
        _o("- Crema de zapallo y arvejas (~290 kcal, P:9g C:42g G:8g)", "lacteos"),
    ],
    "CENA": [
        _o("- Sopa de verduras ligera (~150 kcal, P:5g C:22g G:4g)"),
        _o("- Ensalada de espinacas y palta (~200 kcal, P:4g C:18g G:13g)"),
        _o("- Menestra de frejol ligera (~240 kcal, P:12g C:38g G:4g)"),
        _o("- Tostadas con hummus y pepino (~180 kcal, P:6g C:26g G:6g)", "gluten", "sesamo"),
        _o("- Tortilla vegetal de champiñones (~220 kcal, P:10g C:12g G:15g)", "huevo"),
    ],
    "MERIENDA": [
        _o("- Fruta de estación (~80 kcal, P:1g C:20g G:0g)"),
        _o("- Puñado de almendras y nueces (~160 kcal, P:5g C:8g G:13g)", "frutos_secos"),
        _o("- Tostada integral con palta (~120 kcal, P:3g C:15g G:6g)", "gluten"),
        _o("- Rodajas de manzana con canela (~70 kcal, P:1g C:17g G:0g)"),
        _o("- Taza de quinua cocida ligera (~130 kcal, P:4g C:24g G:2g)"),
    ],
}

# Opciones sin ningún alérgeno común: completan la lista cuando el filtro deja menos de 3.
SEGURAS: List[OpcionFallback] = [
    _o("- Sopa de verduras ligera (~150 kcal, P:5g C:22g G:4g)"),
    _o("- Ensalada fresca de lechuga, tomate y pepino (~60 kcal, P:2g C:10g G:1g)"),
    _o("- Ensalada de espinacas y palta (~200 kcal, P:4g C:18g G:13g)"),
]

MENSAJE_SIN_OPCION_SEGURA = (
    "No tengo una opción segura para recomendarte ahora con tus restricciones. "
    "Consulta con tu nutricionista para que te indique qué puedes comer."
)

# texto del perfil (normalizado, sin tildes) -> etiquetas que debe evitar
_REGLAS_ETIQUETAS: Tuple[Tuple[str, Set[str]], ...] = (
    (r"\b(?:frutos?\s+secos?|nueces|nuez|almendras?|mani|cacahuates?|pistachos?|avellanas?|castanas?)\b", {"frutos_secos"}),
    (r"\bhuevos?\b", {"huevo"}),
    (r"\b(?:gluten|celiac\w*|celiaq\w*|trigo)\b", {"gluten"}),
    (r"\b(?:lactosa|lacteos?|leche|queso|yogur\w*)\b", {"lacteos"}),
    (r"\b(?:mariscos?|camarones?|langostinos?|cangrejos?|calamar\w*|pulpo)\b", {"mariscos"}),
    (r"\bpescados?\b", {"pescado"}),
    (r"\b(?:soya|soja)\b", {"soya"}),
    (r"\b(?:sesamo|ajonjoli)\b", {"sesamo"}),
    (r"\b(?:diabet\w*|azucar\w*|dulces?|glucosa)\b", {"azucar"}),
    (r"\bvegan\w*\b", {"huevo", "lacteos", "carne", "pescado", "mariscos"}),
    (r"\bvegetarian\w*\b", {"carne", "pescado", "mariscos"}),
)
_RX_ALERGIA = re.compile(r"\b(?:alergi\w*|intoleran\w*|sensibilidad)\b")


def _etiquetas_de_texto(texto_norm: str) -> Set[str]:
    out: Set[str] = set()
    for patron, tags in _REGLAS_ETIQUETAS:
        if re.search(patron, texto_norm):
            out |= tags
    return out


def etiquetas_excluidas(condiciones: Optional[Iterable[str]], prohibidos: Optional[Iterable[str]] = None) -> Tuple[Set[str], bool]:
    """(etiquetas a evitar, hay una alergia/intolerancia que no se pudo interpretar)."""
    tags: Set[str] = set()
    ilegible = False
    for c in condiciones or []:
        cn = norm(c)
        t = _etiquetas_de_texto(cn)
        tags |= t
        if _RX_ALERGIA.search(cn) and not t:
            ilegible = True
    for p in prohibidos or []:
        tags |= _etiquetas_de_texto(norm(p))
    return tags, ilegible


def _viola_prohibidos(opcion: OpcionFallback, frases: List[str], tokens: Set[str]) -> bool:
    texto = norm(opcion.texto)
    if tokens & _palabras(opcion.texto):
        return True
    return any(f in texto or f.rstrip("s") in texto for f in frases)


def seleccionar_opciones(
    momento: str,
    condiciones: Optional[Iterable[str]] = None,
    prohibidos: Optional[Iterable[str]] = None,
    aleatorio: bool = True,
    n: int = 3,
) -> List[str]:
    """Hasta ``n`` opciones del momento compatibles con las restricciones. Lista vacía = no hay ninguna segura."""
    m = (momento or "ALMUERZO").upper().strip()
    if m not in OPCIONES:
        m = "ALMUERZO"
    condiciones, prohibidos = list(condiciones or []), [p for p in (prohibidos or []) if p]
    tags, ilegible = etiquetas_excluidas(condiciones, prohibidos)
    if ilegible:
        return []

    frases = [f for f in (norm(p).strip() for p in prohibidos) if len(f) >= 4]
    tokens = _tokens_de_prohibidos(prohibidos)

    def ok(o: OpcionFallback) -> bool:
        return not (o.etiquetas & tags) and not _viola_prohibidos(o, frases, tokens)

    propias = [o for o in OPCIONES[m] if ok(o)]
    n_propias = min(n, len(propias))
    elegidas = random.sample(propias, n_propias) if aleatorio else propias[:n_propias]
    for o in SEGURAS:
        if len(elegidas) >= n:
            break
        if ok(o) and o not in elegidas:
            elegidas.append(o)
    return [o.texto for o in elegidas]


def texto_fallback(
    momento: str,
    condiciones: Optional[Iterable[str]] = None,
    prohibidos: Optional[Iterable[str]] = None,
    aleatorio: bool = True,
) -> str:
    opciones = seleccionar_opciones(momento, condiciones, prohibidos, aleatorio)
    return "\n".join(opciones) if opciones else MENSAJE_SIN_OPCION_SEGURA
