"""
Constructor dinámico de platos: descompone un nombre de plato usando LLM,
crea los registros Plato + PlatoIngrediente y genera los pasos de preparación.

Activado desde asistente_registro_comida.py (Capa 1.5) cuando un plato no existe
en el catálogo local y la consulta parece un plato completo (≥2 palabras).
"""
from __future__ import annotations

import difflib
import json
import re
import unicodedata
from typing import List, Optional


def _sufijos_con_compat(a: str, b: str) -> bool:
    """Guard 'con X': rechaza el match si los modificadores después de 'con' no comparten
    ninguna palabra. Evita que 'tortilla de huevo con pan' → 'Tortilla de Huevo con Espinacas'."""
    def _suf(s: str) -> list[str]:
        idx = s.rfind(" con ")
        return s[idx + 5:].split() if idx >= 0 else []
    s1, s2 = _suf(a), _suf(b)
    if not s1 or not s2:
        return True
    return bool(set(s1) & set(s2))


# Verbos y frases de apertura que el LLM añade al inicio del nombre sugerido
# pero no forman parte del nombre real del plato.
_RE_PREFIJO_VERBAL = re.compile(
    r"^(?:"
    # Verbos de inicio: pueden ir seguidos de artículo + sustantivo temporal + "con"
    r"(?:comienza|empieza|inicia)\s+"
    r"(?:(?:la|el|los|las)\s+)?"
    r"(?:(?:semana|dia|día|mañana|tarde|noche)\s+(?:con\s+)?)?"
    r"(?:de\s+|con\s+)?"
    r"|prueba\s+(?:la\s+|el\s+|un\s+|una\s+)?"
    r"|toma\s+(?:un\s+|una\s+)?"
    r"|consume\s+(?:el\s+|la\s+)?"
    r"|te\s+recomiendo\s+(?:el\s+|la\s+|un\s+|una\s+)?"
    r"|disfruta\s+(?:de\s+)?(?:el\s+|la\s+|un\s+|una\s+)?"
    # Etiquetas de categoría que el LLM antepone al nombre real
    r"|(?:comida|plato|bebida|snack|postre|entrada|merienda|desayuno|almuerzo|cena)\s+"
    r")",
    re.IGNORECASE | re.UNICODE,
)


def _sanitizar_nombre_plato(nombre: str) -> str:
    """Elimina prefijos verbales/categoría que el LLM antepone al nombre real del plato.
    Ej: 'Comida ensalada de lechuga' → 'Ensalada de lechuga'
        'Comienza la de plátano con aceite de oliva' → 'Plátano con aceite de oliva'."""
    limpio = _RE_PREFIJO_VERBAL.sub("", (nombre or "").strip()).strip()
    return limpio if len(limpio) >= 3 else nombre.strip()

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models.plato import Plato, PlatoIngrediente
from app.core.logging_config import get_logger
from app.services.nutricional_result import validar_macros_atwater

logger = get_logger("plato_constructor")


# Tokens que indican claramente entradas no-alimentarias (metales, materiales, sustancias)
_NO_FOOD_TOKENS: frozenset[str] = frozenset({
    "hierro", "acero", "aluminio", "cobre", "plomo", "oro", "plata",
    "titanio", "zinc", "niquelado", "cemento", "ladrillo", "concreto",
    "yeso", "plastico", "vidrio", "tornillo", "tuerca", "alambre",
    "cable", "pintura", "veneno", "gasolina", "petroleo", "solvente",
    "detergente", "lejia", "cloro", "jabon",
})

# ─── Rangos calóricos por horario ────────────────────────────────────────────
# (min, max) en kcal totales del plato generado
_RANGOS_CALORICOS: dict = {
    "desayuno":    (300,  500),
    "almuerzo":    (600,  900),
    "cena":        (300,  550),
    "cualquiera":  (150, 1000),  # rango amplio para platos genéricos
    "snack":       ( 80,  300),
    "merienda":    ( 80,  300),
    # Cebiches y tiraditos: gramaje ligero (pescado 150-200g + acompañamiento)
    "cebiche":          (150,  450),
    "tiradito":         (150,  400),
    "causa ferreñafana":(400,  650),   # plato norteño compacto
    "arroz con pato":   (700, 1000),   # ración almuerzo — mínimo 700 kcal exigido
    "jalea":            (500,  900),   # fritura mixta con absorción de aceite
    # Platos adicionales con rango calórico definido
    "chaufa":           (350,  750),   # arroz chaufa (verduras ~400, con proteína ~600)
    "sopa":             (100,  400),   # sopas / caldos / chupe / crema
    "ensalada":         ( 80,  500),   # ensaladas (simple ~150, con pollo/palta ~400)
    "anticucho":        (200,  500),   # anticuchos (por porción ~ 3-4 pinchos)
    "mazamorra":        (150,  400),   # postre de maíz / mazamorra morada
    "picarones":        (250,  550),   # picarones con miel de chancaca
    "apanado":          (400,  900),   # X apanado: proteína + rebozado + aceite
    "milanesa":         (400,  900),   # milanesa: igual que apanado
}

# ─── Filtro de coherencia semántica ──────────────────────────────────────────
# Ingredientes que NO deben aparecer en platos con ciertos patrones de nombre
_CONFLICTOS_SEMANTICOS: list[tuple[set[str], frozenset[str]]] = [
    # Cebiches: sin palta, zanahoria, tomate, jengibre, salsas industriales NI aceites
    ({"cebiche", "ceviche"},
     frozenset({"palta", "aguacate", "zanahoria", "tomate", "jengibre", "ketchup",
                "salsa de tomate", "mayonesa", "crema de leche", "mostaza",
                "aceite de oliva", "aceite vegetal", "aceite", "mantequilla",
                "crema", "queso", "leche", "yogurt"})),
    # Panes y tostadas: sin papas, arroz ni pollo entero
    ({"tostada", "pan tostado", "sandwich", "sandw"},
     frozenset({"arroz blanco", "papa cocida", "papa sancochada"})),
    # Sopas/cremas: sin aceitunas ni embutidos fríos
    ({"sopa", "crema de", "caldo"},
     frozenset({"mayonesa", "jamonada", "aceitunas", "jamon"})),
    # Postres: sin ingredientes salados de fondo
    ({"torta", "queque", "bizcocho", "mousse", "flan"},
     frozenset({"ajo", "cebolla", "comino", "oregano"})),
    # Cebiches y tiraditos: solo pescado fresco — nunca cocido ni sancochado
    ({"cebiche", "ceviche", "tiradito"},
     frozenset({"palta", "aguacate", "zanahoria", "tomate",
                "pescado blanco cocido", "salmon cocido", "trucha cocida",
                "sancochado", "sancochada"})),
    # FASE 3.5 — Ceviches/tiraditos: sin componentes fritos en texto LLM.
    # Complementa _validar_coherencia_culinaria() que opera post-resolución BD.
    ({"cebiche", "ceviche", "tiradito"},
     frozenset({"frito", "rebozado", "empanizado", "apanado",
                "horneado", "a la plancha con aceite"})),
    # Sudado / aguadito / chilcano: pescado FRESCO o cocido — nunca frito ni apanado
    ({"sudado", "aguadito", "chilcano"},
     frozenset({"pescado blanco frito", "pescado frito", "apanado", "frito",
                "rebozado", "empanizado", "chicharron de pescado"})),
    # Platos al horno / parrilla: prohibir variantes sancochadas
    ({"al horno", "a la parrilla", "horneado", "parrillada"},
     frozenset({"sancochado", "sancochada", "hervido", "hervida"})),
    # Causa Ferreñafana: sin lácteos ni emulsiones (son de la variante limeña)
    ({"causa ferreñafana", "causa ferrenafana"},
     frozenset({"mayonesa", "queso fresco", "queso", "crema de leche",
                "leche evaporada", "mantequilla", "pescado blanco fresco",
                "pescado blanco cocido"})),
    # Jalea: requiere variantes fritas — sin pescado fresco/cocido sin aceite
    ({"jalea"},
     frozenset({"pescado blanco fresco", "pescado blanco cocido",
                "calamar crudo", "langostino crudo"})),
    # Chaufa: no lleva perejil, cilantro ni aceite de oliva — sabor asiático-peruano
    ({"chaufa", "arroz chaufa"},
     frozenset({"perejil", "cilantro", "albahaca", "aceite de oliva",
                "mantequilla", "crema de leche"})),
    # Cualquier "X apanado / empanizado / milanesa": SOLO proteína X + huevo + harina + aceite.
    # NUNCA verduras de acompañamiento — el rebozado es el único "extra" válido.
    ({"apanado", "apanada", "empanizado", "empanizada", "milanesa", "rebozado", "rebozada"},
     frozenset({"cebolla", "ajo", "perejil", "cilantro", "albahaca",
                "palta", "aguacate", "tomate", "zanahoria", "pimiento", "pepino",
                "lechuga", "espinaca", "brocoli", "brócoli", "champiñon", "champiñones",
                "arroz", "papa cocida", "papa sancochada", "fideos",
                "queso", "crema de leche", "leche evaporada", "yogurt", "mantequilla"})),
    # Cebiches/tiraditos de mariscos: sin lácteos ni vegetales cocidos calientes
    ({"cebiche de camaron", "cebiche de langostino", "cebiche mixto"},
     frozenset({"queso", "crema de leche", "papa cocida", "zanahoria cocida",
                "choclo cocido", "cancha tostada"})),
    # FASE 3.5 — Ensaladas: sin frituras industriales incompatibles
    # NO bloquear "pollo frito" (Ensalada César con pollo frito es válida).
    # Solo bloquear elementos que convierten la ensalada en algo diferente.
    ({"ensalada"},
     frozenset({"salchipapa", "papa rellena", "hot dog", "hamburguesa"})),
]


# ─── FASE 4.2: Matriz de incompatibilidades entre ingredientes ────────────────
# Pares (grupo_A, grupo_B): si el plato contiene al menos un ingrediente de
# grupo_A Y al menos uno de grupo_B → rechazar.
# REGLA DE ORO: conservador — solo bloquear combinaciones universalmente absurdas.
# NO bloquear: queso/crema/leche evaporada + pollo (ají de gallina), yogurt + pollo
# (marinado), queso + res (algunas preparaciones). Solo los casos imposibles.
_INCOMPATIBILIDADES_INGREDIENTES: list[tuple[frozenset[str], frozenset[str]]] = [
    # leche en polvo es incompatible con toda proteína animal salada.
    # Uso válido de leche en polvo: mazamorra, arroz con leche, bebidas, repostería.
    # NUNCA en ensaladas, saltados, cebiches, sopas o cualquier proteína salada.
    (
        frozenset({"leche en polvo", "leche descremada en polvo", "leche entera en polvo"}),
        frozenset({
            "pollo", "pechuga", "muslo", "pescado", "res", "cerdo", "chancho",
            "carne", "lomo", "bistec", "camaron", "langostino", "calamar",
            "pulpo", "atun", "salmon", "caballa", "lisa", "mero", "tollo",
            "pavo", "pato", "mariscos", "anchoveta", "bonito", "trucha",
        }),
    ),
    # yogurt con pescado/mariscos: no hay base culinaria en gastronomía peruana.
    # Excepción deliberadamente NO cubierta: yogurt + pollo (marinados válidos).
    (
        frozenset({"yogurt", "yogur"}),
        frozenset({
            "pescado", "camaron", "langostino", "calamar", "pulpo",
            "atun", "salmon", "caballa", "lisa", "mero", "tollo",
            "anchoveta", "bonito", "trucha", "salpreso",
        }),
    ),
    # pescado fresco (para cebiches/tiraditos) con lácteos base.
    # Segunda línea de defensa post-resolución: _CONFLICTOS_SEMANTICOS cubre el
    # texto raw del LLM, esta regla cubre los nombres reales de alimentos en BD.
    (
        frozenset({"pescado blanco fresco", "pescado fresco"}),
        frozenset({
            "leche", "leche evaporada", "leche fresca",
            "queso", "queso fresco", "yogurt", "yogur", "mantequilla",
            "leche en polvo",
        }),
    ),
]


def _validar_compatibilidad_ingredientes(
    resueltos: list[tuple],
) -> tuple[bool, str]:
    """
    FASE 4.2 — Valida compatibilidad culinaria entre ingredientes resueltos en BD.

    Detecta combinaciones culinariamente absurdas que pueden pasar otras validaciones
    (ej: pollo + leche en polvo, pescado + yogurt).
    Opera sobre nombres normalizados de alimentos RESUELTOS (nombres reales en BD).

    Reglas conservadoras — solo bloquea lo universalmente imposible.
    Casos válidos NO bloqueados: ají de gallina (pollo+crema), marinado (pollo+yogurt).

    Retorna (True, "") si compatible; (False, motivo) si hay incompatibilidad.
    """
    if not resueltos:
        return True, ""

    ings_norms = [_norm(alim.nombre) for alim, _ in resueltos]

    for grupo_a, grupo_b in _INCOMPATIBILIDADES_INGREDIENTES:
        match_a = [
            ing for ing in ings_norms
            if any(k in ing for k in grupo_a)
        ]
        if not match_a:
            continue
        match_b = [
            ing for ing in ings_norms
            if any(k in ing for k in grupo_b)
        ]
        if not match_b:
            continue
        return False, (
            f"incompatibilidad culinaria: '{match_a[0]}' con '{match_b[0]}'"
        )

    return True, ""


# ─── Plantillas semánticas por tipo de plato ─────────────────────────────────
# Define ingredientes base obligatorios (al menos 1 debe estar presente) y
# prohibidos (ninguno puede estar) para los tipos de plato más comunes.
# Los checks se aplican sobre nombres normalizados de alimentos ya RESUELTOS en BD,
# complementando el filtro _CONFLICTOS_SEMANTICOS (que opera sobre raw LLM text).
_PLANTILLAS_PLATOS: dict[str, dict] = {
    "ceviche_cebiche": {
        "keywords_plato": ["ceviche", "cebiche"],
        "ingredientes_base": [
            "pescado", "caballa", "lisa", "mero", "tollo", "toyo",
            "merluza", "cabrilla", "ojo de uva", "jurel", "bonito",
            "camaron", "langostino", "pulpo", "calamar",
        ],
        "prohibidos": ["queso", "crema", "leche", "mayonesa", "mantequilla", "yogurt"],
    },
    "arroz_con_pollo": {
        "keywords_plato": ["arroz con pollo"],
        "ingredientes_base": ["pollo", "pechuga", "muslo de pollo"],
        "prohibidos": ["pato", "pavo", "pescado"],
    },
    "arroz_con_pato": {
        "keywords_plato": ["arroz con pato"],
        "ingredientes_base": ["pato"],
        "prohibidos": ["pollo", "pavo", "pescado"],
    },
    "causa": {
        "keywords_plato": ["causa"],
        "ingredientes_base": ["papa"],
        "prohibidos": ["arroz"],
    },
    "lomo_saltado": {
        "keywords_plato": ["lomo saltado"],
        "ingredientes_base": ["lomo", "res", "carne"],
        "prohibidos": ["pollo", "pescado", "mariscos"],
    },
    "ensalada": {
        "keywords_plato": ["ensalada"],
        "ingredientes_base": [],
        "prohibidos": ["papa frita", "salchipapa"],
    },
}


# ─── CAMBIO 2: Ingredientes esenciales por tipo de plato ─────────────────────
# Define reglas OBLIGATORIAS y PROHIBIDAS para los tipos de plato más críticos.
# - obligatorios: al menos 1 ingrediente de la lista DEBE estar presente.
# - prohibidos:   ningún ingrediente de la lista puede estar presente.
# Opera sobre nombres normalizados de alimentos RESUELTOS en BD (post-resolución).
_PLATOS_ESENCIALES: dict[str, dict] = {
    "ceviche": {
        "obligatorios": ["pescado", "limon", "cebolla", "lisa", "caballa",
                         "mero", "tollo", "toyo", "merluza", "cabrilla",
                         "ojo de uva", "jurel", "camaron", "langostino", "pulpo"],
        # El ceviche DEBE tener: al menos un pescado/marisco Y limón Y cebolla
        "obligatorio_todos": [
            # grupo 1: proteína marina (al menos 1)
            ("pescado", "lisa", "caballa", "mero", "tollo", "toyo",
             "merluza", "cabrilla", "ojo de uva", "jurel",
             "camaron", "langostino", "pulpo", "calamar", "anchoveta", "bonito",
             "trucha", "salmon", "atun"),
            # grupo 2: ácido (al menos 1)
            ("limon", "lima", "citrico"),
            # grupo 3: cebolla
            ("cebolla",),
        ],
        "prohibidos": ["aceite", "mantequilla", "crema", "mayonesa", "leche",
                       "queso", "yogurt", "ketchup", "mostaza"],
    },
    "tiradito": {
        "obligatorio_todos": [
            ("pescado", "lisa", "caballa", "mero", "lenguado", "trucha",
             "tollo", "toyo", "merluza", "cabrilla", "ojo de uva",
             "salmon", "atun", "bonito"),
            ("limon", "lima"),
        ],
        "prohibidos": ["aceite", "mantequilla", "crema", "mayonesa"],
    },
    "tortilla": {
        "obligatorio_todos": [
            ("huevo",),
        ],
        "prohibidos": [],
    },
    "arroz con leche": {
        "obligatorio_todos": [
            ("arroz",),
            ("leche",),
        ],
        "prohibidos": ["carne", "pollo", "pescado"],
    },
    "chaufa": {
        # Arroz chaufa siempre lleva arroz y huevo como mínimo
        "obligatorio_todos": [
            ("arroz",),
            ("huevo",),
        ],
        "prohibidos": ["perejil", "cilantro", "aceite de oliva", "mantequilla"],
    },
    "aji de gallina": {
        # Plato bandera — debe llevar pollo, no pescado ni res
        "obligatorio_todos": [
            ("pollo", "pechuga", "muslo"),
        ],
        "prohibidos": ["pescado", "res", "cerdo", "chancho"],
    },
    "tacu tacu": {
        # Tacu tacu = arroz + frejoles mezclados y fritos
        "obligatorio_todos": [
            ("arroz",),
            ("frejol", "frejoles", "menestra"),
        ],
        "prohibidos": [],
    },
    "ensalada": {
        "obligatorio_todos": [],
        "prohibidos": ["salchipapa"],
    },
    # Regla general para CUALQUIER "X apanado/empanizado/milanesa"
    # La clave se busca como substring en el nombre normalizado del plato.
    "apanado": {
        "obligatorio_todos": [
            ("huevo",),
            ("harina", "pan rallado", "pan molido", "galleta molida"),
        ],
        "prohibidos": ["cebolla", "ajo", "perejil", "cilantro", "albahaca",
                       "tomate", "palta", "zanahoria", "lechuga", "espinaca",
                       "arroz", "fideos"],
    },
    "milanesa": {
        "obligatorio_todos": [
            ("huevo",),
            ("harina", "pan rallado", "pan molido"),
        ],
        "prohibidos": ["cebolla", "ajo", "perejil", "cilantro", "arroz", "fideos"],
    },
    "pollo apanado": {
        "obligatorio_todos": [
            ("pollo", "pechuga", "muslo"),
            ("huevo",),
        ],
        "prohibidos": ["pescado", "res", "cerdo", "cebolla", "ajo",
                       "perejil", "cilantro", "tomate", "palta", "zanahoria"],
    },
    "pescado apanado": {
        "obligatorio_todos": [
            ("pescado", "filete", "pescado blanco"),
            ("huevo",),
        ],
        "prohibidos": ["cebolla", "ajo", "perejil", "cilantro", "albahaca",
                       "tomate", "palta", "zanahoria"],
    },
}


def _validar_ingredientes_esenciales(
    nombre_norm: str,
    resueltos: list[tuple],
) -> tuple[bool, str]:
    """
    CAMBIO 2 — Validación de ingredientes esenciales por tipo de plato.

    Para cada tipo en _PLATOS_ESENCIALES cuyo keyword esté en el nombre del plato:
    - Verifica que TODOS los grupos de obligatorio_todos estén cubiertos.
    - Verifica que NINGÚN ingrediente prohibido esté presente.

    Opera sobre ingredientes ya RESUELTOS en BD (nombres reales).
    Retorna (True, "") si válido; (False, motivo) si falla.
    """
    if not resueltos:
        return False, "sin ingredientes resueltos"

    ings_norms = [_norm(alim.nombre) for alim, _ in resueltos]

    for tipo, reglas in _PLATOS_ESENCIALES.items():
        if tipo not in nombre_norm:
            continue

        # 1) Verificar grupos obligatorios (cada grupo debe tener al menos 1 match)
        for grupo in reglas.get("obligatorio_todos", []):
            if not any(
                any(req in ing_n for req in grupo)
                for ing_n in ings_norms
            ):
                return False, (
                    f"falta ingrediente esencial para '{tipo}' — "
                    f"se requiere al menos uno de: {', '.join(grupo[:4])}"
                )

        # 2) Verificar prohibidos (ninguno puede aparecer)
        for prohibido in reglas.get("prohibidos", []):
            if any(prohibido in ing_n for ing_n in ings_norms):
                return False, (
                    f"ingrediente prohibido '{prohibido}' en plato tipo '{tipo}'"
                )

    return True, ""


# ─── CAMBIO 3: Auditoría de platos existentes en BD ──────────────────────────

def auditar_platos_esenciales(session) -> list[dict]:
    """
    CAMBIO 3 — Detecta platos mal construidos en la BD según _PLATOS_ESENCIALES.

    Itera todos los platos, verifica ingredientes obligatorios y prohibidos.
    Retorna lista de {plato_id, nombre, faltantes, prohibidos_encontrados}.
    """
    from app.models.plato import Plato as _Plato

    resultados = []
    platos = session.query(_Plato).all()

    for p in platos:
        nombre_n = _norm(p.nombre or "")
        ingredientes = []
        try:
            for pi in (p.ingredientes or []):
                if pi.alimento:
                    ingredientes.append(_norm(pi.alimento.nombre or ""))
        except Exception:
            continue

        for tipo, reglas in _PLATOS_ESENCIALES.items():
            if tipo not in nombre_n:
                continue

            faltantes = []
            for grupo in reglas.get("obligatorio_todos", []):
                if not any(
                    any(req in ing for req in grupo)
                    for ing in ingredientes
                ):
                    faltantes.append(f"grupo({', '.join(grupo[:3])})")

            prohibidos_encontrados = [
                bad for bad in reglas.get("prohibidos", [])
                if any(bad in ing for ing in ingredientes)
            ]

            if faltantes or prohibidos_encontrados:
                logger.warning(
                    "[Auditoría] Plato id=%s '%s' — faltantes=%s  prohibidos=%s",
                    p.id, p.nombre, faltantes, prohibidos_encontrados,
                )
                resultados.append({
                    "plato_id": p.id,
                    "nombre": p.nombre,
                    "faltantes": faltantes,
                    "prohibidos": prohibidos_encontrados,
                    "ingredientes_actuales": ingredientes[:8],
                })

    return resultados

def validar_semantica_plato(
    nombre_plato: str,
    ingredientes_normalizados: list[str],
) -> tuple[bool, str]:
    """
    Valida coherencia semántica entre nombre del plato e ingredientes resueltos.

    Opera sobre nombres normalizados de alimentos ya resueltos en BD (post-resolución),
    complementando _filtrar_coherencia_semantica() que opera sobre texto raw del LLM.

    Retorna (True, "") si válido; (False, motivo) si alguna plantilla falla.
    """
    nombre_n = _norm(nombre_plato)
    ings_n = [_norm(i) for i in ingredientes_normalizados]

    for _tipo_key, plantilla in _PLANTILLAS_PLATOS.items():
        if not any(kw in nombre_n for kw in plantilla.get("keywords_plato", [])):
            continue

        # 1) Ingredientes prohibidos (ninguno debe estar presente)
        for prohibido in plantilla.get("prohibidos", []):
            proh_n = _norm(prohibido)
            if any(proh_n in ing for ing in ings_n):
                return False, (
                    f"ingrediente prohibido '{prohibido}' en plato '{nombre_plato}'"
                )

        # 2) Al menos un ingrediente base debe estar presente
        base_list = plantilla.get("ingredientes_base", [])
        if base_list:
            base_norms = [_norm(b) for b in base_list]
            if not any(
                any(b in ing or ing in b for b in base_norms)
                for ing in ings_n
            ):
                return False, (
                    f"falta ingrediente base en '{nombre_plato}' "
                    f"(esperado uno de: {', '.join(base_list[:4])})"
                )

    return True, ""


# ─── Verificación de proteína requerida por nombre de plato ──────────────────
# Pares (keywords_en_nombre_plato, keywords_en_alimento_resuelto).
# Si el nombre contiene algún keyword del primer tuple, al menos un ingrediente
# resuelto debe contener algún keyword del segundo.
# Complementa validar_semantica_plato() que solo cubre 6 tipos de _PLANTILLAS_PLATOS.
_PROTEINAS_REQUERIDAS: list[tuple[tuple[str, ...], tuple[str, ...]]] = [
    (("pollo",),                         ("pollo", "pechuga", "muslo")),
    # "gallina" en nombre → acepta gallina o pollo/pechuga (sustituto habitual en ají de gallina)
    (("gallina",),                       ("gallina", "pollo", "pechuga", "muslo")),
    (("pato",),                          ("pato",)),
    (("pavo", "pavita"),                 ("pavo", "pavita")),
    (("bistec", "carne de res", "lomo saltado"),
                                         ("lomo", "res", "carne", "vacuno")),
    (("cerdo", "chancho"),               ("cerdo", "chancho", "tocino")),
    (("cabrito",),                       ("cabrito", "cordero")),
    (("langostino",),                    ("langostino", "camaron")),
    (("pulpo",),                         ("pulpo",)),
    # ── Pescados específicos por nombre ──────────────────────────────────────
    # Cubren casos como "ensalada de atún", "sopa de salmón" donde el nombre
    # menciona el pescado ESPECÍFICO en lugar del genérico "pescado".
    # _COHERENCIA_NOMBRE_INGREDIENTES solo activa cuando "pescado" está en el nombre;
    # estas reglas cubren el gap cuando el nombre dice el pescado directamente.
    (("atun",),                          ("atun",)),
    (("salmon",),                        ("salmon",)),
    (("trucha",),                        ("trucha",)),
    (("caballa",),                       ("caballa",)),
    (("merluza",),                       ("merluza",)),
    (("tollo",),                         ("tollo", "toyo")),
    (("toyo",),                          ("toyo", "tollo")),
    (("mero",),                          ("mero",)),
    (("lisa",),                          ("lisa",)),
    (("camaron",),                       ("camaron", "langostino")),
    # "lisa" como token completo → pez, no adjetivo.
    # Se verifica con word-level token en _validar_consistencia_final().
    (("vacuno",),                        ("vacuno", "res", "carne", "lomo")),
    # ── Bases carbohidrato ───────────────────────────────────────────────────
    # Evita que "Tostada de ARROZ integral" tenga solo pan (bug plato 304).
    # Si el nombre dice "arroz", al menos un ingrediente debe contener "arroz".
    (("arroz",),                         ("arroz",)),
    (("quinua",),                        ("quinua",)),
    (("avena",),                         ("avena",)),
    (("lentejas", "lenteja"),            ("lenteja", "lentejas")),
    (("garbanzo",),                      ("garbanzo",)),
]


def _verificar_proteina_requerida(
    nombre_plato_norm: str,
    resueltos: list[tuple],
) -> tuple[bool, str]:
    """
    Verifica que proteínas nombradas en el plato estén en los ingredientes resueltos.
    Opera sobre nombres normalizados de alimentos ya resueltos en BD.
    """
    ings_norms = [_norm(alim.nombre) for alim, _ in resueltos]
    for kws_nombre, kws_alimento in _PROTEINAS_REQUERIDAS:
        if not any(kw in nombre_plato_norm for kw in kws_nombre):
            continue
        if not any(
            any(ka in ing_n for ka in kws_alimento)
            for ing_n in ings_norms
        ):
            return False, (
                f"proteína '{kws_nombre[0]}' requerida por nombre "
                f"no encontrada en ingredientes resueltos "
                f"(esperado: {', '.join(kws_alimento[:3])})"
            )
    return True, ""


# ─── CAMBIO 1: Validación estricta de ingrediente principal ──────────────────
# Verifica que el ingrediente proteico mencionado en el nombre del plato esté
# REALMENTE presente en los ingredientes resueltos desde la BD.
# Complementa _PROTEINAS_REQUERIDAS con reglas explícitas de cruce de proteínas
# que son los bugs más críticos en producción (pescado→pollo, etc.).
_COHERENCIA_NOMBRE_INGREDIENTES: list[tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]] = [
    # (keywords_en_nombre, keywords_requeridos_en_ing, keywords_prohibidos_en_ing)
    (("pescado",),
     ("pescado", "atun", "salmon", "trucha", "caballa", "lisa", "mero",
      "tollo", "anchoveta", "bonito", "merluza", "tilapia", "lenguado",
      "bacalao", "salpreso"),
     ("pollo", "pechuga", "pavo", "pato", "res", "lomo", "cerdo", "chancho")),
    (("pollo",),
     ("pollo", "pechuga", "muslo", "pollo entero"),
     ("pescado", "atun", "salmon", "pato", "pavo", "res", "lomo", "cerdo")),
    (("pato",),
     ("pato",),
     ("pollo", "pechuga", "pescado", "pavo", "res", "cerdo")),
    (("pavo", "pavita"),
     ("pavo", "pavita"),
     ("pollo", "pechuga", "pescado", "pato", "res", "cerdo")),
    (("huevo",),
     ("huevo",),
     ()),
    (("cerdo", "chancho"),
     ("cerdo", "chancho", "tocino", "chicharron"),
     ("pollo", "pescado", "pato", "res", "lomo")),
]


# ─── CAMBIO 2: Guard final de consistencia nombre↔ingredientes ───────────────
# Cubre keywords proteicos específicos que NO activan _COHERENCIA_NOMBRE_INGREDIENTES
# ni _PROTEINAS_REQUERIDAS porque requieren coincidencia como PALABRA COMPLETA
# (no substring), evitando falsos positivos:
#   - "lisa" ≠ "realista", "lisa" = el pez en "ceviche de lisa"
#   - "res" ≠ "fresco", "res" = carne bovina en "sopa de res"
#   - "atún" ya cubierto por _PROTEINAS_REQUERIDAS pero con doble check de cross-proteína
# Usa set de tokens del nombre (split por espacio) para comparación exacta.
_CONSISTENCIA_FINAL_REGLAS: list[tuple[frozenset, tuple, tuple]] = [
    # (keywords_como_palabra_en_nombre, requeridos_en_ings, prohibidos_en_ings)
    (frozenset({"lisa"}),
     ("lisa",),
     ("pollo", "pechuga", "res", "vacuno", "cerdo", "chancho")),
    (frozenset({"res", "vacuno"}),
     ("res", "vacuno", "lomo", "carne"),
     ("pollo", "pechuga", "pescado", "atun", "salmon", "pato", "cerdo")),
    (frozenset({"atun"}),
     ("atun",),
     ("pollo", "pechuga", "res", "vacuno", "cerdo", "pato")),
    (frozenset({"salmon"}),
     ("salmon",),
     ("pollo", "pechuga", "res", "vacuno", "cerdo", "pato")),
    (frozenset({"trucha"}),
     ("trucha",),
     ("pollo", "pechuga", "res", "vacuno", "cerdo", "pato")),
    (frozenset({"caballa"}),
     ("caballa",),
     ("pollo", "pechuga", "res", "vacuno", "cerdo", "pato")),
    (frozenset({"camaron"}),
     ("camaron", "langostino"),
     ("pollo", "pechuga", "res", "vacuno", "cerdo")),
]


def _validar_consistencia_final(
    nombre_norm: str,
    resueltos: list[tuple],
) -> tuple[bool, str]:
    """
    CAMBIO 2 (FASE 3.4) — Guard final de consistencia usando word-level tokens.

    A diferencia de _validar_coherencia_nombre_ingredientes() que usa substring
    matching sobre el nombre completo, esta función opera sobre el CONJUNTO DE
    PALABRAS del nombre, evitando falsos positivos:
      - "res" ∈ tokens("sopa de res")        → activa regla
      - "res" ∉ tokens("pescado fresco")     → no activa ("fresco" es otra palabra)
      - "lisa" ∈ tokens("ceviche de lisa")   → activa regla
      - "lisa" ∉ tokens("ensalada realista") → "realista" es otro token

    También verifica proteínas cruzadas para estos keywords específicos.
    Retorna (True, "") si válido; (False, motivo) si hay inconsistencia.
    """
    if not resueltos:
        return True, ""

    tokens_nombre: frozenset[str] = frozenset(nombre_norm.split())
    ings_norms = [_norm(alim.nombre) for alim, _ in resueltos]

    for kws_nombre, kws_requeridos, kws_prohibidos in _CONSISTENCIA_FINAL_REGLAS:
        # Solo activa si alguna keyword aparece como PALABRA COMPLETA en el nombre
        matched = kws_nombre & tokens_nombre
        if not matched:
            continue

        matched_kw = next(iter(matched))

        # Verificar que al menos 1 ingrediente aporte la proteína requerida
        if kws_requeridos:
            tiene = any(
                any(kr in ing_n for kr in kws_requeridos)
                for ing_n in ings_norms
            )
            if not tiene:
                return False, (
                    f"nombre contiene palabra '{matched_kw}' pero ningún ingrediente "
                    f"resuelto lo confirma (esperado: {', '.join(kws_requeridos[:3])})"
                )

        # Verificar que no haya proteína cruzada
        for prohibido in kws_prohibidos:
            if any(prohibido in ing_n for ing_n in ings_norms):
                return False, (
                    f"nombre='{matched_kw}' pero ingrediente '{prohibido}' es incompatible"
                )

    return True, ""


# ─── CAMBIO 1 (FASE 3.4 adaptado): Limpieza de nombre según resueltos ─────────
# En lugar de reemplazar el nombre con un genérico (UX inaceptable),
# se eliminan únicamente los sufijos "con X" donde X NO fue resuelto como
# ingrediente real. Preserva el nombre específico del plato.
#
# Ejemplos:
#   "Ceviche de Lisa con Aceite de Oliva" → aceite rechazado → "Ceviche de Lisa"
#   "Pollo al Horno con Arroz" → arroz resuelto → nombre intacto
#   "Tortilla de Huevo con Plátano" → plátano no resuelto → "Tortilla de Huevo"

def _limpiar_nombre_segun_resueltos(nombre_plato: str, resueltos: list[tuple]) -> str:
    """
    CAMBIO 1 (FASE 3.4) — Limpia sufijos 'con X' del nombre del plato cuando X
    no está entre los ingredientes realmente resueltos desde la BD.

    Algoritmo:
      1. Divide el nombre en [base] + [partes 'con X'].
      2. Para cada parte 'con X', tokeniza X y verifica si algún token de ≥4 chars
         coincide con algún token de los nombres de ingredientes resueltos.
      3. Solo conserva las partes 'con X' cuyos ingredientes están realmente resueltos.
      4. Reconstruye el nombre (sin las partes rechazadas).

    No modifica el nombre si no hay partes 'con'; nunca produce un nombre genérico.
    """
    if not resueltos:
        return nombre_plato

    partes = re.split(r'\s+con\s+', nombre_plato, flags=re.IGNORECASE)
    if len(partes) == 1:
        return nombre_plato  # no hay "con" — sin cambios

    base = partes[0].strip()

    # Índice de tokens de todos los ingredientes resueltos (palabras ≥4 chars)
    ings_tokens: set[str] = set()
    for alim, _ in resueltos:
        for tok in _norm(alim.nombre).split():
            if len(tok) >= 4:
                ings_tokens.add(tok)

    partes_validas: list[str] = []
    partes_rechazadas: list[str] = []

    for parte in partes[1:]:
        parte_n = _norm(parte)
        tokens_parte = [t for t in parte_n.split() if len(t) >= 4]
        # La parte es válida si al menos 1 token significativo está en ingredientes resueltos
        if any(tok in ings_tokens for tok in tokens_parte):
            partes_validas.append(parte.strip())
        else:
            partes_rechazadas.append(parte.strip())

    if partes_rechazadas:
        logger.debug(
            "Nombre '%s': eliminados sufijos no resueltos → %s",
            nombre_plato, partes_rechazadas,
        )

    if partes_validas:
        return base + " con " + " con ".join(partes_validas)
    return base


# ─── FASE 3.4b: Validación de ingredientes explícitos en el nombre ────────────
# Verifica que los ingredientes MENCIONADOS EN EL NOMBRE del plato existan
# realmente entre los ingredientes resueltos desde la BD.
#
# Problema que resuelve:
#   LLM genera "Ensalada de Plátano y Cebolla" pero los ingredientes resueltos
#   son plátano + yogurt → "cebolla" está en el nombre pero NO en los ings.
#   _PROTEINAS_REQUERIDAS y _CONSISTENCIA_FINAL_REGLAS solo cubren proteínas;
#   esta función cubre CUALQUIER ingrediente específico en el nombre.
#
# Estrategia:
#   1. Extrae tokens significativos (≥5 chars) después de "con", "y", "de" en el nombre.
#   2. Para cada token, verifica si algún ingrediente resuelto lo contiene.
#   3. Si >50% de los tokens del nombre no están en ings → rechazar.
#   El umbral del 50% evita falsos positivos cuando el nombre incluye descriptores
#   de preparación ("al horno", "a la plancha") que no son ingredientes.
#
# Palabras ignoradas (conectores, adjetivos, preparaciones):
_PALABRAS_IGNORADAS_NOMBRE = frozenset({
    # conectores y artículos
    "con", "sin", "del", "los", "las", "una", "unos", "unas",
    # descriptores de preparación (no son ingredientes)
    "horno", "plancha", "parrilla", "vapor", "frito", "frita", "cocido", "cocida", "asado", "asada",
    "horneado", "horneada",     # hornear — "arroz horneado"
    "apanado", "apanada",       # apanado en pan rallado — "pescado apanado"
    "empanizado", "empanizada", # empanizado — variante de apanado
    "rebozado", "rebozada",     # rebozado en harina/huevo
    "ahumado", "ahumada",       # ahumado (trucha ahumada, pollo ahumado)
    "gratinado", "gratinada",   # gratinado con queso
    "sancochado", "sancochada", # sancochado/hervido
    "caramelizado",             # cebollas caramelizadas
    "agridulce",                # pollo agridulce, cerdo agridulce (chifa)
    "crujiente",                # pollo crujiente, maíz crujiente
    "dorado", "dorada",         # pollo dorado, papa dorada
    "salteado", "salteada",     # salteado de verduras
    "ligera", "ligero", "saludable", "natural", "fresco", "fresca",
    "estilo", "tipo", "especial", "peruano", "peruana", "casero", "casera",
    "salsa", "estofado", "guiso", "sudado", "saltado",
    # tipos de plato — no son ingredientes en sí mismos
    "ensalada", "tostada", "tortilla", "sandwich", "sandwi",
    "ceviche", "cebiche", "tiradito", "causa", "crema", "sopa",
    "batido", "licuado", "smoothi",
    # categorías genéricas — los ingredientes concretos son los que importan
    "verduras", "frutas", "fruta",
    # sinónimos comunes de ingredientes registrados con otro nombre en BD
    "aguacate",     # → Palta
    # pasta/fideos: el ingrediente en BD se llama "pasta cocida", no "tallarines"
    "tallarines", "fideos", "espagueti", "fettuccine",
    # descriptores adicionales de cantidad/método
    "porcion", "porcion", "controlada", "rellena", "relleno",
    # Cortes de carne — el ingrediente en BD es la carne genérica (ej: "Cerdo Lomo Cocido"),
    # no el corte específico. El validador busca "cerdo/pollo/res" en los resueltos.
    "chuleta",     # chuleta de cerdo/res — corte, no ingrediente propio en BD
    "filete",      # filete de res/pollo — ídem
    "bistec",      # bistec de res — ídem
    "lomo",        # lomo fino/saltado — puede ser plato o corte; no verificar como ingrediente
    # términos regionales que no mapean a ingredientes individuales
    "canchita", "serrana", "serrano", "norteno", "criollo", "criolla",
    # Tipos de plato peruanos (no son ingredientes en BD)
    "chaufa",       # arroz chaufa / chifa peruano
    "anticucho",    # anticuchos de corazón / pollo (singular)
    "anticuchos",   # plural — forma más común en nombres de platos
    "mazamorra",    # mazamorra morada / de maíz
    "picarones",    # picarones de camote/zapallo
    "empanada",     # empanada de pollo / carne
    "empanadas",
    "alfajor",      # postre — alfajor de manjar blanco
    "alfajores",
    "pepian",       # seco a lo pepián
    "chicharron",   # chicharrón como tipo de plato (ej: "chicharrón de cerdo con mote")
    # Descriptores de color usados en nombres de platos, no en alimentos BD
    "morado",       # mazamorra morada, chicha morada → el color no es ingrediente
    "morada",
    # Nombres de animal en platos peruanos donde el LLM puede usar "pollo" en lugar de "gallina"
    # _PROTEINAS_REQUERIDAS ya verifica que haya pollo/pechuga si el nombre tiene "gallina"
    "gallina",
})


def _validar_ingredientes_en_nombre(
    nombre_norm: str,
    resueltos: list[tuple],
) -> tuple[bool, str]:
    """
    FASE 3.4b — Verifica que ingredientes MENCIONADOS EN EL NOMBRE existan
    en los ingredientes realmente resueltos desde la BD.

    Solo verifica tokens de ≥5 caracteres que aparezcan después de conectores
    (con, y, de) o al inicio del nombre, ignorando descriptores de preparación.

    Retorna (True, "") si válido; (False, motivo) si hay mención sin correspondencia.
    """
    if not resueltos or not nombre_norm:
        return True, ""

    # Índice de todos los tokens de ingredientes resueltos (≥4 chars)
    ings_tokens: set[str] = set()
    for alim, _ in resueltos:
        for tok in _norm(alim.nombre).split():
            if len(tok) >= 4:
                ings_tokens.add(tok)

    # Extraer tokens significativos del nombre (≥5 chars, no ignorados, no números)
    tokens_nombre = [
        t for t in nombre_norm.split()
        if len(t) >= 5
        and t not in _PALABRAS_IGNORADAS_NOMBRE
        and not t.isdigit()
    ]

    if len(tokens_nombre) < 2:
        return True, ""  # nombre muy corto o genérico — no verificar

    # Verificar cuántos tokens del nombre tienen correspondencia en ingredientes
    ausentes = [
        t for t in tokens_nombre
        if not any(t in ing_tok or ing_tok in t for ing_tok in ings_tokens)
    ]

    # Umbral: si ≥40% de tokens significativos del nombre no están en ings → rechazar.
    # 0.4 en lugar de 0.5: cierra el gap donde exactamente 50% ausentes pasaba el guard
    # (ej: "batido leche almendras frutas" con "almendras" y "batido" ausentes = 50%).
    if len(ausentes) >= len(tokens_nombre) * 0.4:
        return False, (
            f"ingrediente(s) del nombre sin correspondencia en resueltos: "
            f"{ausentes[:3]} — ings disponibles: "
            f"{[t for t in list(ings_tokens)[:6]]}"
        )

    return True, ""


# ═══════════════════════════════════════════════════════════════════════════
# FASE 3.5 — VALIDACIÓN CULINARIA Y COHERENCIA DE PREPARACIÓN
# ═══════════════════════════════════════════════════════════════════════════

# ─── Clasificación culinaria del plato ───────────────────────────────────────
# Mapea el nombre normalizado del plato a un tipo culinario.
# Se usa para aplicar reglas de coherencia específicas por categoría.
# Orden de evaluación: de más específico a más genérico.
_TIPOS_CULINARIOS_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("crudo_marino",    ("ceviche", "cebiche", "tiradito", "leche de tigre")),
    ("crudo_general",   ("sashimi", "carpaccio", "tartare", "tartár")),
    ("ensalada",        ("ensalada",)),
    ("sopa",            ("sopa", "caldo", "chupe", "aguadito", "crema de", "consomé")),
    ("guiso",           ("guiso", "estofado", "adobo", "pepián", "seco de")),
    ("cocido_caliente", ("saltado", "al horno", "horneado", "a la parrilla", "frito")),
    # ── Nuevos tipos para bloquear ingredientes imposibles ──────────────────
    # Tostada: base = pan. Puede llevar untables sólidos o frescos encima.
    # PROHIBIDO: leche líquida/polvo (pan mojado ≠ tostada), jugos, bebidas.
    ("tostada",         ("tostada", "pan tostado", "tostado de pan")),
    # Batido / bebida: base líquida (leche, agua, jugo). PROHIBIDO: panes, arroz, pastas.
    ("batido",          ("batido", "smoothie", "licuado", "jugo de", "bebida de")),
    # Snack sólido: frutas, frutos secos, yogur. PROHIBIDO: arroz, pasta, carne pesada.
    ("snack",           ("snack", "fruta", "colacion", "bocado")),
]


def _inferir_tipo_culinario(nombre_norm: str) -> str:
    """
    FASE 3.5 — Clasifica el tipo culinario del plato según el nombre normalizado.
    Retorna una categoría de la lista _TIPOS_CULINARIOS_KEYWORDS o 'mixto'.
    """
    for tipo, keywords in _TIPOS_CULINARIOS_KEYWORDS:
        if any(kw in nombre_norm for kw in keywords):
            return tipo
    return "mixto"


# ─── Reglas de coherencia culinaria por tipo ─────────────────────────────────
# Formato: (tipo_culinario, keywords_prohibidos_en_ings_resueltos, motivo)
# Opera sobre nombres NORMALIZADOS de alimentos ya resueltos desde la BD.
# Reglas CONSERVADORAS para evitar falsos positivos:
#   - "frito" solo se bloquea en platos crudos (crudo_marino) donde es claramente
#     incompatible. En ensaladas NO se bloquea ("Ensalada César con Pollo Frito" es válida).
#   - "horneado" y "cocido" solo en tipo crudo_marino.
#   - Para ensaladas solo se bloquean frituras industriales pesadas (papa frita, chicharron).
_REGLAS_COHERENCIA_CULINARIA: list[tuple[str, tuple[str, ...], str]] = [
    # Platos crudos marinos: ningún ingrediente puede ser cocido, frito u horneado
    ("crudo_marino",
     ("frito", "cocido", "sancochado", "horneado", "a la parrilla",
      "hervido", "asado", "rebozado", "empanizado"),
     "plato crudo marino no puede tener ingredientes cocidos/fritos"),
    # Ensaladas: sin frituras industriales que cambian la naturaleza del plato.
    ("ensalada",
     ("papa frita", "salchipapa", "chicharron de cerdo", "hot dog"),
     "ensalada con componentes de fritura industrial incompatibles"),
    # Sopas: sin emulsiones frías ni aliños que degradan la textura
    ("sopa",
     ("mayonesa", "crema agria"),
     "sopa con emulsiones frías incompatibles"),
    # ── NUEVAS REGLAS ────────────────────────────────────────────────────────
    # Tostada: no puede tener leche líquida/polvo — convertiría el pan en
    # algo mojado, perdiendo la textura crujiente que define una tostada.
    # Tampoco arroz ni pasta — son acompañamientos incompatibles con tostada.
    ("tostada",
     ("leche en polvo", "leche descremada polvo", "leche evaporada",
      "leche fresca", "leche entera", "leche",
      "arroz blanco", "arroz cocido", "pasta cocida", "fideos"),
     "tostada no puede contener leche líquida/polvo ni carbohidratos pesados"),
    # Batido: no puede tener bases sólidas/carbohidratos no líquidos.
    # Un batido es una bebida — arroz, pan o pasta no se licúan en un batido peruano.
    ("batido",
     ("pan integral", "pan tostado", "arroz", "pasta", "fideos",
      "papa", "yuca", "camote"),
     "batido/licuado no puede contener bases sólidas como pan, arroz o pasta"),
]


def _validar_coherencia_culinaria(
    nombre_norm: str,
    resueltos: list[tuple],
) -> tuple[bool, str]:
    """
    FASE 3.5 — Valida que los ingredientes RESUELTOS (BD) sean coherentes
    con el tipo culinario del plato.

    Estrategia conservadora:
    - Solo aplica reglas donde la incompatibilidad es OBVIA e INCUESTIONABLE.
    - Platos "mixto" pasan sin restricción — no hay reglas para casos ambiguos.
    - Usa ingredientes reales de BD (post-resolución), no texto raw del LLM.

    Retorna (True, "") si coherente; (False, motivo) si hay incompatibilidad clara.
    """
    if not resueltos:
        return True, ""

    tipo = _inferir_tipo_culinario(nombre_norm)
    if tipo == "mixto":
        return True, ""  # sin reglas para platos genéricos — no bloquear

    ings_norms = [_norm(alim.nombre) for alim, _ in resueltos]

    for tipo_regla, prohibidos, motivo in _REGLAS_COHERENCIA_CULINARIA:
        if tipo != tipo_regla:
            continue
        for prohibido in prohibidos:
            conflicto = next(
                (ing_n for ing_n in ings_norms if prohibido in ing_n), None
            )
            if conflicto:
                return False, (
                    f"{motivo} — ingrediente conflictivo: '{conflicto}'"
                )

    return True, ""


# ─── Validación de preparación vs tipo culinario (LOG-ONLY) ──────────────────
# No bloquea la creación del plato. Solo registra si los pasos de preparación
# generados por el LLM contienen verbos culinariamente incompatibles con el tipo.
# Esto detecta bugs en el LLM de preparación sin interrumpir el flujo.

_VERBOS_ESPERADOS_POR_TIPO: dict[str, set[str]] = {
    "crudo_marino":    {"mezclar", "marinar", "macerar", "exprimir", "agregar", "revolver"},
    "sopa":            {"hervir", "cocinar", "agregar", "calentar", "colar", "sofreir"},
    "guiso":           {"saltear", "freir", "guisar", "sofreir", "cocinar", "agregar"},
    "cocido_caliente": {"saltear", "freir", "hornear", "hervir", "cocinar", "calentar"},
}

_VERBOS_INCOMPATIBLES_POR_TIPO: dict[str, set[str]] = {
    "crudo_marino":  {"freir", "hornear", "hervir", "saltear", "cocinar a fuego"},
    "crudo_general": {"freir", "hornear", "hervir", "saltear"},
}


def _validar_preparacion_vs_tipo(nombre_norm: str, preparacion: list[str]) -> None:
    """
    FASE 3.5 CAMBIO 4 — Log-only: verifica que los pasos de preparación
    sean coherentes con el tipo culinario del plato.

    No bloquea la persistencia. Solo genera warnings para detectar bugs
    en el LLM de preparación (ej. ceviche con pasos de cocción).
    """
    if not preparacion:
        return

    tipo = _inferir_tipo_culinario(nombre_norm)
    if tipo == "mixto":
        return

    texto_prep = " ".join(str(p) for p in preparacion).lower()

    # Verificar verbos incompatibles (más crítico — siempre loguear)
    incompatibles = _VERBOS_INCOMPATIBLES_POR_TIPO.get(tipo, set())
    for verbo in incompatibles:
        if verbo in texto_prep:
            logger.warning(
                "[prep_culinaria] '%s' (tipo=%s): verbo incompatible '%s' en preparación — "
                "revisar LLM de preparacion",
                nombre_norm, tipo, verbo,
            )
            return  # una advertencia es suficiente

    # Verificar verbos esperados (suave — solo debug si no hay ninguno)
    esperados = _VERBOS_ESPERADOS_POR_TIPO.get(tipo, set())
    if esperados and not any(v in texto_prep for v in esperados):
        logger.debug(
            "[prep_culinaria] '%s' (tipo=%s): preparación sin verbos esperados %s",
            nombre_norm, tipo, sorted(esperados),
        )


def _token_en_texto(token: str, texto: str) -> bool:
    """Verifica que `token` aparezca como palabra completa en `texto` (no substring)."""
    return bool(re.search(
        r'(?<![a-záéíóúüñ])' + re.escape(token) + r'(?![a-záéíóúüñ])',
        texto,
    ))


def _validar_coherencia_nombre_ingredientes(
    nombre_norm: str,
    resueltos: list[tuple],
) -> tuple[bool, str]:
    """
    CAMBIO 1 — Validación estricta de ingrediente principal.

    Verifica que la proteína declarada en el nombre del plato esté
    EFECTIVAMENTE presente en los ingredientes resueltos y que no
    haya proteínas cruzadas (ej. nombre=pescado pero ing=pollo).

    Retorna (True, "") si válido; (False, motivo) si hay incoherencia.
    """
    if not resueltos:
        return False, "sin ingredientes resueltos"

    ings_norms = [_norm(alim.nombre) for alim, _ in resueltos]

    for kws_nombre, kws_requeridos, kws_prohibidos in _COHERENCIA_NOMBRE_INGREDIENTES:
        if not any(kw in nombre_norm for kw in kws_nombre):
            continue

        # Si el nombre menciona esta proteína, debe estar en los ingredientes
        if kws_requeridos:
            tiene_requerido = any(
                any(kr in ing_n for kr in kws_requeridos)
                for ing_n in ings_norms
            )
            if not tiene_requerido:
                return False, (
                    f"nombre contiene '{kws_nombre[0]}' pero ningún ingrediente "
                    f"resuelto lo confirma (ings: {', '.join(ings_norms[:4])})"
                )

        # No debe haber proteínas contradictorias.
        # Usar word-boundary (_token_en_texto) para evitar falsos positivos:
        # "res" no debe detectarse en "fresco", "tomate fresco", "perejil", etc.
        if kws_prohibidos:
            for prohibido in kws_prohibidos:
                if any(_token_en_texto(prohibido, ing_n) for ing_n in ings_norms):
                    return False, (
                        f"nombre='{kws_nombre[0]}' pero ingrediente '{prohibido}' "
                        f"es incompatible — proteínas cruzadas"
                    )

    return True, ""


def _redondear_a_5g(gramos: float) -> int:
    """Redondea un gramaje al múltiplo de 5g más cercano, mínimo 5g."""
    return max(5, round(gramos / 5) * 5)


def _calcular_kcal_resueltos(resueltos: list[tuple]) -> float:
    """Calcula kcal totales de la lista [(Alimento, gramos)]."""
    total = 0.0
    for alim, gramos in resueltos:
        kcal_100g = float(getattr(alim, "calorias_100g", 0) or 0)
        total += kcal_100g * gramos / 100.0
    return total


def _autocorregir_gramajes(
    resueltos: list[tuple],
    tipo_plato: str,
) -> list[tuple]:
    """
    Si las kcal totales están fuera del rango para tipo_plato,
    escala todos los gramajes proporcionalmente para ajustar al límite.
    Además aplica hard cap de 500g por ingrediente individual — ninguna
    porción realista supera esa cantidad en un solo ingrediente.
    Retorna la lista corregida.
    """
    # Hard cap por ingrediente: máx 500g individual
    # (Lechuga 480g, Arroz 500g, etc. son porciones absurdas para 1 persona)
    _MAX_GRAMOS_ING = 500.0
    capped = []
    for alim, gramos in resueltos:
        if gramos > _MAX_GRAMOS_ING:
            logger.warning(
                "Hard cap gramaje: '%s' %.0fg → %.0fg (máx por ingrediente)",
                alim.nombre, gramos, _MAX_GRAMOS_ING,
            )
            gramos = _MAX_GRAMOS_ING
        capped.append((alim, gramos))
    resueltos = capped

    min_kcal, max_kcal = _RANGOS_CALORICOS.get(tipo_plato, (150, 1000))
    kcal_actual = _calcular_kcal_resueltos(resueltos)

    if kcal_actual <= 0:
        return resueltos

    if kcal_actual > max_kcal:
        factor = max_kcal / kcal_actual
        corregidos = [(alim, _redondear_a_5g(gramos * factor)) for alim, gramos in resueltos]
        logger.info("Autocorrección gramajes: %.0f→%.0f kcal (límite %s)", kcal_actual, _calcular_kcal_resueltos(corregidos), max_kcal)
        return corregidos

    if kcal_actual < min_kcal and len(resueltos) > 0:
        factor = min_kcal / kcal_actual
        # Solo escalar si el factor no es extremo (máx ×2.5) para no crear porciones absurdas
        if factor <= 2.5:
            corregidos = [(alim, _redondear_a_5g(gramos * factor)) for alim, gramos in resueltos]
            logger.info("Autocorrección gramajes: %.0f→%.0f kcal (mínimo %s)", kcal_actual, _calcular_kcal_resueltos(corregidos), min_kcal)
            return corregidos

    return resueltos


def _filtrar_coherencia_semantica(
    nombre_plato: str,
    ingredientes_raw: list[dict],
) -> list[dict]:
    """
    Elimina ingredientes que choquen semánticamente con el nombre del plato.
    Devuelve la lista filtrada; imprime advertencia por cada ingrediente removido.
    """
    nombre_lower = nombre_plato.lower()
    filtrados = list(ingredientes_raw)

    for patrones_nombre, ingredientes_prohibidos in _CONFLICTOS_SEMANTICOS:
        if not any(p in nombre_lower for p in patrones_nombre):
            continue
        antes = len(filtrados)
        filtrados = [
            ing for ing in filtrados
            if not any(
                prohibido in ing.get("nombre_es", "").lower()
                for prohibido in ingredientes_prohibidos
            )
        ]
        eliminados = antes - len(filtrados)
        if eliminados:
            logger.info("Coherencia semántica: %d ingrediente(s) incompatibles eliminados de '%s'", eliminados, nombre_plato)

    return filtrados


# ─── Detección de ingrediente principal ─────────────────────────────────────

_PLATO_STOPWORDS: frozenset[str] = frozenset({
    "con", "de", "en", "al", "a", "la", "el", "del", "las", "los",
    "y", "e", "sin", "para", "sobre", "tipo", "estilo",
})


def _detectar_ingrediente_principal(
    nombre_plato_norm: str,
    ingredientes_raw: list[dict],
) -> Optional[str]:
    """
    Retorna el nombre del ingrediente que mayor overlap semántico tiene con el
    nombre del plato (tokens ≥4 chars, sin stopwords).

    Usado en crear_plato_dinamico() para detectar omisión silenciosa de la
    proteína principal y abortar la construcción de forma explícita.

    Returns:
        nombre_es del ingrediente principal, o None si no hay overlap claro.
    """
    palabras = [
        w for w in nombre_plato_norm.split()
        if w not in _PLATO_STOPWORDS and len(w) >= 4
    ]
    if not palabras:
        return None
    for item in ingredientes_raw:
        ing_norm = _norm(item.get("nombre_es", ""))
        if any(p in ing_norm for p in palabras):
            return item["nombre_es"]
    return None


# ─── Normalización interna ───────────────────────────────────────────────────

def _norm(texto: str) -> str:
    s = (texto or "").strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9\s_-]", " ", s)
    return re.sub(r"\s{2,}", " ", s).strip()


def _initcap(texto: str) -> str:
    return texto.strip().capitalize()


# ─── Helpers LLM ─────────────────────────────────────────────────────────────

async def _descomponer_plato_llm(nombre_plato: str) -> List[dict]:
    """
    Pide a Groq que descomponga el plato en ingredientes con gramos.
    Retorna lista de {nombre_es, gramos} o lista vacía si falla.
    """
    try:
        from app.services.ia_service import ia_engine
        prompt = (
            f"Eres chef nutricionista peruano. Descompón el plato \"{nombre_plato}\" "
            f"en sus ingredientes principales.\n"
            f"IMPORTANTE: Si el nombre NO corresponde a un plato o alimento REAL que existe "
            f"en la gastronomía (ej. ingredientes ficticios, mitológicos o imaginarios), "
            f"responde exactamente: []\n"
            f"Responde SOLO JSON válido (array):\n"
            f'[{{"nombre_es":"<ingrediente>","gramos":<entero>}},...]\n'
            f"Reglas OBLIGATORIAS:\n"
            f"- Número de ingredientes según el plato:\n"
            f"  * Platos simples (ensalada, snack, fruta, yogur, batido): EXACTAMENTE 2-4 ingredientes PRINCIPALES.\n"
            f"    NO agregues aceite, sal, especias, ajo ni cebolla como relleno en platos simples.\n"
            f"  * Platos de fondo (sopa, guiso, segundo, saltado, arroz): 4-7 ingredientes.\n"
            f"  * Nunca superar 8 ingredientes en total.\n"
            f"- Coherencia ESTRICTA por tipo de plato:\n"
            f"  * TOSTADA: el ingrediente BASE es pan (integral o blanco). Encima solo van\n"
            f"    untables sólidos o frescos: queso, palta, tomate, jamón, atún, mermelada, miel.\n"
            f"    PROHIBIDO en tostadas: leche (líquida o en polvo), leche evaporada, arroz, pasta.\n"
            f"    Ejemplo CORRECTO: Tostada con Queso = [pan integral 70g, queso fresco 30g]\n"
            f"    Ejemplo INCORRECTO: Tostada con Leche en Polvo = [pan, leche en polvo] ← NUNCA\n"
            f"  * BATIDO / LICUADO / SMOOTHIE: es una bebida. Ingredientes = frutas + base líquida.\n"
            f"    PROHIBIDO en batidos: pan, arroz, pasta, papa, fideos.\n"
            f"    REGLA CRÍTICA: Si el nombre especifica el tipo de leche o líquido, úsalo EXACTAMENTE.\n"
            f"    Ejemplos: 'leche de almendras' → 'Leche de Almendras' (NO 'Leche Fresca');\n"
            f"    'leche de avena' → 'Leche de Avena'; 'leche de coco' → 'Leche de Coco';\n"
            f"    'bebida de soya' → 'Bebida De Soya'. NUNCA sustituir por leche animal.\n"
            f"    'leche de vaca' / 'leche entera' / 'leche fresca' → 'Leche Fresca Entera' (líquida, NUNCA en polvo).\n"
            f"    Ejemplo genérico: Batido de Plátano = [plátano 120g, leche fresca entera 200g]\n"
            f"  * ENSALADA: ingredientes frescos/crudos. PROHIBIDO: ingredientes sancochados pesados\n"
            f"    como papa, yuca, camote (a menos que el nombre lo especifique).\n"
            f"- Total gramos por tipo:\n"
            f"  * Platos completos (segundo, guiso, saltado, arroz): 400-750g total\n"
            f"  * Cebiche / tiradito: PESCADO 130-180g + acompañamiento (cebolla, limón, papa, ají) → total 350-500g\n"
            f"    NUNCA más de 200g de pescado en cebiche. Ejemplo: Merluza 150g, Cebolla 80g, Jugo de Limón 50g, Papa Cocida 80g, Ají 15g, Cilantro 5g\n"
            f"  * Snacks / batidos / bebidas: 100-300g total\n"
            f"- Nombres genéricos (sin marcas)\n"
            f"- SIEMPRE expresa cantidades en GRAMOS enteros, NUNCA en unidades genéricas\n"
            f"  Ejemplos de conversión: 1 huevo→55g, 1 plátano→120g, 1 cebolla→80g,\n"
            f"  1 tomate→100g, 1 papa→150g, 1 limón→50g, 1 palta→150g\n"
            f"- Gramajes razonables: queso fresco MAX 40g en tostada, aceite de oliva MAX 10g en ensaladas\n"
            f"- Usa el estado correcto del ingrediente según el método del plato:\n"
            f"  * Cebiche / tiradito / crudo → pescado FRESCO\n"
            f"  * Al horno / asado / parrilla → ingrediente crudo o 'Al Horno' (NO sancochado)\n"
            f"  * NUNCA uses 'Cocido' ni 'Sancochado' para preparaciones frías\n"
            f"- Si el nombre del plato especifica un pescado (ej. 'ceviche de cabrilla', 'sudado de mero'),\n"
            f"  USA ESE PESCADO EXACTAMENTE. NUNCA lo reemplaces por otro.\n"
            f"- Si el plato es marino/Omega-3 y NO especifica pescado, usa en este orden:\n"
            f"  1. Caballa  2. Lisa  3. Mero  4. Tollo\n"
            f"  El Atún y el Salmón SOLO si el usuario los pide explícitamente.\n"
            f"- Causa Ferreñafana: proteína='Pescado Salpreso'. Acompañamiento: Papa+Camote+Huevo+Plátano+Cebolla/Ají.\n"
            f"- Arroz con Pato: proteína='Pato'. Base: arroz verde (culantro+ají+chicha). Mínimo 700 kcal.\n"
            f"- Jalea de mariscos: usa SIEMPRE variantes FRITAS — 'Pescado Blanco Frito', 'Calamar Frito'.\n"
            f"- Chaufa / Arroz Chaufa: ingredientes OBLIGATORIOS: Arroz Blanco Cocido (200g), Huevo (55g), "
            f"Sillao (15g), Jengibre (5g), Cebolla (80g), Ajo (10g), Aceite Vegetal (10g). "
            f"Verduras opcionales según variante: Zanahoria, Pimiento, Cebolla China. "
            f"NUNCA: perejil, cilantro, aceite de oliva.\n"
            f"- X Apanado / X Empanizado / Milanesa de X: ingredientes SIEMPRE = "
            f"proteína X (150-200g) + Huevo (55g) + Harina De Trigo Fortificada Con Hierro (30g) + Aceite Vegetal (15g). "
            f"Ejemplos: Pollo Apanado → Pechuga De Pollo Cocida 165g. "
            f"Pescado Apanado → Pescado Blanco Frito 200g (usar la variante 'Frito'). "
            f"Res/Milanesa → Carne De Res Magra Cocida 150g. "
            f"NUNCA agregar: cebolla, ajo, perejil, cilantro, palta, tomate, zanahoria, arroz, fideos."
        )
        resp = await ia_engine._llamar_groq(prompt=prompt, max_tokens=300, temp=0.05)
        resp = re.sub(r"```(?:json)?", "", resp).strip().strip("`")
        m = re.search(r"\[.*\]", resp, re.DOTALL)
        if not m:
            return []
        items = json.loads(m.group(0))
        parsed = [
            {"nombre_es": str(it.get("nombre_es", "")).strip(),
             "gramos": max(1, int(it.get("gramos", 50)))}
            for it in items
            if it.get("nombre_es") and str(it.get("nombre_es", "")).strip()
        ]

        # ── Post-procesado por tipo de plato ─────────────────────────────────
        # El LLM tiende a agregar ingredientes de relleno o incoherentes.
        # Filtramos por tipo de plato para mantener coherencia culinaria.
        _nombre_n = _norm(nombre_plato)

        # 1) Platos SIMPLES: eliminar rellenos genéricos (aceite, sal, especias, ajo)
        _PLATOS_SIMPLES_KEYWORDS = (
            "ensalada", "fruta", "yogur", "yogurt", "batido", "snack",
            "manzana", "platano", "naranja", "mandarina",
        )
        if any(kw in _nombre_n for kw in _PLATOS_SIMPLES_KEYWORDS):
            _RELLENOS_GENERICOS = frozenset({
                "aceite de oliva", "aceite vegetal", "aceite", "sal comun", "sal",
                "especias", "condimento", "pimienta", "oregano", "comino",
                "ajo", "cebolla",
            })
            _reales = [
                p for p in parsed
                if not any(r in _norm(p["nombre_es"]) for r in _RELLENOS_GENERICOS)
            ]
            if len(_reales) >= 2:
                if len(_reales) < len(parsed):
                    logger.debug(
                        "Plato simple '%s': eliminados %d ingredientes de relleno",
                        nombre_plato, len(parsed) - len(_reales),
                    )
                parsed = _reales

        # 2) TOSTADAS: el único líquido permitido es aceite de oliva (para untar).
        #    Leche líquida/en polvo es incompatible — pan mojado no es tostada.
        #    Arroz o pasta tampoco van encima de una tostada.
        if "tostada" in _nombre_n or "pan tostado" in _nombre_n:
            _PROHIBIDOS_TOSTADA = frozenset({
                "leche", "leche en polvo", "leche descremada", "leche fresca",
                "leche evaporada", "leche entera",
                "arroz", "pasta cocida", "fideos",
            })
            _antes = len(parsed)
            parsed = [
                p for p in parsed
                if not any(pr in _norm(p["nombre_es"]) for pr in _PROHIBIDOS_TOSTADA)
            ]
            if len(parsed) < _antes:
                logger.warning(
                    "Tostada '%s': eliminados %d ingredientes incompatibles (leche/arroz/pasta)",
                    nombre_plato, _antes - len(parsed),
                )

        # 3) BATIDOS / LICUADOS: no pueden contener panes, arroz ni pastas.
        if any(kw in _nombre_n for kw in ("batido", "licuado", "smoothie", "jugo de")):
            _PROHIBIDOS_BATIDO = frozenset({
                "pan integral", "pan tostado", "pan", "arroz", "pasta",
                "fideos", "papa", "yuca", "camote",
            })
            _antes = len(parsed)
            parsed = [
                p for p in parsed
                if not any(pr in _norm(p["nombre_es"]) for pr in _PROHIBIDOS_BATIDO)
            ]
            if len(parsed) < _antes:
                logger.warning(
                    "Batido '%s': eliminados %d ingredientes sólidos incompatibles",
                    nombre_plato, _antes - len(parsed),
                )

        return parsed
    except Exception as e:
        logger.error("Error descomponiendo '%s': %s", nombre_plato, e)
        return []


async def _generar_preparacion_llm(
    nombre_plato: str, nombres_ingredientes: List[str]
) -> Optional[list]:
    """
    Genera pasos de preparación como array JSON de strings.
    REGLA CRÍTICA: La preparación SOLO puede mencionar los ingredientes de la
    lista `nombres_ingredientes`. Nunca inventa ni sustituye ingredientes.
    Retorna lista de strings o None si falla.
    """
    try:
        from app.services.ia_service import ia_engine
        lista_numerada = "\n".join(f"  {i+1}. {ing}" for i, ing in enumerate(nombres_ingredientes))
        prompt = (
            f"Genera los pasos de preparación del plato \"{nombre_plato}\".\n"
            f"\n"
            f"LISTA DE INGREDIENTES DISPONIBLES (SOLO ESTOS, ninguno más):\n"
            f"{lista_numerada}\n"
            f"\n"
            f"REGLAS ABSOLUTAS — violarlas es un error crítico:\n"
            f"1. SOLO usa ingredientes de la lista anterior. PROHIBIDO mencionar cualquier\n"
            f"   ingrediente que NO esté en esa lista (ej: si no está 'espinaca', no la menciones).\n"
            f"2. TODOS los ingredientes de la lista deben aparecer en al menos un paso.\n"
            f"3. Cada paso debe mencionar al menos un ingrediente de la lista.\n"
            f"4. Usa el estado correcto del ingrediente: si dice 'Cocido', es cocido; si dice\n"
            f"   'Fresco', es crudo. NO cambies el estado del ingrediente.\n"
            f"5. La preparación debe ser coherente con el nombre del plato: {nombre_plato}.\n"
            f"\n"
            f"Responde SOLO un array JSON de 4 a 6 strings en español. Sin texto extra.\n"
            f'Ejemplo: ["Calienta el aceite en una sartén.", '
            f'"Bate los huevos con sal y pimienta.", '
            f'"Agrega las espinacas y cocina 2 minutos."]'
        )
        resp = await ia_engine._llamar_groq(prompt=prompt, max_tokens=600, temp=0.05)
        resp = unicodedata.normalize("NFC", resp)
        resp = re.sub(r"```(?:json)?", "", resp).strip().strip("`")
        m = re.search(r"\[.*\]", resp, re.DOTALL)
        if not m:
            return None
        pasos = json.loads(m.group(0))
        pasos_limpios = [str(p).strip() for p in pasos if str(p).strip()]

        # ── Guard post-generación: detectar si la preparación menciona ingredientes
        # que no estaban en la lista original (señal de alucinación del LLM).
        # Solo loguea — no bloquea, para no perder el plato por un falso positivo.
        _ings_norm_set: set[str] = set()
        for ing in nombres_ingredientes:
            for tok in _norm(ing).split():
                if len(tok) >= 4:
                    _ings_norm_set.add(tok)

        # Palabras genéricas que siempre son válidas en una preparación
        _PREP_STOPWORDS = frozenset({
            "agua", "sal", "calor", "fuego", "temperatura", "minutos",
            "plato", "tazon", "bowl", "sarten", "olla", "taza", "cuchara",
            "mezcla", "agrega", "cocina", "sirve", "coloca", "corta", "pela",
            "calienta", "hierve", "sofrie", "saltea", "hornea", "bate", "lava",
            "pica", "ralla", "escurre", "sazona", "prueba", "revuelve",
        })
        texto_prep = " ".join(pasos_limpios).lower()
        texto_prep_norm = _norm(texto_prep)
        tokens_prep = [
            t for t in texto_prep_norm.split()
            if len(t) >= 5 and t not in _PREP_STOPWORDS
        ]
        _tokens_sin_cobertura = [
            t for t in tokens_prep
            if not any(t in ing_tok or ing_tok in t for ing_tok in _ings_norm_set)
        ]
        if len(_tokens_sin_cobertura) > 3:
            logger.warning(
                "[prep_alucinacion] '%s': preparación menciona tokens sin cobertura en ings: %s",
                nombre_plato, _tokens_sin_cobertura[:5],
            )

        return pasos_limpios
    except Exception as e:
        logger.error("Error generando preparacion para '%s': %s", nombre_plato, e)
        return None


# ─── Trinidad Nutricional (logging, no bloqueante) ───────────────────────────

def _verificar_trinidad_nutricional(plato: Plato) -> None:
    """Log de integridad: confirma que macros = Σ(alimento.macro × gramos/100)."""
    try:
        macros = plato.calcular_macros()
        logger.info(
            "[Trinidad] Plato '%s' (id=%s): kcal=%s prot=%s carb=%s gras=%s",
            plato.nombre, plato.id,
            macros["calorias"], macros["proteinas_g"],
            macros["carbohidratos_g"], macros["grasas_g"],
        )
    except Exception as e:
        logger.error("[Trinidad] Error verificando plato id=%s: %s", getattr(plato, "id", "?"), e)


def _loguear_resultado_nutricional(
    plato,
    resueltos: list[tuple],
    tipo_plato: str,
    nombre_display: str,
) -> None:
    """Calcula y loguea ResultadoNutricional post-creación (no bloqueante)."""
    try:
        from app.services.nutricional_result import (
            ResultadoNutricional,
            confidence_score,
            validar_plato_nutricional,
        )
        fuentes = [getattr(alim, "fuente", None) or "bd" for alim, _ in resueltos]
        macros = plato.calcular_macros()
        advertencias = validar_plato_nutricional(
            nombre_display, macros["calorias"], macros["proteinas_g"], tipo_plato
        )
        confianza = confidence_score(fuentes)
        resultado = ResultadoNutricional(
            estado="ok" if not advertencias else "incompleto",
            kcal=macros["calorias"],
            proteina=macros["proteinas_g"],
            carbohidratos=macros["carbohidratos_g"],
            grasas=macros["grasas_g"],
            confianza=confianza,
            modo_resolucion="reconstruido",
            nombre_plato=nombre_display,
            advertencias=advertencias,
            plato_id=plato.id,
        )
        if advertencias:
            logger.warning("[ResultadoNutricional] %s: %s", nombre_display, advertencias)
        else:
            logger.info(
                "[ResultadoNutricional] %s → kcal=%.0f confianza=%.2f",
                nombre_display, resultado.kcal, resultado.confianza,
            )
    except Exception as e:
        logger.error("[ResultadoNutricional] Error: %s", e)


# ─── Sanitizador de nombres LLM ──────────────────────────────────────────────
# El LLM a veces devuelve "nombre_es" con porciones incrustadas:
#   "2 rebanadas  pan integral" → "pan integral"
#   "15g  miel"                → "miel"
#   "1 taza de avena"          → "avena"
# Esos nombres causan creación de alimentos con nombres raros (IDs 790/792/793/809).
_RE_PORCION_PREFIJA = re.compile(
    r"^\d+[\.,]?\d*\s*"
    r"(cucharaditas|cucharadita|cucharadas|cucharada|rebanadas|rebanada|"
    r"unidades|unidad|pizcas|pizca|ramitos|ramito|dientes|diente|"
    r"hojas|hoja|rodajas|rodaja|trozos|trozo|porciones|porcion|"
    r"vasos|vaso|tazas|taza|kg|gr|ml|g|l)?\s*(de\s+)?",
    re.IGNORECASE,
)


def _limpiar_nombre_ingrediente(nombre_es: str) -> str:
    """Elimina porciones/unidades del inicio del nombre_es generado por LLM."""
    limpio = _RE_PORCION_PREFIJA.sub("", (nombre_es or "").strip()).strip()
    return limpio if limpio else nombre_es


# ─── Constructor principal ───────────────────────────────────────────────────

async def crear_plato_dinamico(
    db: Session,
    nombre_plato: str,
    tipo_plato: str = "cualquiera",
) -> Optional[Plato]:
    """
    Construye un plato desconocido desde cero:
      Groq → ingredientes → resolver/crear en alimentos → INSERT plato+ingredientes+preparacion.

    Retorna el objeto Plato creado (con ingredientes cargados) o None si falla.
    """
    # FIREWALL: esta función solo crea Plato + PlatoIngrediente.
    # Llama a _buscar_o_crear_alimento_async() (solo alimentos).
    # NUNCA es llamada desde _buscar_o_crear_alimento_async() — no hay recursión.
    from app.services.asistente_nutricion import _buscar_o_crear_alimento_async
    import difflib

    nombre_plato = _sanitizar_nombre_plato(nombre_plato)
    nombre_norm = _norm(nombre_plato)
    if not nombre_norm or len(nombre_norm) < 3:
        return None

    # Auto-detectar tipo_plato → aplica rango calórico correcto
    if tipo_plato == "cualquiera":
        if any(kw in nombre_norm for kw in ("cebiche", "ceviche")):
            tipo_plato = "cebiche"
        elif "tiradito" in nombre_norm:
            tipo_plato = "tiradito"
        elif "ferrenafana" in nombre_norm or "ferreñafana" in nombre_norm:
            tipo_plato = "causa ferreñafana"
        elif "arroz con pato" in nombre_norm:
            tipo_plato = "arroz con pato"
        elif nombre_norm.startswith("jalea"):
            tipo_plato = "jalea"
        elif "chaufa" in nombre_norm:
            tipo_plato = "chaufa"
        elif any(kw in nombre_norm for kw in ("sopa", "caldo", "chupe", "aguadito")):
            tipo_plato = "sopa"
        elif "ensalada" in nombre_norm:
            tipo_plato = "ensalada"
        elif "mazamorra" in nombre_norm:
            tipo_plato = "mazamorra"
        elif "picarones" in nombre_norm:
            tipo_plato = "picarones"
        elif "anticucho" in nombre_norm:
            tipo_plato = "anticucho"
        elif any(kw in nombre_norm for kw in ("apanado", "apanada", "empanizado",
                                               "milanesa", "rebozado")):
            tipo_plato = "almuerzo"

    # Guardia no-alimento: rechazar si contiene tokens claramente no-alimentarios
    _tokens = set(nombre_norm.split())
    if _tokens & _NO_FOOD_TOKENS:
        _rechazados = _tokens & _NO_FOOD_TOKENS
        logger.warning("'%s' rechazado — tokens no-alimentarios: %s", nombre_plato, _rechazados)
        return None

    # REGLA 5: deduplicación fuzzy antes de INSERT.
    # Threshold 0.80 cubre variantes regionales ("norteño", "casero", "especial")
    # que el threshold anterior (0.85) dejaba pasar y generaban platos duplicados.
    # Busca en platos cuya primera palabra coincida (eficiencia) O en los 40 más recientes.
    _first_word = nombre_norm.split()[0]
    existing = (
        db.query(Plato)
        .filter(Plato.nombre_normalizado.like(f"{_first_word}%"))
        .limit(40)
        .all()
    )
    for p in existing:
        _pnn = p.nombre_normalizado or ""
        sim = difflib.SequenceMatcher(None, nombre_norm, _pnn).ratio()
        if sim >= 0.80 and _sufijos_con_compat(nombre_norm, _pnn):
            logger.info("Plato similar encontrado (REGLA 5): '%s' (sim=%.2f)", p.nombre, sim)
            return p

    # Descomponer ingredientes con LLM
    ingredientes_raw = await _descomponer_plato_llm(nombre_plato)
    if not ingredientes_raw:
        logger.warning("LLM no devolvió ingredientes para '%s'", nombre_plato)
        return None

    # Filtro de coherencia semántica (antes de resolver en BD)
    ingredientes_raw = _filtrar_coherencia_semantica(nombre_plato, ingredientes_raw)

    # Resolver o crear cada ingrediente en BD
    # Detectar ingrediente principal antes de resolver (para guard de omisión)
    _nombre_ing_principal = _detectar_ingrediente_principal(nombre_norm, ingredientes_raw)

    resueltos: list[tuple] = []  # [(Alimento, gramos)]
    for item in ingredientes_raw:
        nombre_ing_es = _limpiar_nombre_ingrediente(item["nombre_es"])
        gramos = item["gramos"]
        nombre_ing_norm = _norm(nombre_ing_es)
        alim = await _buscar_o_crear_alimento_async(db, nombre_ing_norm, nombre_ing_es)

        # Fallback de simplificación: si el nombre completo falla (ej. "Perejil Deshidratado"),
        # quitar el último descriptor y reintentar ("Perejil").
        # Solo 1 paso para no sobre-simplificar ("Ají Amarillo" no debe reducirse a "Ají").
        if not alim:
            _partes = nombre_ing_es.split()
            if len(_partes) >= 2:
                _nombre_simple = " ".join(_partes[:-1])
                _norm_simple   = _norm(_nombre_simple)
                if len(_norm_simple) >= 4:
                    alim = await _buscar_o_crear_alimento_async(db, _norm_simple, _nombre_simple)
                    if alim:
                        logger.info(
                            "Ingrediente '%s' → simplificado a '%s'",
                            nombre_ing_es, _nombre_simple,
                        )

        if alim:
            resueltos.append((alim, gramos))
        else:
            logger.warning("Ingrediente no resuelto: '%s'", nombre_ing_es)

    # Guard de omisión silenciosa: si el ingrediente principal no se resolvió,
    # el plato tendría macros incorrectos (ej. "Arroz con Pato" sin Pato → ~200 kcal).
    # Falla explícitamente en lugar de devolver un resultado engañoso.
    if _nombre_ing_principal:
        _principal_norm = _norm(_nombre_ing_principal)
        _resueltos_norms = {_norm(alim.nombre) for alim, _ in resueltos}
        if not any(_principal_norm in rn or rn in _principal_norm for rn in _resueltos_norms):
            logger.error(
                "Ingrediente principal '%s' no resuelto para '%s' — construcción abortada "
                "(evita devolver kcal incorrectas por omisión silenciosa)",
                _nombre_ing_principal, nombre_plato,
            )
            return None

    # CAMBIO 3 — Mínimo de ingredientes reales con gramaje significativo
    if len(resueltos) < 2:
        logger.warning("Insuficientes ingredientes resueltos (%d) para '%s'", len(resueltos), nombre_plato)
        return None

    if all(g < 10 for _, g in resueltos):
        logger.warning(
            "Plato '%s' rechazado: todos los ingredientes tienen gramaje traza (<10g): %s",
            nombre_plato, [(alim.nombre, g) for alim, g in resueltos],
        )
        return None

    # BLOQUE 2: Gramaje total mínimo — el prompt pide 400-750g pero sin gate duro
    # un LLM que devuelva ingredientes traza (5g×2) pasaría todos los checks.
    _peso_total = sum(g for _, g in resueltos)
    if _peso_total < 50:
        logger.warning(
            "Plato '%s' rechazado: peso total %.0fg < mínimo 50g",
            nombre_plato, _peso_total,
        )
        return None

    # FASE 4.2 — Compatibilidad entre ingredientes (antes de calcular macros).
    # Bloquea combinaciones culinariamente absurdas: pollo+leche en polvo, pescado+yogurt.
    _ok_compat, _motivo_compat = _validar_compatibilidad_ingredientes(resueltos)
    if not _ok_compat:
        logger.warning(
            "Plato '%s' rechazado por incompatibilidad de ingredientes: %s",
            nombre_plato, _motivo_compat,
        )
        return None

    # CAMBIO 1 — Validación estricta de coherencia nombre↔ingrediente principal.
    # CRÍTICO: evita persistir platos con proteína cruzada (pescado→pollo, etc.)
    _ok_coh, _motivo_coh = _validar_coherencia_nombre_ingredientes(nombre_norm, resueltos)
    if not _ok_coh:
        logger.warning(
            "Plato '%s' rechazado: incoherencia nombre↔ingredientes — %s",
            nombre_plato, _motivo_coh,
        )
        return None

    # BLOQUE 3a: Proteína requerida por nombre — cubre cualquier keyword proteico,
    # no solo los 6 tipos de _PLANTILLAS_PLATOS. Complementa validar_semantica_plato().
    _ok_prot, _motivo_prot = _verificar_proteina_requerida(nombre_norm, resueltos)
    if not _ok_prot:
        logger.warning(
            "Plato '%s' rechazado por proteína faltante: %s", nombre_plato, _motivo_prot
        )
        return None

    # BLOQUE 3b: Validación semántica de plantilla sobre ingredientes resueltos.
    # Opera post-resolución (nombres reales de alimentos en BD), a diferencia de
    # _filtrar_coherencia_semantica() que opera sobre texto raw del LLM.
    _ings_resueltos_nombres = [alim.nombre for alim, _ in resueltos]
    _ok_sem, _motivo_sem = validar_semantica_plato(nombre_plato, _ings_resueltos_nombres)
    if not _ok_sem:
        logger.warning(
            "Plato '%s' rechazado por semántica: %s", nombre_plato, _motivo_sem
        )
        return None

    # BLOQUE 3c — CAMBIO 2: Validación de ingredientes ESENCIALES por tipo.
    # Garantiza que ceviche tenga pescado+limón+cebolla y no lleve aceite, etc.
    # Más granular que _PLANTILLAS_PLATOS: verifica GRUPOS obligatorios independientes.
    _ok_esen, _motivo_esen = _validar_ingredientes_esenciales(nombre_norm, resueltos)
    if not _ok_esen:
        logger.warning(
            "Plato '%s' rechazado — ingredientes esenciales: %s",
            nombre_plato, _motivo_esen,
        )
        return None

    # BLOQUE 3d — CAMBIO 2 (FASE 3.4): Guard final de consistencia (word-level).
    # Cubre keywords proteicos que NO disparan _COHERENCIA_NOMBRE_INGREDIENTES:
    # "lisa" (pez), "res" (sin matchear "fresco"), "atún", "salmón", "camarón", etc.
    # Usa tokens de palabra completa para evitar falsos positivos por substring.
    _ok_cf, _motivo_cf = _validar_consistencia_final(nombre_norm, resueltos)
    if not _ok_cf:
        logger.warning(
            "Plato '%s' rechazado — consistencia final: %s",
            nombre_plato, _motivo_cf,
        )
        return None

    # BLOQUE 3e — FASE 3.5: Coherencia culinaria (tipo de plato vs ingredientes BD).
    # Verifica que los ingredientes RESUELTOS sean compatibles con el tipo culinario.
    # Reglas conservadoras: solo bloquea incompatibilidades OBVIAS (ceviche+cocido, etc.).
    _ok_cul, _motivo_cul = _validar_coherencia_culinaria(nombre_norm, resueltos)
    if not _ok_cul:
        logger.warning(
            "Plato '%s' rechazado — incoherencia culinaria: %s",
            nombre_plato, _motivo_cul,
        )
        return None

    # BLOQUE 3f — FASE 3.4b: Ingredientes mencionados en el nombre deben estar en resueltos.
    # Bloquea: "Ensalada de Plátano y Cebolla" con ings=plátano+yogurt (cebolla ausente).
    # Umbral del 50%: tolera descriptores de preparación en el nombre que no son ingredientes.
    _ok_ien, _motivo_ien = _validar_ingredientes_en_nombre(nombre_norm, resueltos)
    if not _ok_ien:
        logger.warning(
            "Plato '%s' rechazado — nombre menciona ingredientes no resueltos: %s",
            nombre_plato, _motivo_ien,
        )
        return None

    resueltos = _autocorregir_gramajes(resueltos, tipo_plato)

    # CAMBIO 2 — BLOQUEAR MACROS EN 0 (CRÍTICO)
    # Se calcula DESPUÉS de autocorregir gramajes para no rechazar platos que
    # simplemente tenían porciones fuera de rango y ya fueron corregidos.
    _kcal_total = _calcular_kcal_resueltos(resueltos)
    _prot_t = sum(float(getattr(a, "proteina_100g", 0) or 0) * g / 100 for a, g in resueltos)
    _carb_t = sum(float(getattr(a, "carbohidratos_100g", 0) or 0) * g / 100 for a, g in resueltos)
    _gras_t = sum(float(getattr(a, "grasas_100g", 0) or 0) * g / 100 for a, g in resueltos)

    if _kcal_total <= 0:
        logger.warning(
            "Plato '%s' rechazado: kcal=0 o inválido — ingredientes sin datos nutricionales en BD",
            nombre_plato,
        )
        return None

    if (_prot_t + _carb_t + _gras_t) <= 0:
        logger.warning(
            "Plato '%s' rechazado: macros totales en 0 (prot+carb+gras=0)",
            nombre_plato,
        )
        return None

    # BLOQUE 5: Coherencia nutricional del plato completo (Atwater ±15%).
    # Reutiliza validar_macros_atwater() sobre la suma de macros de todos los ingredientes.
    # Nota: platos con vegetales INS/CENAN pueden tener desviaciones estructurales por fibra;
    # si se rechazan aquí, el caller deberá servir el plato desde BD o LLM sin persistir.
    _ok_atw, _motivo_atw = validar_macros_atwater(_kcal_total, _prot_t, _carb_t, _gras_t)
    if not _ok_atw:
        logger.warning(
            "Plato '%s' rechazado: %s", nombre_plato, _motivo_atw
        )
        return None

    # CAMBIO 1 (FASE 3.4) — Limpiar nombre: eliminar sufijos "con X" donde X no fue resuelto.
    # Evita que el nombre persista componentes rechazados (ej. "con Aceite de Oliva").
    # Se aplica DESPUÉS de todas las validaciones para operar sobre los resueltos definitivos.
    nombre_plato = _limpiar_nombre_segun_resueltos(nombre_plato, resueltos)
    nombre_norm = _norm(nombre_plato)

    # Crear el registro Plato + PlatoIngredientes en una sola transacción.
    # Ambos flushes están dentro del mismo BEGIN implícito; el único COMMIT ocurre
    # al final, garantizando que nunca exista un plato sin ingredientes en BD.
    nombre_display = _initcap(nombre_plato)
    try:
        plato = Plato(
            nombre=nombre_display[:255],
            nombre_normalizado=nombre_norm[:255],
            tipo_plato=tipo_plato,
            origen="llm",
        )
        db.add(plato)
        db.flush()  # obtener plato.id — aún dentro de la transacción, sin commit

        for orden, (alim, gramos) in enumerate(resueltos):
            db.add(PlatoIngrediente(
                plato_id=plato.id,
                alimento_id=alim.id,
                gramos=float(_redondear_a_5g(gramos)),
                orden=orden,
            ))

        db.flush()  # validar FK antes de llamar al LLM (costoso)

        nombres_ings = [alim.nombre for alim, _ in resueltos]
        preparacion = await _generar_preparacion_llm(nombre_plato, nombres_ings)
        if preparacion:
            plato.preparacion = preparacion
            # FASE 3.5 CAMBIO 4 — Log-only: verifica coherencia pasos vs tipo culinario.
            # No bloquea el plato; detecta bugs del LLM de preparación.
            _validar_preparacion_vs_tipo(nombre_norm, preparacion)

        db.commit()  # único commit — plato + ingredientes + preparacion atómicos
        db.refresh(plato)

        logger.info(
            "Plato '%s' creado (id=%s, %d ingredientes)",
            nombre_display, plato.id, len(resueltos),
        )
        _verificar_trinidad_nutricional(plato)
        _loguear_resultado_nutricional(plato, resueltos, tipo_plato, nombre_display)
        return plato

    except IntegrityError:
        db.rollback()
        existente = db.query(Plato).filter(
            Plato.nombre_normalizado == nombre_norm[:255]
        ).first()
        logger.info(
            "Race condition en plato '%s' — devolviendo existente id=%s",
            nombre_plato, existente.id if existente else None,
        )
        return existente

    except Exception as e:
        db.rollback()
        logger.error("Error creando plato '%s': %s", nombre_plato, e)
        return None
