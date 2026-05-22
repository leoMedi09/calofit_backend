"""
Registro de alimentos por NLP — Flujo de 5 capas.

CAPA 0: NLPFoodExtractor (Llama-3 JSON + cálculo determinista BD)
CAPA 1: Catálogo platos (plato_ingredientes × alimentos, macros en tiempo real)
CAPA 2-4: AlimentosDBService (alias → exact → USDA → FatSecret)
CAPA 5: Llama-3 estimación (último recurso, fuente=llm)
"""
from __future__ import annotations

import re
import difflib
import unicodedata
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.core.utils import get_peru_date
from app.models.alimento import Alimento
from app.models.alimento_unidad import AlimentoUnidad  # noqa: F401 (used in _get_porcion_estandar)
from app.models.historial import ProgresoCalorias

from app.core.logging_config import get_logger
from app.services.alimentos_db_service import AlimentosDBService, _norm as _norm_al
from app.services.asistente_nutricion import (
    advertencia_alimentos_prohibidos,
    registrar_preferencias_alimentos,
    verificar_conflicto_macros,
)
from app.services.nutricional_result import validar_macros_atwater
from app.services.trazabilidad import crear_comida_registros

logger = get_logger("registro_comida")

# ── Helpers de normalización (usados también en asistente_service) ───────────

# Rangos mínimos para platos nombrados conocidos.
# Si CAPA 0 estima por debajo del mínimo, cede el control a CAPA 1.5 (plato_constructor).
_CAPA0_RANGOS_MIN: dict[str, int] = {
    "arroz con pato":       700,
    "arroz con cabrito":    650,
    "seco de cabrito":      600,
    "seco de res":          600,
    "lomo saltado":         600,
    "aji de gallina":       550,
    "pollo a la brasa":     600,
    "causa ferreñafana":    400,
    "causa ferrenafana":    400,
    "jalea":                500,
    "caldo de gallina":     350,
    "sopa seca":            500,
    "tallarin saltado":     550,
    "sudado de pescado":    500,
    "arroz con leche":      300,
    "ceviche":              200,
    "tiradito":             200,
}


def _capa0_bajo_rango_plato(nombre_detectado: str, kcal: float) -> bool:
    """True si el nombre es un plato conocido y las kcal están por debajo del mínimo esperado."""
    n = (nombre_detectado or "").lower().strip()
    for patron, kcal_min in _CAPA0_RANGOS_MIN.items():
        if patron in n:
            return kcal < kcal_min
    return False


def _msg_tiene_porcion_lata(msg: str) -> bool:
    low = (msg or "").lower()
    if "lata" not in low and "latas" not in low:
        return False
    return bool(
        re.search(
            r"(?i)\b(\d+(?:[.,]\d+)?|media|medio|un|una|uno|dos|tres|cuatro|cinco)\s+lat\w*",
            low,
        )
    )


def _norm_plato(s: str) -> str:
    # Normaliza para comparar con nombre_normalizado en BD (= unaccent(lower(nombre))).
    # NFD descompone tildes; el filtro combining las elimina → "ají" → "aji".
    s = unicodedata.normalize("NFD", (s or "").strip().lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"\[.*?\]", "", s).split("[")[0].strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", s).strip()


def _sufijos_con_compatibles(a_norm: str, b_norm: str) -> bool:
    """Guard 'con X': verifica que los modificadores de ambos textos compartan
    al least una palabra. Evita que 'tortilla de huevo con pan' matchee
    'tortilla de huevo con atún' por alta similitud de la raíz común.
    Si alguno no tiene sufijo 'con', permite el match (no aplica el guard)."""
    def _sufijo(s: str) -> list[str]:
        idx = s.rfind(" con ")
        return s[idx + 5:].split() if idx >= 0 else []
    s1, s2 = _sufijo(a_norm), _sufijo(b_norm)
    if not s1 or not s2:
        return True          # sin sufijo → no hay conflicto
    return bool(set(s1) & set(s2))   # al menos 1 palabra en común


def _sufijos_de_compatibles(a_norm: str, b_norm: str) -> bool:
    """Guard 'de ESPECIE': bloquea falsos matches como 'ceviche de cabrilla' vs
    'ceviche de caballa' (similitud 0.92 pero peces distintos).
    Si ambos nombres tienen 'de X', la primera palabra de X debe coincidir."""
    m_a = re.search(r"\bde\s+(\w+)", a_norm)
    m_b = re.search(r"\bde\s+(\w+)", b_norm)
    if not m_a or not m_b:
        return True   # sin "de X" → no aplica guard
    return m_a.group(1) == m_b.group(1)


def _parse_qty(prefix: str) -> float:
    p = (prefix or "").strip().lower()
    if not p:
        return 1.0
    m = re.match(r"^\s*(\d+(?:[.,]\d+)?)\b", p)
    if m:
        return float(m.group(1).replace(",", "."))
    mapa = {"un": 1, "uno": 1, "una": 1, "dos": 2, "tres": 3,
            "cuatro": 4, "cinco": 5, "media": 0.5, "medio": 0.5}
    for k, v in mapa.items():
        if p.startswith(k):
            return float(v)
    return 1.0


# ── Modo Estándar — detección de incertidumbre ───────────────────────────────

_RE_NO_SE = re.compile(
    r"(?i)\b(no\s+s[eé]|no\s+tengo\s+(?:el\s+dato|idea|datos)|no\s+recuerdo|"
    r"no\s+me\s+acuerdo|no\s+estoy\s+seguro|aprox|aproximado|m[aá]s\s+o\s+menos|"
    r"ni\s+idea|no\s+s[eé]\s+cu[aá]nto|no\s+s[eé]\s+la\s+cantidad)\b"
)

# Porciones estándar por categoría (fallback cuando no hay alimento_unidades)
_PORCIONES_ESTANDAR: dict = {
    "cereal":     100,  # arroz, pasta, quinua cocidos
    "carne":       90,  # pollo, res, cerdo
    "vegetal":     80,  # verduras, ensaladas
    "fruta":      120,  # fruta entera
    "lácteo":     200,  # leche, yogurt (ml → g)
    "legumbre":    80,  # lentejas, frijoles cocidos
    "pan":         60,  # rebanada de pan
    "bebida":     240,  # vaso de bebida
    "default":    100,  # fallback universal
}


# ── Rangos calóricos por horario (para advertencias al registrar) ─────────────
RANGOS_HORARIO: dict = {
    "desayuno": (300, 500),
    "almuerzo": (600, 900),
    "cena":     (300, 550),
    "merienda": ( 80, 300),
    "snack":    ( 80, 300),
}

# Advertencia soft: por encima de este límite se emite aviso al usuario
_KCAL_MAX_REGISTRO = 1500

# Hard stop: por encima de estos límites el registro se BLOQUEA
# hasta que el usuario corrija la cantidad.
_KCAL_HARD_STOP = 2500          # kcal máximas aceptables por comida individual
_GRAMOS_HARD_STOP = 1000        # gramos máximos por ingrediente (excepto agua/líquidos)

# Alimentos cuyo alto gramaje es razonable (bebidas, sopas, caldos)
_ALIMENTOS_LIQUIDOS = frozenset({
    "agua", "caldo", "sopa", "jugo", "zumo", "leche", "refresco", "gaseosa",
    "limonada", "chicha", "maracuya", "emoliente", "te", "cafe", "infusion",
})

# Techo histórico (advertencia soft, no bloqueo)
_GRAMOS_MAX_POR_ALIMENTO = 2000


def _es_alimento_liquido(nombre: str) -> bool:
    # Comparación por tokens para evitar falsos positivos por substring:
    # "te" (bebida) no debe matchear con "mantequilla"
    tokens = set((nombre or "").lower().split())
    return bool(tokens & _ALIMENTOS_LIQUIDOS)


def _validar_hard_stop(extraccion: dict) -> Optional[str]:
    """
    Bloqueo de seguridad (Hard Stop): rechaza el registro si la extracción
    supera los límites físicos de coherencia.

    Retorna mensaje de error para mostrar al usuario, o None si es válido.
    """
    kcal = float(extraccion.get("calorias", 0) or 0)
    if kcal > _KCAL_HARD_STOP:
        return (
            f"No puedo registrar {kcal:.0f} kcal en una sola comida "
            f"(límite: {_KCAL_HARD_STOP} kcal). "
            f"Por favor verifica la cantidad y vuelve a indicarme cuánto comiste."
        )
    gramos = float(extraccion.get("porcion_g", 0) or 0)
    nombres = extraccion.get("alimentos_detectados", [])
    nombre_alim = nombres[0] if nombres else ""
    if gramos > _GRAMOS_HARD_STOP and not _es_alimento_liquido(nombre_alim):
        return (
            f"No puedo registrar {gramos:.0f}g de '{nombre_alim}' — "
            f"ese gramaje supera el límite de {_GRAMOS_HARD_STOP}g por ingrediente. "
            f"¿Quizás quisiste decir {gramos / 10:.0f}g? Corrígeme la cantidad."
        )
    return None


def _validar_gramaje_extraccion(extraccion: dict) -> Optional[str]:
    """Advertencia soft para gramajes inusuales que no llegan al Hard Stop."""
    gramos = float(extraccion.get("porcion_g", 0) or 0)
    if gramos > _GRAMOS_MAX_POR_ALIMENTO:
        return (
            f"⚠ Gramaje inusual: {gramos:.0f}g para un solo alimento "
            f"(máximo recomendado: {_GRAMOS_MAX_POR_ALIMENTO}g). "
            f"Registro marcado como Pendiente de Revisión."
        )
    return None


def _inferir_momento_dia_por_hora() -> Optional[str]:
    from app.core.utils import inferir_momento_dia_peru
    return inferir_momento_dia_peru()


def _inferir_momento_dia(mensaje: str) -> Optional[str]:
    """
    Infiere el momento del día. Prioridad: keywords del mensaje.
    Si el mensaje no tiene keywords, usa la hora del servidor como fallback.
    """
    m = (mensaje or "").lower()
    if any(w in m for w in ["desayun", "mañana", "breakfast"]):
        return "desayuno"
    if any(w in m for w in ["almorz", "almuerz", "almorzar", "mediodía", "mediodia"]):
        return "almuerzo"
    if any(w in m for w in ["cenar", "cené", "cene", "noche"]):
        return "cena"
    if any(w in m for w in ["merienda", "snack", "colación", "tarde"]):
        return "merienda"
    return _inferir_momento_dia_por_hora()  # fallback: reloj del servidor


def _advertencia_rango_horario(kcal: float, momento: Optional[str]) -> Optional[str]:
    """
    Devuelve un string de advertencia si las kcal del plato/alimento están fuera
    del rango esperado para ese momento del día. None si está dentro del rango.
    """
    if not momento or momento not in RANGOS_HORARIO:
        return None
    min_k, max_k = RANGOS_HORARIO[momento]
    if kcal > max_k:
        return (
            f"⚠ Esta comida supera el rango recomendado para {momento} "
            f"({min_k}-{max_k} kcal). Registrado de todas formas."
        )
    if kcal < min_k * 0.5:
        return (
            f"ℹ Esta comida está muy por debajo del rango habitual para {momento}. "
            f"¿Fue una porción pequeña?"
        )
    return None


def _es_respuesta_no_se(mensaje: str) -> bool:
    """True si el mensaje indica que el usuario no conoce la cantidad exacta."""
    return bool(_RE_NO_SE.search(mensaje or ""))


def _get_porcion_estandar(alimento_nombre: str, db: Session) -> tuple[float, str]:
    """
    Retorna (gramos_estandar, descripcion) para un alimento.
    Busca primero en alimento_unidades, luego usa fallback por categoría.
    """
    from app.models.alimento import Alimento
    alim = db.query(Alimento).filter(
        Alimento.nombre_normalizado.ilike(f"%{alimento_nombre[:30].lower()}%")
    ).first()

    if alim:
        from app.models.alimento_unidad import AlimentoUnidad
        unidad = db.query(AlimentoUnidad).filter(
            AlimentoUnidad.alimento_id == alim.id
        ).first()
        if unidad and unidad.gramos:
            return float(unidad.gramos), f"1 {unidad.nombre} (~{int(unidad.gramos)}g)"

        # Inferir categoría desde nombre normalizado
        cat_map = [
            ({"arroz", "pasta", "fideos", "quinua", "avena"}, "cereal"),
            ({"pollo", "res", "cerdo", "pescado", "atun", "carne"}, "carne"),
            ({"leche", "yogur", "queso"}, "lácteo"),
            ({"pan", "tostada"}, "pan"),
            ({"manzana", "plátano", "naranja", "pera", "fruta"}, "fruta"),
            ({"lenteja", "frijol", "garbanzo", "menestra"}, "legumbre"),
        ]
        alim_lower = (alim.nombre or alimento_nombre).lower()
        for keywords, cat in cat_map:
            if any(kw in alim_lower for kw in keywords):
                gramos = _PORCIONES_ESTANDAR[cat]
                return float(gramos), f"porción estándar (~{gramos}g)"

    gramos = _PORCIONES_ESTANDAR["default"]
    return float(gramos), f"porción estándar (~{gramos}g)"


# Regex: detecta prefijos de gramaje o recipiente que NO deben activar Capa 1.5.
# REGLA 2: incluir "medio plato de", "media porción de", "un poco de" para evitar
# que se creen platos falsos como "Plato De Arroz Con Pollo".
_CAPA15_SKIP_RE = re.compile(
    r'^\d+(?:[.,]\d+)?\s*(?:g|gr|kg|ml|l|litros?|vasos?|tazas?|copas?)\b'
    r'|^(?:un|una|medio|media)\s+(?:vaso|taza|copa|botella|lata|plato)\b'
    r'|^(?:vaso|taza|copa|botella|jarra|lata)\s+de\b'
    r'|^(?:un\s+poco\s+de|un\s+poquito\s+de|algo\s+de|medio\s+plato\s+de|media\s+porci[oó]n\s+de)\b',
    re.IGNORECASE,
)


def _es_candidato_plato_capa15(item: str) -> bool:
    """True si el item parece un plato complejo (no una cantidad ni recipiente simple)."""
    return len(item.split()) >= 2 and not _CAPA15_SKIP_RE.match(item)


_RE_PREFIJO_IMPERATIVO = re.compile(
    r"(?i)^\s*(?:"
    r"reg[ií]strame\s+(?:un[ao]?\s+)?|registra\s+(?:el|la|que)\s+|"
    r"anota\s+(?:el|la|que)\s+|an[oó]tame\s+(?:un[ao]?\s+)?|"
    r"guarda\s+(?:el|la|que)\s+|gu[aá]rdame\s+(?:un[ao]?\s+)?|"
    r"agr[eé]game\s+(?:un[ao]?\s+)?|agregame\s+(?:un[ao]?\s+)?|"
    r"ponme\s+(?:en\s+)?(?:el\s+)?|apunta\s+(?:el|la|que)\s+|"
    r"he\s+comido\s+|com[ií]\s+|me\s+com[ií]\s+|hoy\s+com[ií]\s+"
    r")\s*"
)
_RE_CANTIDAD_INICIO = re.compile(
    r"(?i)^\s*(?:un[ao]?\s+|unos\s+|unas\s+|medio\s+|media\s+|algo\s+de\s+|un\s+poco\s+de\s+)\s*"
)


def _split_items_from_message(msg: str) -> List[str]:
    t = (msg or "").lower().strip()
    # Primera pasada: verbos imperativos / de acción al inicio
    t = _RE_PREFIJO_IMPERATIVO.sub("", t).strip()
    # Segunda pasada: residuo "comí una/un" o artículos indefinidos
    t = _RE_PREFIJO_IMPERATIVO.sub("", t).strip()
    # Strip leading "de " antes de artículos: "comi de un X" → "de un X" → "un X" → "X"
    t = re.sub(r"(?i)^de\s+", "", t)
    t = _RE_CANTIDAD_INICIO.sub("", t).strip()
    t = re.sub(r"(?i)\b(otra\s+vez|de\s+nuevo|nuevamente)\b", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    # Separadores: "y", "más"/"mas", "además de", "junto con", coma, punto y coma
    _SEP = re.compile(
        r"\s+y\s+|\s+m[aá]s\s+(?:un\s+|una\s+)?|\s+adem[aá]s\s+de\s+|\s+junto\s+con\s+|,|;",
        re.IGNORECASE,
    )
    # Verbos de ingesta al inicio de sub-ítems (ej: "tome leche", "bebí agua")
    _RE_VERBO_SUBITEM = re.compile(
        r"^(?:tom[eé]|beb[íi]|com[íi]|almorc[eé]|cen[eé]|desayun[eé]|meriend[eé]|prob[eé])"
        r"\s+(?:un[ao]?\s+|medio\s+|media\s+)?",
        re.IGNORECASE,
    )
    partes = [p.strip() for p in _SEP.split(t) if p.strip()]
    return [_RE_VERBO_SUBITEM.sub("", p).strip() or p for p in partes]


def _expandir_compuestos_con(items: List[str], db: Session) -> List[str]:
    """
    Expande un item "plato_conocido + con + acompañamiento" en dos items separados.
    Solo actúa cuando el prefijo (antes del último " con ") existe en platos BD.
    Ejemplo: "arroz con pato con papas fritas" → ["arroz con pato", "papas fritas"]
    No toca "arroz con pato" (sin plato-prefijo conocido antes de "pato").
    """
    from sqlalchemy import text as _sql_t
    resultado: List[str] = []
    for item in items:
        if " con " not in item:
            resultado.append(item)
            continue
        # No segmentar si el nombre completo ya existe como plato propio en BD.
        # Usa exact match primero; si falla, fuzzy ≥ 0.88 para tolerar artículos
        # ("con la ensalada" vs "con ensalada") y tildes variables.
        norm_full = _norm_plato(item)
        _exact = db.execute(
            _sql_t("SELECT 1 FROM platos WHERE nombre_normalizado = :n LIMIT 1"),
            {"n": norm_full},
        ).fetchone()
        if _exact:
            resultado.append(item)
            continue
        # Fuzzy: buscar candidatos con la misma primera palabra, comparar similitud
        _fw = norm_full.split()[0] if norm_full else ""
        if _fw:
            _cands = db.execute(
                _sql_t("SELECT nombre_normalizado FROM platos "
                       "WHERE nombre_normalizado LIKE :p LIMIT 60"),
                {"p": _fw + "%"},
            ).fetchall()
            _best_score, _best_cand_norm = 0.0, ""
            for _r in _cands:
                _s = difflib.SequenceMatcher(None, norm_full, (_r[0] or "")).ratio()
                if _s > _best_score:
                    _best_score, _best_cand_norm = _s, (_r[0] or "")
            if _best_score >= 0.88 and _sufijos_con_compatibles(norm_full, _best_cand_norm):
                resultado.append(item)
                continue
        last_con = item.rfind(" con ")
        if last_con <= 0:
            resultado.append(item)
            continue
        prefijo = item[:last_con].strip()
        sufijo = item[last_con + 5:].strip()
        if not sufijo:
            resultado.append(item)
            continue
        norm_prefijo = _norm_plato(prefijo)
        norm_sufijo  = _norm_plato(sufijo)
        existe = db.execute(
            _sql_t("SELECT 1 FROM platos WHERE nombre_normalizado = :n LIMIT 1"),
            {"n": norm_prefijo},
        ).fetchone()
        if existe and len(norm_sufijo) >= 4:
            # Sufijo suficientemente largo → split en plato + acompañamiento
            logger.info("Segmentación plato+acompañamiento: '%s' → ['%s', '%s']", item, prefijo, sufijo)
            resultado.extend([prefijo, sufijo])
        elif existe:
            # Sufijo corto (ej. 'pan', 'aji') → dejar completo para que CAPA 1.5
            # construya el plato con todos los ingredientes correctos
            logger.info("Segmentación omitida (sufijo corto '%s'): '%s' → CAPA 1.5", sufijo, item)
            resultado.append(item)
        else:
            resultado.append(item)
    return resultado


# ── Handler principal ─────────────────────────────────────────────────────────

class RegistroComidaHandler:
    """Orquesta el flujo de 5 capas para registrar alimentos por NLP o manual."""

    # ── API pública ──────────────────────────────────────────────────────────

    async def registrar(
        self,
        mensaje: str,
        perfil,
        plan_hoy_data: dict,
        db: Session,
        ia_engine,
    ) -> Dict[str, Any]:
        """Procesa texto/voz para registrar alimentos. Devuelve dict de respuesta."""
        msg_lower = (mensaje or "").lower().strip()

        # ── Modo Estándar: detectar "no sé" antes de la cadena de capas ─────────
        if _es_respuesta_no_se(mensaje):
            return self._respuesta_porcion_estandar_generica(mensaje, db)

        _parece_ejercicio = any(
            x in msg_lower
            for x in ("corr", "trot", "camin", "gym", "pesas", "entren", "sentadilla",
                       "flexion", "serie", "repes", "burpee", "bici", "elev")
        )
        _parece_comida = any(
            x in msg_lower
            for x in ("comi", "comí", "almorcé", "almorce", "desayuné", "desayune",
                       "cené", "cene", "tomé", "tome", "bebí", "bebi", "meriendé",
                       "me comi", "me comí", "probé", "probe", "me jalé")
        )

        pre_extraccion: Optional[dict] = None
        # Reserva de CAPA 0 para platos multi-palabra: solo se usa si CAPA 1 falla.
        # Esto evita que una estimación LLM de bajo accuracy bloquee el catálogo BD.
        _capa0_fallback: Optional[dict] = None

        # CAPA 0: NLPFoodExtractor
        if not _parece_ejercicio:
            capa0_result = await self._capa0_nlp(mensaje, msg_lower, _parece_comida, ia_engine, db)
            if capa0_result.get("_final"):
                return {k: v for k, v in capa0_result.items() if k != "_final"}
            _kcal_c0   = capa0_result.get("calorias", 0)
            _nombre_c0 = (capa0_result.get("alimentos_detectados") or [""])[0]
            # Sanity check por rango mínimo (ej. "Causa Ferreñafana" truncada a 110 kcal)
            _bajo_rango = (
                _capa0_bajo_rango_plato(_nombre_c0, _kcal_c0)
                or _capa0_bajo_rango_plato(msg_lower, _kcal_c0)
            )
            if _kcal_c0 > 0 and not _bajo_rango:
                # ── GUARD DE COHERENCIA: si el mensaje tiene ≥4 palabras (plato complejo)
                # pero CAPA 0 devolvió un nombre con MENOS palabras que el mensaje,
                # es señal de que la IA simplificó/perdió ingredientes.
                # En ese caso: forzar Capa 1.5 (plato_constructor) como autoridad.
                _palabras_msg = len([w for w in msg_lower.split() if len(w) > 2])
                _palabras_c0  = len((_nombre_c0 or "").split())
                _es_simplificacion_peligrosa = (
                    _palabras_msg >= 4
                    and _palabras_c0 <= 3
                    and _palabras_c0 < (_palabras_msg // 2)
                )
                if _es_simplificacion_peligrosa:
                    # Guardar como fallback de último recurso pero preferir Capa 1.5
                    _capa0_fallback = capa0_result
                    logger.info(
                        "CAPA 0 simplificó '%s' → '%s' (%d→%d palabras): cediendo a Capa 1.5",
                        msg_lower[:60], _nombre_c0, _palabras_msg, _palabras_c0,
                    )
                else:
                    # Platos con variantes (ej. "ceviche de merluza" → Capa 0 devuelve
                    # solo "Ceviche"): si el nombre es genérico de 1 palabra Y el usuario
                    # especificó una variante con "de X", ceder a Capa 1/1.5.
                    _PLATOS_MULTI_VARIANTE = frozenset({
                        "ceviche", "cebiche", "tiradito", "sudado", "seco", "causa",
                        "lomo", "jalea", "arroz", "guiso", "estofado", "caldo",
                    })
                    _es_plato_multi_variante = (
                        len((_nombre_c0 or "").split()) == 1
                        and (_nombre_c0 or "").lower().strip() in _PLATOS_MULTI_VARIANTE
                        and " de " in msg_lower
                    )
                    if len((_nombre_c0 or "").split()) >= 2 or _es_plato_multi_variante:
                        # Platos multi-palabra o variantes: CAPA 0 NO tiene autoridad final.
                        _capa0_fallback = capa0_result
                    else:
                        # Alimento simple de una sola palabra: CAPA 0 es suficientemente precisa
                        pre_extraccion = capa0_result

        # Porción de lata (capa especial antes del catálogo)
        if not pre_extraccion and _msg_tiene_porcion_lata(mensaje):
            pre_extraccion = self._capa_lata(mensaje, db)

        # CAPA 1: catálogo platos (incluye CAPA 1.5 — plato_constructor)
        if not pre_extraccion:
            intento_platos = await self._capa1_platos(mensaje, perfil, db)
            if intento_platos:
                pre_extraccion = intento_platos
            elif _capa0_fallback:
                # CAPA 1 no encontró el plato → intentar AlimentosDB con el nombre de CAPA 0
                # Guard de name drift: si Groq añadió palabras de categoría que el usuario
                # NO mencionó (ej. "chicharrón de chancho" → "Pan De Chicharrón De Chiclayo"),
                # usar el texto original del usuario para la búsqueda en BD/APIs.
                _input_limpio = _RE_PREFIJO_IMPERATIVO.sub("", msg_lower).strip()
                _PALABRAS_CATEGORIA = frozenset({
                    "pan", "caldo", "sopa", "estofado", "guiso", "crema", "ensalada",
                    "jugo", "nectar", "bebida", "postre", "torta", "bizcocho",
                    "tamal", "humita", "empanada", "chicha", "refresco",
                })
                _c0_words   = set(_norm_al(_nombre_c0).split())
                _user_words = set(_norm_al(_input_limpio).split())
                _tipo_drift = bool(_PALABRAS_CATEGORIA & (_c0_words - _user_words))
                _char_drift = difflib.SequenceMatcher(
                    None, _norm_al(_nombre_c0), _norm_al(_input_limpio)
                ).ratio() < 0.55
                _es_drift = _tipo_drift or _char_drift
                _nombre_lookup = _input_limpio if _es_drift else _nombre_c0
                if _es_drift:
                    logger.info(
                        "Name drift detectado: CAPA 0 extrajo '%s' → usando texto original '%s' (tipo_drift=%s)",
                        _nombre_c0, _input_limpio[:40], _tipo_drift,
                    )
                _al_srv = AlimentosDBService(db)
                _al_id_fb = _al_srv.resolver_alimento_id(_nombre_lookup)
                if _al_id_fb:
                    _al_obj_fb = db.query(Alimento).filter(Alimento.id == _al_id_fb).first()
                    if _al_obj_fb:
                        # Extraer cantidad del mensaje (ml, g, o recipiente estándar)
                        _gramos_fb = 100.0
                        _m_ml_fb = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:ml|cc)", msg_lower)
                        _m_g_fb  = re.search(r"(\d+(?:[.,]\d+)?)\s*g\b", msg_lower)
                        _VOL_FB = {"vaso": 240.0, "taza": 200.0, "copa": 150.0,
                                   "botella": 500.0, "jarra": 1000.0, "lata": 355.0}
                        _m_rec_fb = re.search(
                            r"\b(vaso|taza|copa|botella|jarra|lata)\b", msg_lower
                        )
                        if _m_ml_fb:
                            _gramos_fb = float(_m_ml_fb.group(1).replace(",", "."))
                        elif _m_g_fb:
                            _gramos_fb = float(_m_g_fb.group(1).replace(",", "."))
                        elif _m_rec_fb:
                            _gramos_fb = _VOL_FB.get(_m_rec_fb.group(1), 100.0)
                        _fac_fb = _gramos_fb / 100.0
                        pre_extraccion = {
                            **_capa0_fallback,
                            "calorias":        round(float(_al_obj_fb.calorias_100g or 0) * _fac_fb, 1),
                            "proteinas_g":     round(float(_al_obj_fb.proteina_100g or 0) * _fac_fb, 1),
                            "carbohidratos_g": round(float(_al_obj_fb.carbohidratos_100g or 0) * _fac_fb, 1),
                            "grasas_g":        round(float(_al_obj_fb.grasas_100g or 0) * _fac_fb, 1),
                            "alimentos_detectados": [_al_obj_fb.nombre],
                            "porcion_g": _gramos_fb,
                            "origen": "bd_alimento",
                        }
                        logger.info(
                            "CAPA 0 fallback corregido con BD: '%s' → %s (%.0fg = %.1f kcal)",
                            _nombre_c0, _al_obj_fb.nombre, _gramos_fb,
                            float(_al_obj_fb.calorias_100g or 0) * _fac_fb,
                        )
                # Si no está en BD local → intentar USDA/FatSecret/Groq + guardar en BD
                if not pre_extraccion and _nombre_c0:
                    try:
                        from app.services.asistente_nutricion import _buscar_o_crear_alimento_async
                        _al_pipeline = await _buscar_o_crear_alimento_async(
                            db, _norm_al(_nombre_lookup), _nombre_lookup
                        )
                        if _al_pipeline:
                            _gramos_pl = 100.0
                            _m_ml_pl = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:ml|cc)", msg_lower)
                            _m_g_pl  = re.search(r"(\d+(?:[.,]\d+)?)\s*g\b", msg_lower)
                            _m_rec_pl = re.search(
                                r"\b(vaso|taza|copa|botella|jarra|lata)\b", msg_lower
                            )
                            if _m_ml_pl:
                                _gramos_pl = float(_m_ml_pl.group(1).replace(",", "."))
                            elif _m_g_pl:
                                _gramos_pl = float(_m_g_pl.group(1).replace(",", "."))
                            elif _m_rec_pl:
                                _gramos_pl = {"vaso":240,"taza":200,"copa":150,
                                              "botella":500,"jarra":1000,"lata":355}.get(
                                    _m_rec_pl.group(1), 100.0)
                            _fac_pl = _gramos_pl / 100.0
                            pre_extraccion = {
                                **_capa0_fallback,
                                "calorias":        round(float(_al_pipeline.calorias_100g or 0) * _fac_pl, 1),
                                "proteinas_g":     round(float(_al_pipeline.proteina_100g or 0) * _fac_pl, 1),
                                "carbohidratos_g": round(float(_al_pipeline.carbohidratos_100g or 0) * _fac_pl, 1),
                                "grasas_g":        round(float(_al_pipeline.grasas_100g or 0) * _fac_pl, 1),
                                "alimentos_detectados": [_al_pipeline.nombre],
                                "porcion_g": _gramos_pl,
                                "origen": "bd_alimento",
                            }
                            logger.info(
                                "CAPA 0 fallback → pipeline externo: '%s' → %s guardado en BD (%.0fg = %.1f kcal)",
                                _nombre_c0, _al_pipeline.nombre, _gramos_pl,
                                float(_al_pipeline.calorias_100g or 0) * _fac_pl,
                            )
                    except Exception as _epipe:
                        logger.debug("Pipeline externo para '%s': %s", _nombre_c0, _epipe)
                if not pre_extraccion:
                    logger.info(
                        "CAPA 1 no resolvió '%s' — usando estimación CAPA 0 como fallback",
                        _nombre_c0,
                    )
                    pre_extraccion = _capa0_fallback

        # ── Resolver ítems pendientes de CAPA 1 vía alimentos ────────────────
        # Cuando CAPA 1 encontró algún plato pero dejó ítems en _no_resueltos
        # (ej. "arroz con pollo y incakola" → arroz resuelto, incakola pendiente),
        # intentar cada ítem pendiente en la tabla de alimentos.
        if pre_extraccion and pre_extraccion.get("_no_resueltos"):
            _pendientes = pre_extraccion.pop("_no_resueltos", [])
            _srv = AlimentosDBService(db)
            _RE_VERBO_ITEM = re.compile(
                r"^(?:tom[eé]\s+|beb[íi]\s+|com[íi]\s+|"
                r"(?:un|una|unos|unas|medio|media)\s+)?",
                re.IGNORECASE,
            )
            # Capturar macros del plato principal antes de acumular extras
            _macros_por_alimento: list[dict] = []
            _nombres_plato = pre_extraccion.get("alimentos_detectados") or []
            if _nombres_plato:
                _kcal_plato = round(float(pre_extraccion.get("calorias", 0)), 2)
                _prot_plato = round(float(pre_extraccion.get("proteinas_g", 0)), 2)
                _carb_plato = round(float(pre_extraccion.get("carbohidratos_g", 0)), 2)
                _gras_plato = round(float(pre_extraccion.get("grasas_g", 0)), 2)
                for _np in _nombres_plato:
                    _macros_por_alimento.append({
                        "nombre": _np,
                        "kcal":   _kcal_plato / max(1, len(_nombres_plato)),
                        "prot_g": _prot_plato / max(1, len(_nombres_plato)),
                        "carb_g": _carb_plato / max(1, len(_nombres_plato)),
                        "gras_g": _gras_plato / max(1, len(_nombres_plato)),
                    })
            for _item in _pendientes:
                _q = _RE_VERBO_ITEM.sub("", _item.strip()).strip()
                if not _q or len(_q) < 2:
                    continue
                try:
                    # Detectar formato "tipo_recipiente:nombre" (ej. "vaso:cocoa")
                    # producido por CAPA 1 cuando el ítem tenía prefijo de recipiente.
                    _rec_tipo_resol: str = ""
                    _m_rec_fmt = re.match(
                        r'^(vaso|taza|copa|botella|jarra|lata):(.+)$', _q, re.IGNORECASE
                    )
                    if _m_rec_fmt:
                        _rec_tipo_resol = _m_rec_fmt.group(1).lower()
                        _q = _m_rec_fmt.group(2).strip()
                    # Extraer cantidad ml/g del ítem antes de buscar el alimento
                    _gramos_item = 100.0
                    _m_ml = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:ml|cc)", _q, re.IGNORECASE)
                    _m_g  = re.search(r"(\d+(?:[.,]\d+)?)\s*g\b", _q, re.IGNORECASE)
                    if _m_ml:
                        _gramos_item = float(_m_ml.group(1).replace(",", "."))
                        _q = re.sub(r"\s*(?:de\s+)?\d+(?:[.,]\d+)?\s*(?:ml|cc)", "", _q, flags=re.IGNORECASE).strip()
                    elif _m_g:
                        _gramos_item = float(_m_g.group(1).replace(",", "."))
                        _q = re.sub(r"\s*\d+(?:[.,]\d+)?\s*g\b", "", _q, flags=re.IGNORECASE).strip()
                    # CAPA 1: buscar en BD local
                    _al_id = _srv.resolver_alimento_id(_q)
                    _al_obj = (
                        db.query(Alimento).filter(Alimento.id == _al_id).first()
                        if _al_id else None
                    )
                    # CAPA 2-3-Groq: si no está en BD, llamar pipeline completo
                    if not _al_obj:
                        try:
                            from app.services.asistente_nutricion import _buscar_o_crear_alimento_async
                            _q_norm = _norm_al(_q)
                            _al_obj = await _buscar_o_crear_alimento_async(db, _q_norm, _q)
                            if _al_obj:
                                logger.info("Ítem pendiente '%s' → USDA/FatSecret/Groq → %s", _item, _al_obj.nombre)
                        except Exception as _ef:
                            logger.debug("Pipeline externo para '%s': %s", _q, _ef)
                    if _al_obj:
                        # Si el ítem vino de un recipiente (vaso:cocoa), consultar
                        # alimento_unidades para obtener los gramos reales del polvo/líquido.
                        # Ejemplo: vaso de cocoa → 10g de polvo, no 240ml × densidad.
                        if _rec_tipo_resol:
                            try:
                                from sqlalchemy import text as _sql_au
                                _au = db.execute(_sql_au(
                                    "SELECT gramos FROM alimento_unidades "
                                    "WHERE alimento_id = :ai AND nombre = :u LIMIT 1"
                                ), {"ai": _al_obj.id, "u": _rec_tipo_resol}).fetchone()
                                if _au:
                                    _gramos_item = float(_au[0])
                                else:
                                    _VOL_FALLBACK = {
                                        "vaso": 240.0, "taza": 200.0, "copa": 150.0,
                                        "botella": 500.0, "jarra": 1000.0, "lata": 355.0,
                                    }
                                    _gramos_item = _VOL_FALLBACK.get(_rec_tipo_resol, 100.0)
                            except Exception:
                                pass
                        _factor = _gramos_item / 100.0
                        _kcal_extra  = round(float(_al_obj.calorias_100g or 0) * _factor, 1)
                        _prot_extra  = round(float(_al_obj.proteina_100g or 0) * _factor, 2)
                        _carb_extra  = round(float(_al_obj.carbohidratos_100g or 0) * _factor, 2)
                        _gras_extra  = round(float(_al_obj.grasas_100g or 0) * _factor, 2)
                        pre_extraccion["calorias"]        = round(pre_extraccion["calorias"]        + _kcal_extra, 1)
                        pre_extraccion["proteinas_g"]     = round(pre_extraccion["proteinas_g"]     + _prot_extra, 1)
                        pre_extraccion["carbohidratos_g"] = round(pre_extraccion["carbohidratos_g"] + _carb_extra, 1)
                        pre_extraccion["grasas_g"]        = round(pre_extraccion["grasas_g"]        + _gras_extra, 1)
                        pre_extraccion.setdefault("alimentos_detectados", []).append(_al_obj.nombre)
                        pre_extraccion.setdefault("extras_nutricionales", []).append(
                            f"{_al_obj.nombre} {int(_gramos_item)}g ({_kcal_extra} kcal)"
                        )
                        _macros_por_alimento.append({
                            "nombre": _al_obj.nombre,
                            "kcal":   _kcal_extra,
                            "prot_g": _prot_extra,
                            "carb_g": _carb_extra,
                            "gras_g": _gras_extra,
                        })
                        logger.info("Ítem pendiente '%s' resuelto → %s (%.0fg = %.1f kcal)", _item, _al_obj.nombre, _gramos_item, _kcal_extra)
                    else:
                        logger.warning("Ítem pendiente '%s' no resuelto ni en BD ni en APIs externas", _item)
                except Exception as _ep:
                    logger.debug("Error resolviendo ítem pendiente '%s': %s", _item, _ep)
            if _macros_por_alimento:
                pre_extraccion["alimentos_con_macros"] = _macros_por_alimento
        # ─────────────────────────────────────────────────────────────────────

        return await self._aplicar_y_persistir(pre_extraccion, perfil, plan_hoy_data, db, ia_engine, mensaje)

    # ── Modo Estándar ─────────────────────────────────────────────────────────

    def _respuesta_porcion_estandar_generica(
        self, mensaje: str, db: Session
    ) -> Dict[str, Any]:
        """
        Cuando el usuario responde 'no sé' o similar al ser preguntado por
        una cantidad, extrae el alimento del mensaje y aplica porción estándar.
        Garantiza que siempre haya un registro (critical para adherencia ML).
        """
        # Eliminar verbos de acción del inicio para evitar "Porción estándar de Comí un X"
        _msg_limpio = re.sub(
            r"(?i)^(com[íi]|tom[eé]|beb[íi]|desayun[eé]|almorc[eé]|cen[eé]|"
            r"meriend[eé]|consumi[óo]|llev[eé]|tuv[ei]|prob[eé])\s+"
            r"(un|una|unos|unas|medio|media|algo\s+de|un\s+poco\s+de)?\s*",
            "",
            (mensaje or "").strip(),
        )
        # Intentar extraer nombre de alimento del mensaje ya limpio
        m = re.search(
            r"(?i)(?:de\s+|del?\s+|cuánto\s+(?:de\s+)?)?([a-záéíóúüñ][\w\s]{2,25}?)(?:\s*[,.]|$)",
            _msg_limpio,
        )
        nombre_alim = m.group(1).strip().capitalize() if m else "alimento"
        gramos, descripcion = _get_porcion_estandar(nombre_alim, db)

        return {
            "success": False,
            "tipo_detectado": "modo_estandar",
            "alimentos": [nombre_alim],
            "campo_faltante": "cantidad",
            "porcion_estandar_g": gramos,
            "porcion_descripcion": descripcion,
            "mensaje": (
                f"No hay problema, lo registraré con una {descripcion} de {nombre_alim}. "
                f"Confírmame y lo anoto, o dime la cantidad exacta si la recuerdas."
            ),
            "requiere_confirmacion": True,
            "datos": {"gramos_sugeridos": gramos},
        }

    async def registrar_manual(self, body: dict, perfil, db: Session) -> Dict[str, Any]:
        """Registro manual: el usuario ingresa macros/kcal por porción o por 100g."""
        nombre = str((body or {}).get("nombre") or "").strip()
        if not nombre or len(nombre) < 2:
            raise ValueError("Nombre inválido")

        kcal = float((body or {}).get("calorias") or 0)
        p    = float((body or {}).get("proteinas_g") or 0)
        c    = float((body or {}).get("carbohidratos_g") or 0)
        g    = float((body or {}).get("grasas_g") or 0)
        porcion_g = float((body or {}).get("porcion_g") or 0) or 100.0

        if kcal <= 0:
            raise ValueError("Calorías deben ser > 0")

        categoria      = str((body or {}).get("categoria") or "manual").strip()[:100]
        unidad         = (body or {}).get("unidad")
        unidad         = str(unidad).strip() if unidad else None
        gramos_unidad  = (body or {}).get("gramos_por_unidad")
        gramos_unidad  = float(gramos_unidad) if gramos_unidad is not None else None

        nn  = _norm_al(nombre)
        f   = 100.0 / max(1.0, porcion_g)
        row = db.query(Alimento).filter(Alimento.nombre_normalizado == nn).first()
        if row:
            row.nombre            = nombre[:255]
            row.calorias_100g     = round(kcal * f, 2)
            row.proteina_100g     = round(p * f, 2)
            row.carbohidratos_100g = round(c * f, 2)
            row.grasas_100g       = round(g * f, 2)
            row.categoria         = categoria
            row.fuente            = "manual"
        else:
            row = Alimento(
                nombre=nombre[:255], nombre_normalizado=nn[:255],
                calorias_100g=round(kcal * f, 2), proteina_100g=round(p * f, 2),
                carbohidratos_100g=round(c * f, 2), grasas_100g=round(g * f, 2),
                fibra_100g=None, azucar_100g=None,
                categoria=categoria, fuente="manual", id_externo=None,
            )
            db.add(row)
            db.flush()

        if unidad and gramos_unidad and gramos_unidad > 0:
            u_norm = unidad.strip().lower()[:100]
            existing_u = (
                db.query(AlimentoUnidad)
                .filter(AlimentoUnidad.alimento_id == row.id,
                        AlimentoUnidad.nombre.ilike(u_norm))
                .first()
            )
            if existing_u:
                existing_u.gramos = float(gramos_unidad)
            else:
                db.add(AlimentoUnidad(
                    alimento_id=row.id, nombre=u_norm, gramos=float(gramos_unidad)
                ))

        hoy = get_peru_date()

        extraccion = {
            "es_comida": True, "es_ejercicio": False,
            "calorias": round(kcal, 1), "proteinas_g": round(p, 1),
            "carbohidratos_g": round(c, 1), "grasas_g": round(g, 1),
            "fibra_g": 0, "azucar_g": 0, "sodio_mg": 0,
            "alimentos_detectados": [nombre], "ejercicios_detectados": [],
            "calidad_nutricional": "Alta", "porcion_g": porcion_g, "origen": "manual",
        }

        # REGLA 3: modo aditivo (misma lógica que _aplicar_y_persistir)
        from app.services.trazabilidad import crear_comida_registros
        crear_comida_registros(
            client_id=perfil.id,
            fecha=hoy,
            extraccion=extraccion,
            texto_original=nombre,
            db=db,
            momento=None,
        )
        progreso = db.query(ProgresoCalorias).filter(
            ProgresoCalorias.client_id == perfil.id,
            ProgresoCalorias.fecha == hoy,
        ).first()
        if not progreso:
            progreso = ProgresoCalorias(client_id=perfil.id, fecha=hoy)
            db.add(progreso)
        progreso.calorias_consumidas      = (progreso.calorias_consumidas or 0) + int(round(kcal))
        progreso.proteinas_consumidas     = round((progreso.proteinas_consumidas or 0) + p, 1)
        progreso.carbohidratos_consumidos = round((progreso.carbohidratos_consumidos or 0) + c, 1)
        progreso.grasas_consumidas        = round((progreso.grasas_consumidas or 0) + g, 1)
        registrar_preferencias_alimentos(extraccion, perfil, db)
        db.commit()

        return {
            "success": True, "tipo_detectado": "comida",
            "alimentos": [nombre], "advertencia_prohibido": None, "alerta_macros": None,
            "balance_actualizado": {
                "consumido": progreso.calorias_consumidas,
                "quemado":   progreso.calorias_quemadas,
            },
            "datos": {
                "calorias": extraccion["calorias"], "proteinas_g": extraccion["proteinas_g"],
                "carbohidratos_g": extraccion["carbohidratos_g"],
                "grasas_g": extraccion["grasas_g"],
                "azucar_g": 0, "fibra_g": 0, "sodio_mg": 0, "calidad": "Alta",
            },
            "mensaje": f"✅ Registré manual: {nombre} — {extraccion['calorias']} kcal.",
        }

    # ── Capas privadas ────────────────────────────────────────────────────────

    async def _capa0_nlp(
        self, mensaje: str, msg_lower: str, parece_comida: bool, ia_engine, db: Session
    ) -> dict:
        """CAPA 0: NLPFoodExtractor — Llama-3 extrae JSON, Python calcula desde BD."""
        try:
            from app.services.nlp_food_extractor import (
                NLPFoodExtractor, contiene_modificador_ficticio, _nombre_es_no_alimento
            )
            extractor = NLPFoodExtractor(ia_engine, db)

            if contiene_modificador_ficticio(mensaje):
                return {
                    "_final": True, "success": False,
                    "tipo_detectado": "ficcion_bloqueada", "alimentos": [], "datos": {},
                    "mensaje": (
                        "No puedo registrar ese alimento porque contiene un ingrediente "
                        "ficticio o mitológico. Por favor regístra un alimento real. 🍽️"
                    ),
                }

            # Guard: mensaje entero que es claramente no-comida
            # Extrae la "palabra base" del mensaje limpio y la valida contra NO_ALIMENTOS.
            _msg_base = re.sub(
                r"(?i)^(com[ií]|tom[eé]|beb[ií]|registra?|anota?|guard[ao])\s+(un[ao]?\s+)?",
                "", msg_lower,
            ).strip()
            _palabra_base = _msg_base.split()[0] if _msg_base else ""
            if _nombre_es_no_alimento(_msg_base) or _nombre_es_no_alimento(_palabra_base):
                return {
                    "_final": True, "success": False,
                    "tipo_detectado": "no_alimento_bloqueado", "alimentos": [], "datos": {},
                    "mensaje": (
                        f"'{_palabra_base or _msg_base}' no parece ser un alimento. "
                        "¿Quisiste decir otra cosa? Registra una comida o bebida real. 🍽️"
                    ),
                }

            if extractor._es_negacion(mensaje):
                return {
                    "_final": True, "success": True,
                    "tipo_detectado": "ninguno", "alimentos": [], "datos": {},
                    "mensaje": "Entendido, no registré ningún alimento. Si comiste algo, cuéntame 😊",
                }

            resultado = await extractor.extraer(mensaje)

            if resultado and resultado.calorias_total > 0:
                adv = resultado.advertencia
                _total_g = sum(it.gramos_totales or 0 for it in resultado.items)
                ext = {
                    "calorias": resultado.calorias_total,
                    "proteinas_g": resultado.proteinas_total,
                    "carbohidratos_g": resultado.carbohidratos_total,
                    "grasas_g": resultado.grasas_total,
                    "fibra_g": 0, "azucar_g": 0, "sodio_mg": 0,
                    "es_comida": True, "es_ejercicio": False,
                    "alimentos_detectados": resultado.nombres,
                    "ejercicios_detectados": [],
                    "calidad_nutricional": "Alta",
                    "origen": "nlp_extractor",
                    "advertencia": adv,
                    "porcion_g": _total_g if _total_g > 0 else None,
                    "alimentos_con_macros": [
                        {
                            "nombre": it.alimento,
                            "kcal":   it.calorias,
                            "prot_g": it.proteinas_g,
                            "carb_g": it.carbohidratos_g,
                            "gras_g": it.grasas_g,
                            "gramos": it.gramos_totales,
                        }
                        for it in resultado.items
                    ],
                }
                if adv and ("kcal" in adv or "correcto" in adv.lower()):
                    ext["_warn_cantidad"] = adv
                return ext

            if resultado is None and parece_comida:
                _comunes = {
                    "arroz", "pollo", "papa", "fideos", "lentejas", "pan", "leche",
                    "huevo", "carne", "pescado", "fruta", "ensalada", "sopa", "agua",
                    "avena", "queso", "platano", "manzana", "naranja", "brocoli",
                    "atun", "salmon", "ceviche", "lomo", "palta", "tomate", "cebolla",
                }
                palabras = set(re.sub(r"[^a-z\s]", "", msg_lower).split())
                if not (palabras & _comunes):
                    nombre_aprox = re.sub(
                        r"^(comi|comí|tome|tomé|desayune|almorcé|almorce|bebi|bebí)\s*",
                        "", msg_lower, flags=re.IGNORECASE,
                    ).strip()[:50]
                    return {
                        "_final": True, "success": False,
                        "tipo_detectado": "desconocido", "alimentos": [], "datos": {},
                        "mensaje": (
                            f"No reconocí '{nombre_aprox}' como un alimento. "
                            "Prueba con nombres más comunes como: "
                            "'pollo a la plancha', 'arroz con leche', 'manzana'. 🍽️"
                        ),
                    }
        except Exception as e:
            logger.error("Error CAPA 0: %s", e)
        return {}

    def _capa_lata(self, mensaje: str, db: Session) -> Optional[dict]:
        """Detecta porciones tipo 'media lata de atún' y calcula macros."""
        try:
            svc  = AlimentosDBService(db)
            pors = svc.extraer_porciones_desde_texto(mensaje)
            if pors:
                return {
                    "calorias":        round(sum(p.kcal for p in pors), 1),
                    "proteinas_g":     round(sum(p.p_g for p in pors), 1),
                    "carbohidratos_g": round(sum(p.c_g for p in pors), 1),
                    "grasas_g":        round(sum(p.g_g for p in pors), 1),
                    "fibra_g": 0, "azucar_g": 0, "sodio_mg": 0,
                    "es_comida": True, "es_ejercicio": False,
                    "alimentos_detectados": [p.nombre_alimento for p in pors],
                    "ejercicios_detectados": [],
                    "calidad_nutricional": "Alta", "origen": "postgres",
                }
        except Exception:
            pass
        return None

    async def _capa1_platos(self, mensaje: str, perfil, db: Session) -> Optional[dict]:
        """CAPA 1: Busca en catálogo platos calculando macros desde plato_ingredientes × alimentos."""
        from sqlalchemy import text as _sql
        from app.models.plato import Plato

        items = _split_items_from_message(mensaje)
        if not items:
            return None
        # Expandir "plato_conocido + con + acompañamiento" antes del loop de búsqueda
        items = _expandir_compuestos_con(items, db)

        _SQL = (
            "SELECT p.id, p.nombre,"
            " SUM(a.calorias_100g * pi2.gramos / 100.0),"
            " SUM(a.proteina_100g * pi2.gramos / 100.0),"
            " SUM(a.carbohidratos_100g * pi2.gramos / 100.0),"
            " SUM(a.grasas_100g * pi2.gramos / 100.0),"
            " SUM(pi2.gramos)"   # col[6]: peso total del plato para escalar por gramos
            " FROM platos p"
            " JOIN plato_ingredientes pi2 ON pi2.plato_id = p.id"
            " JOIN alimentos a ON a.id = pi2.alimento_id"
            " WHERE p.nombre_normalizado = :q"
            " GROUP BY p.id, p.nombre LIMIT 1"
        )

        # ── Fast-path: buscar en historial_recomendaciones reciente (últimas 24h) ──
        # Cuando el sistema recomienda "Crema de Verduras con Plátano" y el usuario
        # escribe "comí crema de verduras con plátano", buscamos primero en su historial
        # reciente para evitar que el NLP falle al re-identificar el plato recomendado.
        try:
            from app.models.historial_recomendacion import HistorialRecomendacion
            from datetime import timedelta
            _msg_clean_hist = _RE_PREFIJO_IMPERATIVO.sub("", mensaje.lower()).strip()
            _msg_clean_hist = unicodedata.normalize("NFC", _msg_clean_hist)
            _desde = datetime.now() - timedelta(hours=24)
            _historial_rows = (
                db.query(HistorialRecomendacion)
                .filter(
                    HistorialRecomendacion.client_id == perfil.id,
                    HistorialRecomendacion.created_at >= _desde,
                    HistorialRecomendacion.plato_id.isnot(None),
                )
                .order_by(HistorialRecomendacion.created_at.desc())
                .limit(15)
                .all()
            )
            for _hr in _historial_rows:
                _nn_hist = unicodedata.normalize("NFC", (_hr.nombre_plato or "").lower().strip())
                _score_hist = difflib.SequenceMatcher(a=_msg_clean_hist, b=_nn_hist).ratio()
                # Guard: no aceptar si los modificadores "con X" son incompatibles
                if _score_hist >= 0.80 and _sufijos_con_compatibles(_msg_clean_hist, _nn_hist) and _sufijos_de_compatibles(_msg_clean_hist, _nn_hist):
                    # Match fuerte con un plato recomendado → buscar sus macros reales en BD
                    from sqlalchemy import text as _sql_t
                    _row_hist = db.execute(_sql_t(
                        "SELECT p.id, p.nombre,"
                        " SUM(a.calorias_100g * pi2.gramos / 100.0),"
                        " SUM(a.proteina_100g * pi2.gramos / 100.0),"
                        " SUM(a.carbohidratos_100g * pi2.gramos / 100.0),"
                        " SUM(a.grasas_100g * pi2.gramos / 100.0)"
                        " FROM platos p"
                        " JOIN plato_ingredientes pi2 ON pi2.plato_id = p.id"
                        " JOIN alimentos a ON a.id = pi2.alimento_id"
                        " WHERE p.id = :pid GROUP BY p.id, p.nombre"
                    ), {"pid": _hr.plato_id}).fetchone()
                    if _row_hist:
                        logger.info(
                            "Fast-path historial: '%s' → plato id=%s '%s' (score=%.2f)",
                            _msg_clean_hist, _hr.plato_id, _hr.nombre_plato, _score_hist,
                        )
                        from app.services.asistente_nutricion import _cargar_ingredientes_bd
                        _desglose_h, _desglose_total_h = [], ""
                        try:
                            _kcal_h = round(float(_row_hist[2] or 0), 1)
                            _p_h = round(float(_row_hist[3] or 0), 1)
                            _c_h = round(float(_row_hist[4] or 0), 1)
                            _g_h = round(float(_row_hist[5] or 0), 1)
                            _desglose_h = _cargar_ingredientes_bd(db, _hr.plato_id)
                            _desglose_total_h = (
                                f"Total: {_kcal_h} kcal | P:{_p_h}g | C:{_c_h}g | G:{_g_h}g"
                            )
                        except Exception:
                            _kcal_h = round(float(_row_hist[2] or 0), 1)
                            _p_h = round(float(_row_hist[3] or 0), 1)
                            _c_h = round(float(_row_hist[4] or 0), 1)
                            _g_h = round(float(_row_hist[5] or 0), 1)
                        return {
                            "calorias":        _kcal_h,
                            "proteinas_g":     _p_h,
                            "carbohidratos_g": _c_h,
                            "grasas_g":        _g_h,
                            "fibra_g": 0, "azucar_g": 0, "sodio_mg": 0,
                            "es_comida": True, "es_ejercicio": False,
                            "alimentos_detectados": [_row_hist[1]],
                            "ejercicios_detectados": [],
                            "calidad_nutricional": "Alta",
                            "origen": "platos",
                            "desglose_ingredientes": _desglose_h,
                            "desglose_total": _desglose_total_h,
                        }
        except Exception as _eh:
            logger.debug("Fast-path historial error (no crítico): %s", _eh)


        matched: List[tuple] = []
        # REGLA 1: trackear ítems que CAPA 1 no pudo resolver (evita pérdida silenciosa)
        _no_resueltos_c1: List[str] = []

        # REGLA 2: prefijos de recipiente que deben quitarse antes de buscar en BD
        _RE_PREFIJO_RECIPIENTE = re.compile(
            r'^(?:plato\s+de|porci[oó]n\s+de|medio\s+plato\s+de|media\s+porci[oó]n\s+de'
            r'|vaso\s+de|taza\s+de|copa\s+de|botella\s+de|jarra\s+de|lata\s+de)\s+',
            re.IGNORECASE,
        )
        # Volúmenes estándar por recipiente (en ml, usado como gramos para líquidos)
        _VOL_RECIPIENTE: dict[str, float] = {
            "vaso": 240.0, "taza": 200.0, "copa": 150.0,
            "botella": 500.0, "jarra": 1000.0, "lata": 355.0,
        }

        # Regex para prefijo de gramos: "250g de X" / "250 gr de X" / "200ml de X"
        _RE_GRAM_PREFIX = re.compile(
            r"^(\d+(?:[.,]\d+)?)\s*(?:g|gr|gramos?|ml|cc)\s+(?:de\s+)?(.+)$",
            re.IGNORECASE,
        )

        for it in items:
            # ── Detectar prefijo de gramaje explícito: "250g de ají de gallina" ──
            # Si el ítem empieza con Xg/Xml, extraer el alimento y guardar gramos
            # para escalar los macros del plato tras encontrarlo en BD.
            _gram_explicito: Optional[float] = None
            _m_gp = _RE_GRAM_PREFIX.match(it)
            if _m_gp:
                _gram_explicito = float(_m_gp.group(1).replace(",", "."))
                it = _m_gp.group(2).strip()

            m = re.match(
                r"^\s*((?:\d+(?:[.,]\d+)?)|uno|una|un|dos|tres|cuatro|cinco|medio|media)\s+(.*)$",
                it, re.IGNORECASE,
            )
            qty, name_part = 1.0, it
            if m:
                qty, name_part = _parse_qty(m.group(1)), m.group(2).strip()
                # Si name_part empieza con unidad de masa/volumen, es otro gram prefix
                # (ej. el regex numérico separó "250" de "g de ají de gallina")
                _m_unit_residuo = re.match(
                    r'^(?:g|gr|gramos?|ml|cc)\s+(?:de\s+)?', name_part, re.IGNORECASE
                )
                if _m_unit_residuo and _gram_explicito is None:
                    _gram_explicito = qty
                    name_part = name_part[_m_unit_residuo.end():].strip()
                    qty = 1.0

            # REGLA 2: eliminar prefijo de recipiente antes de buscar
            # Detectar volumen estándar si el prefijo es un recipiente conocido
            _m_rec = re.match(
                r'^(vaso|taza|copa|botella|jarra|lata)\s+de\s+', name_part, re.IGNORECASE
            )
            # Anotar tipo de recipiente para que el resolver consulte alimento_unidades.
            _vol_anotado: float = 0.0
            _rec_tipo_anotado: str = ""
            if _m_rec:
                _rec_tipo_anotado = _m_rec.group(1).lower()
                _rec_vol = _VOL_RECIPIENTE.get(_rec_tipo_anotado, 0.0)
                if _rec_vol:
                    name_part = name_part[_m_rec.end():].strip()
                    _vol_anotado = _rec_vol
            name_part = _RE_PREFIJO_RECIPIENTE.sub("", name_part).strip()

            nn = _norm_plato(name_part)
            if not nn or len(nn) < 4:
                continue

            row = db.execute(_sql(_SQL), {"q": nn}).fetchone()
            if not row:
                # Fallback similitud coseno sobre los últimos 250 platos
                cands = db.query(Plato.id, Plato.nombre, Plato.nombre_normalizado).order_by(Plato.id.desc()).limit(250).all()
                best_id, best_score, best_norm = None, 0.0, ""
                for pid, pnombre, pnn in cands:
                    rn = _norm_plato(pnn or pnombre or "")
                    score = difflib.SequenceMatcher(a=nn, b=rn).ratio()
                    if score > best_score:
                        best_score, best_id, best_norm = score, pid, rn
                if best_id and best_score >= 0.88 and _sufijos_con_compatibles(nn, best_norm) and _sufijos_de_compatibles(nn, best_norm):
                    row = db.execute(_sql(
                        "SELECT p.id, p.nombre,"
                        " SUM(a.calorias_100g * pi2.gramos / 100.0),"
                        " SUM(a.proteina_100g * pi2.gramos / 100.0),"
                        " SUM(a.carbohidratos_100g * pi2.gramos / 100.0),"
                        " SUM(a.grasas_100g * pi2.gramos / 100.0),"
                        " SUM(pi2.gramos)"
                        " FROM platos p JOIN plato_ingredientes pi2 ON pi2.plato_id = p.id"
                        " JOIN alimentos a ON a.id = pi2.alimento_id WHERE p.id = :pid"
                        " GROUP BY p.id, p.nombre"
                    ), {"pid": best_id}).fetchone()
                # Último intento: buscar con nombre completo incluyendo el recipiente.
                # "cocoa" falla → prueba "vaso de cocoa" → encuentra plato 486 directamente.
                if not row and _rec_tipo_anotado:
                    _nn_full_rec = _norm_plato(f"{_rec_tipo_anotado} de {name_part}")
                    row = db.execute(_sql(_SQL), {"q": _nn_full_rec}).fetchone()
            if row:
                # Escalar por gramos explícitos: "250g de ají de gallina"
                # row[6] = SUM(pi2.gramos) = peso estándar del plato completo
                if _gram_explicito and _gram_explicito > 0:
                    _peso_std = float(row[6] or 0)
                    if _peso_std > 0:
                        _scale = _gram_explicito / _peso_std
                        row = (row[0], row[1],
                               (row[2] or 0) * _scale,
                               (row[3] or 0) * _scale,
                               (row[4] or 0) * _scale,
                               (row[5] or 0) * _scale,
                               _gram_explicito)
                        logger.info(
                            "CAPA 1: '%s' escalado a %.0fg / %.0fg std → factor %.2f",
                            row[1], _gram_explicito, _peso_std, _scale,
                        )
                matched.append((row, qty))
            else:
                # REGLA 1: ítem no resuelto por CAPA 1 — registrar para no perderlo silenciosamente
                # Usar formato "tipo_recipiente:nombre" para que el resolver
                # pueda consultar alimento_unidades (ej. vaso:cocoa → 10g).
                if _rec_tipo_anotado:
                    _unresolved_str = f"{_rec_tipo_anotado}:{name_part}"
                elif _vol_anotado:
                    _unresolved_str = f"{int(_vol_anotado)}ml {name_part}"
                else:
                    _unresolved_str = name_part
                _no_resueltos_c1.append(_unresolved_str)

        if not matched:
            # ── Capa 1.5: construcción dinámica de platos ──────────────────────
            # Caso A0 (NUEVO): si hay múltiples items pero el texto completo parece
            # un ÚNICO plato compuesto (ej: "ensalada de plátano con aceite de oliva"),
            # intentar construirlo como plato único ANTES de dividir por items.
            # Esto previene que "aceite de oliva" se trate como plato separado.
            if len(items) > 1:
                # Reconstruir el nombre del plato completo desde el mensaje original
                _msg_clean = _RE_PREFIJO_IMPERATIVO.sub("", mensaje.lower()).strip()
                _msg_clean = _RE_CANTIDAD_INICIO.sub("", _msg_clean).strip()
                _msg_clean = re.sub(r"\s+", " ", _msg_clean).strip()
                # Solo intentar si el texto limpio tiene "con" pero NO tiene "y"/"más"
                # (si hay "y"/"más" es multi-alimento, no un plato compuesto único)
                _tiene_separador_multi = bool(re.search(r"\s+y\s+|\s+m[aá]s\s+", _msg_clean))
                if " con " in _msg_clean and not _tiene_separador_multi and len(_msg_clean.split()) >= 3:
                    try:
                        from app.services.plato_constructor import crear_plato_dinamico
                        _plato_full = await crear_plato_dinamico(db, _msg_clean)
                        if _plato_full:
                            _mf = _plato_full.calcular_macros()
                            logger.info(
                                "Capa1.5 A0: texto completo '%s' → plato '%s' (%.0f kcal)",
                                _msg_clean, _plato_full.nombre, _mf["calorias"],
                            )
                            return {
                                "calorias":        _mf["calorias"],
                                "proteinas_g":     _mf["proteinas_g"],
                                "carbohidratos_g": _mf["carbohidratos_g"],
                                "grasas_g":        _mf["grasas_g"],
                                "fibra_g": 0, "azucar_g": 0, "sodio_mg": 0,
                                "es_comida": True, "es_ejercicio": False,
                                "alimentos_detectados": [_plato_full.nombre],
                                "ejercicios_detectados": [],
                                "calidad_nutricional": "Alta",
                                "origen": "plato_dinamico",
                            }
                    except Exception as _eA0:
                        logger.debug("Capa1.5 A0: no construido '%s': %s", _msg_clean, _eA0)

            # Caso A: query de un solo item que parece un plato completo
            if len(items) == 1 and _es_candidato_plato_capa15(items[0]):
                try:
                    from app.services.plato_constructor import crear_plato_dinamico
                    plato_nuevo = await crear_plato_dinamico(db, items[0])
                    if plato_nuevo:
                        macros = plato_nuevo.calcular_macros()
                        from app.services.asistente_nutricion import _cargar_ingredientes_bd
                        desglose_d, desglose_total_d = [], ""
                        try:
                            desglose_d = _cargar_ingredientes_bd(db, plato_nuevo.id)
                            desglose_total_d = (
                                f"Total: {macros['calorias']} kcal"
                                f" | P:{macros['proteinas_g']}g"
                                f" | C:{macros['carbohidratos_g']}g"
                                f" | G:{macros['grasas_g']}g"
                            )
                        except Exception:
                            pass
                        return {
                            "calorias":        macros["calorias"],
                            "proteinas_g":     macros["proteinas_g"],
                            "carbohidratos_g": macros["carbohidratos_g"],
                            "grasas_g":        macros["grasas_g"],
                            "fibra_g": 0, "azucar_g": 0, "sodio_mg": 0,
                            "es_comida": True, "es_ejercicio": False,
                            "alimentos_detectados": [plato_nuevo.nombre],
                            "ejercicios_detectados": [],
                            "calidad_nutricional": "Media",
                            "origen": "plato_dinamico",
                            "desglose_ingredientes": desglose_d,
                            "desglose_total": desglose_total_d,
                        }
                except Exception as e:
                    logger.error("Capa1.5: error construyendo plato dinámico: %s", e)

            # Caso B: query con múltiples items — intentar cada candidato ≥2 palabras
            elif len(items) > 1:
                from app.services.plato_constructor import crear_plato_dinamico
                from app.models.alimento import Alimento as _Alimento
                candidatos = [it for it in items if _es_candidato_plato_capa15(it)]
                for _cand in candidatos[:2]:  # máx 2 construcciones por query
                    try:
                        _pn = await crear_plato_dinamico(db, _cand)
                        if _pn:
                            _m = _pn.calcular_macros()
                            matched.append((
                                (_pn.id, _pn.nombre, _m["calorias"], _m["proteinas_g"],
                                 _m["carbohidratos_g"], _m["grasas_g"]),
                                1.0,
                            ))
                            # REGLA 1: ítem resuelto por CAPA 1.5 → quitar de no_resueltos
                            _cand_strip = re.sub(r'^(?:un|una|medio|media)\s+', '', _cand, flags=re.IGNORECASE)
                            _cand_norm = _norm_plato(_cand_strip)
                            _no_resueltos_c1 = [
                                n for n in _no_resueltos_c1
                                if _norm_plato(n) != _cand_norm
                            ]
                    except Exception as _e15:
                        logger.error("Capa1.5: error construyendo '%s': %s", _cand, _e15)

            if not matched:
                return None

        # Caso B2: ítems que no resolvió la búsqueda de platos → buscar en alimentos directamente.
        # Itera sobre _no_resueltos_c1 (nombres ya sin prefijo de cantidad) para capturar
        # tanto items de 1 palabra ("arroz") como de 2+ ("un durazno" → "durazno" normalizado).
        if _no_resueltos_c1:
            from app.models.alimento import Alimento as _Alimento
            for _simp in list(_no_resueltos_c1)[:3]:
                # Strip "tipo:nombre" format produced by the recipiente handler
                # e.g. "vaso:cocoa" → "cocoa" so it resolves against alimentos correctly
                _simp_limpio = re.sub(r'^[a-z]+:', '', _simp)
                _simp_n = _norm_plato(_simp_limpio)
                if not _simp_n or len(_simp_n) < 3:
                    continue
                _alim = (
                    db.query(_Alimento)
                    .filter(_Alimento.nombre_normalizado == _simp_n)
                    .first()
                ) or (
                    db.query(_Alimento)
                    .filter(_Alimento.nombre_normalizado.like(f"{_simp_n}%"))
                    .order_by(_Alimento.id)
                    .first()
                )
                if _alim and (_alim.calorias_100g or 0) > 0:
                    _gramos_std = 200.0 if any(
                        kw in _simp_n for kw in ("arroz", "pasta", "fideos", "pan", "yuca", "papa")
                    ) else 150.0
                    _kcal_a = round(float(_alim.calorias_100g) * _gramos_std / 100, 1)
                    _p_a    = round(float(_alim.proteina_100g or 0) * _gramos_std / 100, 1)
                    _c_a    = round(float(_alim.carbohidratos_100g or 0) * _gramos_std / 100, 1)
                    _g_a    = round(float(_alim.grasas_100g or 0) * _gramos_std / 100, 1)
                    # row[7]=True marca este ítem como alimento (no plato) para el desglose builder
                    matched.append((
                        (_alim.id, _alim.nombre, _kcal_a, _p_a, _c_a, _g_a, _gramos_std, True),
                        1.0,
                    ))
                    _no_resueltos_c1 = [n for n in _no_resueltos_c1 if _norm_plato(n) != _simp_n]
                    logger.info(
                        "Capa1 B2: '%s' → alimento '%s' %.0fg (%.0f kcal)",
                        _simp, _alim.nombre, _gramos_std, _kcal_a,
                    )

        # Guard anti-duplicado eliminado — el usuario puede registrar el mismo plato
        # múltiples veces en la misma sesión (desayuno, almuerzo, cena, o porciones extra)

        kcal = p_g = c_g = g_g = 0.0
        nombres = []
        for row, qty in matched:
            _row_kcal = float(row[2] or 0) * qty
            # CAMBIO 5 — Filtrar platos con kcal=0 en CAPA 1 (datos corruptos en BD)
            if _row_kcal <= 0:
                logger.warning(
                    "CAPA 1: plato '%s' (id=%s) tiene kcal=0 — omitido (ingredientes sin macros en BD)",
                    row[1], row[0],
                )
                continue
            kcal += _row_kcal
            p_g  += float(row[3] or 0) * qty
            c_g  += float(row[4] or 0) * qty
            g_g  += float(row[5] or 0) * qty
            nombres.append(row[1])

        # CAMBIO 5b — Si todos los platos fueron filtrados por kcal=0, forzar fallback
        if not nombres:
            logger.warning(
                "CAPA 1: todos los platos en '%s' rechazados por kcal=0 — cayendo a fallback",
                (mensaje or "")[:60],
            )
            return None

        desglose: list[str] = []
        desglose_total = ""
        from app.services.asistente_nutricion import _cargar_ingredientes_bd
        for _row, _qty in matched:
            # row[7]=True → alimento simple (Caso B2), NO llamar _cargar_ingredientes_bd
            # porque row[0] es alimento_id, no plato_id — causaría desglose del plato incorrecto
            _es_alimento_simple = len(_row) > 7 and _row[7] is True
            if not _es_alimento_simple:
                try:
                    _ing_list = _cargar_ingredientes_bd(db, _row[0])
                    if _ing_list:
                        desglose.extend(_ing_list)
                        continue
                except Exception:
                    pass
            # Fallback: línea única para alimentos simples (Caso B2) — row[6] = gramos
            _gramos_d = float(_row[6]) if len(_row) > 6 else 100.0
            _kcal_d   = round(float(_row[2] or 0), 1)
            _gramos_s = str(int(_gramos_d)) if _gramos_d == int(_gramos_d) else str(_gramos_d)
            _kcal_s   = str(int(_kcal_d)) if _kcal_d == int(_kcal_d) else str(_kcal_d)
            desglose.append(f"{_row[1]} {_gramos_s}g ({_kcal_s} kcal)")
        if desglose:
            desglose_total = (
                f"Total: {round(kcal, 1)} kcal"
                f" | P:{round(p_g, 1)}g | C:{round(c_g, 1)}g | G:{round(g_g, 1)}g"
            )

        if _no_resueltos_c1:
            logger.warning(
                "CAPA 1: %d ítem(s) no resuelto(s) en query '%s': %s",
                len(_no_resueltos_c1), (mensaje or "")[:60], _no_resueltos_c1,
            )

        return {
            "es_comida": True, "es_ejercicio": False,
            "calorias": round(kcal, 1), "proteinas_g": round(p_g, 1),
            "carbohidratos_g": round(c_g, 1), "grasas_g": round(g_g, 1),
            "fibra_g": 0, "azucar_g": 0, "sodio_mg": 0,
            "alimentos_detectados": nombres, "ejercicios_detectados": [],
            "calidad_nutricional": "Alta", "origen": "platos",
            "desglose_ingredientes": desglose,
            "desglose_total": desglose_total,
            "_no_resueltos": _no_resueltos_c1,
        }

    async def _aplicar_y_persistir(
        self,
        extraccion: Optional[dict],
        perfil,
        plan_hoy_data: dict,
        db: Session,
        ia_engine,
        mensaje: str,
    ) -> Dict[str, Any]:
        """Acumula en progreso_calorias y devuelve el dict de respuesta final."""
        if not extraccion:
            extraccion = await self._capa5_llm(mensaje, ia_engine, db)

        # Ingrediente ficticio detectado por CAPA 5 → respuesta directa sin persistir
        if extraccion and extraccion.get("_es_error_ficcion"):
            return extraccion

        if not extraccion or not extraccion.get("calorias"):
            return {
                "success": False, "tipo_detectado": "ninguno",
                "alimentos": [], "datos": {},
                "mensaje": "No pude identificar el alimento. ¿Puedes ser más específico? 🍽️",
            }

        # ── Hard Stop: bloquear antes de persistir ────────────────────────────
        error_hard_stop = _validar_hard_stop(extraccion)
        if error_hard_stop:
            return {
                "success": False,
                "tipo_detectado": "correccion_requerida",
                "alimentos": extraccion.get("alimentos_detectados", []),
                "datos": {},
                "mensaje": error_hard_stop,
            }

        hoy     = get_peru_date()
        momento = _inferir_momento_dia(mensaje)

        adv_prohibido = advertencia_alimentos_prohibidos(extraccion, perfil)

        # REGLA 3: escribir en comida_registros (trazabilidad) + actualizar progreso_calorias
        # MODO ADITIVO — preserva datos históricos del sistema anterior.
        # recalcular_progreso_diario() queda como herramienta de auditoría, NO se llama aquí.
        from app.services.trazabilidad import crear_comida_registros
        registros_creados = crear_comida_registros(
            client_id=perfil.id,
            fecha=hoy,
            extraccion=extraccion,
            texto_original=mensaje,
            db=db,
            momento=momento,
        )

        progreso = db.query(ProgresoCalorias).filter(
            ProgresoCalorias.client_id == perfil.id,
            ProgresoCalorias.fecha == hoy,
        ).first()
        if not progreso:
            progreso = ProgresoCalorias(client_id=perfil.id, fecha=hoy)
            db.add(progreso)
        _kcal_new = float(extraccion.get("calorias", 0) or 0)
        _prot_new = float(extraccion.get("proteinas_g", 0) or 0)
        _carb_new = float(extraccion.get("carbohidratos_g", 0) or 0)
        _gras_new = float(extraccion.get("grasas_g", 0) or 0)
        progreso.calorias_consumidas      = (progreso.calorias_consumidas or 0) + int(round(_kcal_new))
        progreso.proteinas_consumidas     = round((progreso.proteinas_consumidas or 0) + _prot_new, 1)
        progreso.carbohidratos_consumidos = round((progreso.carbohidratos_consumidos or 0) + _carb_new, 1)
        progreso.grasas_consumidas        = round((progreso.grasas_consumidas or 0) + _gras_new, 1)

        registrar_preferencias_alimentos(extraccion, perfil, db)
        db.commit()

        nombres     = extraccion.get("alimentos_detectados", [])
        nombre_str  = ", ".join(nombres) if nombres else "tu registro"
        alerta_macros = verificar_conflicto_macros(progreso, plan_hoy_data, perfil)
        momento_reloj = _inferir_momento_dia_por_hora()
        adv_horario   = _advertencia_rango_horario(extraccion.get("calorias", 0), momento)

        # Conflicto temporal: usuario pide "almuerzo" a las 11 PM, etc.
        adv_temporal: Optional[str] = None
        if momento and momento_reloj and momento != momento_reloj:
            adv_temporal = (
                f"⚠ Conflicto temporal: registras '{momento}' pero son horas de "
                f"'{momento_reloj}'. Registrado de todas formas."
            )

        # Veracidad calórica: registro con kcal absurdas alerta al usuario
        kcal_reg = float(extraccion.get("calorias", 0) or 0)
        adv_gula: Optional[str] = None
        if kcal_reg > _KCAL_MAX_REGISTRO:
            adv_gula = (
                f"⚠ Registro inusual: {kcal_reg:.0f} kcal en una sola comida "
                f"(límite de coherencia: {_KCAL_MAX_REGISTRO} kcal). "
                f"Verifica las cantidades."
            )

        # Techo físico de gramaje por ingrediente (2 kg)
        adv_gramaje = _validar_gramaje_extraccion(extraccion)

        msg_final = f"✅ Registré: {nombre_str} — {extraccion.get('calorias', 0)} kcal. ¡Buen trabajo!".replace("*", "")
        desglose = extraccion.get("desglose_ingredientes")
        desglose_total = extraccion.get("desglose_total", "")
        if desglose and extraccion.get("origen") in ("platos", "plato_dinamico"):
            lineas = "\n".join(f"  • {line}" for line in desglose)
            msg_final += (
                f"\n\n📊 Desglose (platos → plato_ingredientes → alimentos):\n"
                f"{lineas}"
            )
            if desglose_total:
                msg_final += f"\n  ━━ {desglose_total}"
            extras = extraccion.get("extras_nutricionales", [])
            if extras:
                msg_final += "\n" + "\n".join(f"  • {e}" for e in extras)
        elif not desglose and nombres and extraccion.get("calorias", 0) > 0:
            _macros_items = extraccion.get("alimentos_con_macros", [])
            if len(_macros_items) > 1:
                # Múltiples alimentos → un bullet por ítem con sus gramos reales
                _lineas = []
                for _m in _macros_items:
                    _g = int(round(_m.get("gramos", 100)))
                    _lineas.append(
                        f"  • {_m['nombre']} ({_g}g) — {round(_m['kcal'], 1)} kcal"
                        f" | P:{round(_m['prot_g'], 1)}g"
                        f" | C:{round(_m['carb_g'], 1)}g"
                        f" | G:{round(_m['gras_g'], 1)}g"
                    )
                msg_final += "\n\n📊 Detalle nutricional:\n" + "\n".join(_lineas)
            else:
                _d_kcal = round(float(extraccion.get("calorias", 0)), 1)
                _d_prot = round(float(extraccion.get("proteinas_g", 0)), 1)
                _d_carb = round(float(extraccion.get("carbohidratos_g", 0)), 1)
                _d_gras = round(float(extraccion.get("grasas_g", 0)), 1)
                _d_gramos = extraccion.get("porcion_g") or extraccion.get("gramos") or 100
                _gramos_txt = f" ({int(_d_gramos)}g)"
                msg_final += (
                    f"\n\n📊 Detalle nutricional:\n"
                    f"  • {nombre_str}{_gramos_txt} — {_d_kcal} kcal"
                    f" | P:{_d_prot}g | C:{_d_carb}g | G:{_d_gras}g"
                )
        if adv_prohibido:
            msg_final += f"\n\n{adv_prohibido}"
        if alerta_macros:
            msg_final += f"\n\n{alerta_macros}"
        if adv_horario:
            msg_final += f"\n\n{adv_horario}"
        if adv_temporal:
            msg_final += f"\n\n{adv_temporal}"
        if adv_gula:
            msg_final += f"\n\n{adv_gula}"
        if adv_gramaje:
            msg_final += f"\n\n{adv_gramaje}"
        warn = extraccion.get("_warn_cantidad") or extraccion.get("advertencia")
        if warn and ("kcal" in str(warn) or "correcto" in str(warn).lower()):
            msg_final += f"\n\n⚠️ {warn}"

        # ── Consideración técnica: trazabilidad del origen y sustituciones ────
        _origen = extraccion.get("origen", "bd")
        _origen_map = {
            "bd":           "catálogo CaloFit (BD local)",
            "nlp_extractor":"catálogo CaloFit (NLP extractor)",
            "usda":         "USDA FoodData Central (API externa)",
            "fatsecret":    "FatSecret (API externa)",
            "estimado":     "estimación LLM (Groq fallback)",
            "llm":          "estimación LLM (Groq fallback)",
            "postgres":     "catálogo CaloFit (BD local)",
        }
        _fuente_txt = _origen_map.get(_origen, _origen)
        _nombre_lower = " ".join(nombres).lower()
        _partes_ct: list[str] = [f"Macros obtenidos desde {_fuente_txt}."]
        if "pescado salpreso" in _nombre_lower or "ferreñafana" in mensaje.lower():
            _partes_ct.append(
                "Se utilizó el perfil nutricional de Pescado Salpreso (curado/seco) "
                "para mayor precisión; se omitieron Mayonesa y Queso Fresco por "
                "tratarse de una Causa Ferreñafana tradicional (variante norteña)."
            )
        if "pescado blanco frito" in _nombre_lower or "jalea" in mensaje.lower():
            _partes_ct.append(
                "Se aplicó el factor de absorción de aceite en fritura "
                "(Pescado Blanco Frito: 212 kcal/100g vs. 92 kcal fresco)."
            )
        if extraccion.get("_plato_dinamico"):
            _partes_ct.append("Plato construido dinámicamente mediante plato_constructor (origen='llm').")
        consideracion_tecnica = " ".join(_partes_ct)

        items_registrados = [
            {
                "nombre":          r.nombre_alimento,
                "kcal":            r.kcal,
                "tipo_resolucion": r.tipo_resolucion,
                "confianza":       r.confianza,
            }
            for r in registros_creados
        ]

        return {
            "success": True, "tipo_detectado": "comida",
            "alimentos": nombres,
            "items_registrados":  items_registrados,
            "items_no_resueltos": extraccion.get("_no_resueltos", []),
            "advertencia_prohibido": adv_prohibido,
            "advertencia_horario":   adv_horario,
            "advertencia_temporal":  adv_temporal,
            "advertencia_gula":      adv_gula,
            "advertencia_gramaje":   adv_gramaje,
            "alerta_macros":         alerta_macros,
            "consideracion_tecnica": consideracion_tecnica,
            "balance_actualizado": {
                "consumido": progreso.calorias_consumidas,
                "quemado":   progreso.calorias_quemadas,
            },
            "datos": {
                "calorias":        extraccion.get("calorias", 0),
                "proteinas_g":     extraccion.get("proteinas_g", 0),
                "carbohidratos_g": extraccion.get("carbohidratos_g", 0),
                "grasas_g":        extraccion.get("grasas_g", 0),
                "azucar_g":        extraccion.get("azucar_g", 0),
                "fibra_g":         extraccion.get("fibra_g", 0),
                "sodio_mg":        extraccion.get("sodio_mg", 0),
                "porcion_g":       extraccion.get("porcion_g") or extraccion.get("gramos"),
                "calidad":         extraccion.get("calidad_nutricional", "Media"),
            },
            "mensaje": msg_final,
        }

    async def _capa5_llm(self, mensaje: str, ia_engine, db: Session) -> Optional[dict]:
        """CAPA 5 (último recurso): Llama-3 estima macros para ingredientes REALES simples.
        Rechaza ficticios. Solo si las 4 capas anteriores fallaron."""
        try:
            import json
            prompt = (
                f"El usuario dijo: '{mensaje}'. Identifica el ALIMENTO SIMPLE (no platos complejos). "
                "IMPORTANTE: Si el alimento NO existe en la gastronomía real (ficticio, mitológico "
                "o imaginario como 'carne de unicornio', 'huevo de dragón'), devuelve exactamente: "
                "{\"nombre\":\"__DESCONOCIDO__\",\"calorias\":0,\"proteinas_g\":0,"
                "\"carbohidratos_g\":0,\"grasas_g\":0,\"porcion_g\":100}\n"
                "Si el alimento SÍ es real, devuelve SOLO JSON: "
                "{\"nombre\":\"...\",\"calorias\":0,\"proteinas_g\":0,"
                "\"carbohidratos_g\":0,\"grasas_g\":0,\"porcion_g\":100}"
            )
            resp = await ia_engine._llamar_groq(prompt, max_tokens=200, temp=0.05)
            data = json.loads(resp)

            nombre = data.get("nombre", "__DESCONOCIDO__")

            # Rechazar ingrediente ficticio o desconocido
            if nombre == "__DESCONOCIDO__" or not nombre.strip():
                return {
                    "success": False,
                    "tipo_detectado": "ingrediente_desconocido",
                    "alimentos": [],
                    "datos": {},
                    "mensaje": (
                        f"Lo siento, no reconozco el ingrediente indicado. "
                        f"Por favor usa un ingrediente real o regístralo manualmente "
                        f"en la base de datos."
                    ),
                    "_es_error_ficcion": True,
                }

            if data.get("calorias", 0) <= 0:
                return None

            porcion  = float(data.get("porcion_g") or 100)
            f        = 100.0 / max(1.0, porcion)
            nn       = _norm_al(nombre)

            existing = db.query(Alimento).filter(Alimento.nombre_normalizado == nn).first()
            if not existing:
                _kcal = round(data["calorias"] * f, 2)
                _prot = round(data.get("proteinas_g", 0) * f, 2)
                _carb = round(data.get("carbohidratos_g", 0) * f, 2)
                _gras = round(data.get("grasas_g", 0) * f, 2)
                _ok, _motivo = validar_macros_atwater(_kcal, _prot, _carb, _gras)
                if not _ok:
                    logger.warning("LLM alimento '%s' descartado — %s", nombre, _motivo)
                    return None
                db.add(Alimento(
                    nombre=nombre[:255], nombre_normalizado=nn[:255],
                    calorias_100g=_kcal,
                    proteina_100g=_prot,
                    carbohidratos_100g=_carb,
                    grasas_100g=_gras,
                    fuente="Groq (estimado)",
                    es_confiable=False,
                    pendiente_validacion=True,
                ))
                db.flush()

            return {
                "es_comida": True, "es_ejercicio": False,
                "calorias":        round(float(data["calorias"]), 1),
                "proteinas_g":     round(float(data.get("proteinas_g", 0)), 1),
                "carbohidratos_g": round(float(data.get("carbohidratos_g", 0)), 1),
                "grasas_g":        round(float(data.get("grasas_g", 0)), 1),
                "fibra_g": 0, "azucar_g": 0, "sodio_mg": 0,
                "alimentos_detectados": [nombre], "ejercicios_detectados": [],
                "calidad_nutricional": "Media", "origen": "llm",
            }
        except Exception as e:
            logger.error("CAPA 5 LLM error: %s", e)
        return None


registro_comida_handler = RegistroComidaHandler()


# ── Helper de caché de cards ──────────────────────────────────────────────────

def registrar_desde_cache(payload: dict, perfil, db: Session) -> dict:
    """
    Registra comida o ejercicio usando los datos cacheados de una card del chat.
    Compartido por consultar() y registrar_por_nlp() cuando reciben un consulta_id.
    """
    from app.core.utils import get_peru_date
    from app.services.asistente_ejercicio import (
        es_payload_ejercicio,
        registrar_ejercicio_desde_payload_tarjeta,
    )
    from app.services.asistente_nutricion import registrar_comida_desde_payload_tarjeta

    hoy      = get_peru_date()
    progreso = db.query(ProgresoCalorias).filter(
        ProgresoCalorias.client_id == perfil.id, ProgresoCalorias.fecha == hoy
    ).first()
    if not progreso:
        progreso = ProgresoCalorias(client_id=perfil.id, fecha=hoy)
        db.add(progreso)

    if es_payload_ejercicio(payload):
        meta = registrar_ejercicio_desde_payload_tarjeta(payload, perfil, progreso, db)
        msg  = f"✅ Registré: {meta['nombre']} — {meta['calorias']:.0f} kcal quemadas."
    else:
        meta = registrar_comida_desde_payload_tarjeta(payload, perfil, progreso, db)
        msg  = f"✅ Registré: {meta['nombre']} — {meta['calorias']:.0f} kcal."
        crear_comida_registros(
            client_id=perfil.id,
            fecha=hoy,
            extraccion={
                "alimentos_detectados": [payload.get("nombre", meta.get("nombre", ""))],
                "calorias":        meta.get("calorias", 0),
                "proteinas_g":     meta.get("proteinas_g", 0),
                "carbohidratos_g": meta.get("carbohidratos_g", 0),
                "grasas_g":        meta.get("grasas_g", 0),
                "origen":          payload.get("origen", "platos"),
            },
            texto_original="[cache]",
            db=db,
            momento=None,
        )

    db.commit()
    tipo = meta["tipo_detectado"]
    key  = "alimentos" if tipo == "comida" else "ejercicios"
    return {
        "success": True, "tipo_detectado": tipo, key: [meta["nombre"]],
        "balance_actualizado": {
            "consumido": progreso.calorias_consumidas,
            "quemado":   progreso.calorias_quemadas,
        },
        "datos": {
            "calorias":        meta.get("calorias", 0),
            "proteinas_g":     meta.get("proteinas_g", 0),
            "carbohidratos_g": meta.get("carbohidratos_g", 0),
            "grasas_g":        meta.get("grasas_g", 0),
        },
        "mensaje": msg,
    }
