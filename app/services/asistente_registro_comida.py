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
from app.models.preferencias import PreferenciaAlimento
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
    # NFC: normaliza tildes para que 'plátano' (NFD del teclado móvil)
    # coincida con 'plátano' (NFC guardado en BD).
    s = unicodedata.normalize("NFC", (s or "")).strip().lower()
    s = re.sub(r"\[.*?\]", "", s).split("[")[0].strip()
    s = re.sub(r"[^a-z0-9áéíóúüñ\s]", " ", s, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", s).strip()


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
    """
    Mapea la hora actual del servidor (Perú, UTC-5) al momento del día.
    Usado para validar conflictos entre lo que el usuario dice y la hora real.
    """
    import datetime
    hora = (datetime.datetime.utcnow() - datetime.timedelta(hours=5)).hour
    if  5 <= hora <=  9: return "desayuno"
    if 10 <= hora <= 14: return "almuerzo"
    if 15 <= hora <= 17: return "merienda"
    if 18 <= hora <= 21: return "cena"
    return "snack"  # 22:00-04:59 → snack nocturno


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
    t = _RE_CANTIDAD_INICIO.sub("", t).strip()
    t = re.sub(r"(?i)\b(otra\s+vez|de\s+nuevo|nuevamente)\b", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return [p.strip() for p in re.split(r"\s+y\s+|,|;", t) if p.strip()]


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
        # No segmentar si el nombre completo ya existe como plato propio en BD
        norm_full = _norm_plato(item)
        if db.execute(
            _sql_t("SELECT 1 FROM platos WHERE nombre_normalizado = :n LIMIT 1"),
            {"n": norm_full},
        ).fetchone():
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
        existe = db.execute(
            _sql_t("SELECT 1 FROM platos WHERE nombre_normalizado = :n LIMIT 1"),
            {"n": norm_prefijo},
        ).fetchone()
        if existe:
            logger.info("Segmentación plato+acompañamiento: '%s' → ['%s', '%s']", item, prefijo, sufijo)
            resultado.extend([prefijo, sufijo])
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
                elif len((_nombre_c0 or "").split()) >= 2:
                    # Platos multi-palabra (≥2 tokens): CAPA 0 NO tiene autoridad final.
                    # Se defiere a CAPA 1/1.5 que consulta ingredientes reales en BD.
                    # CAPA 0 queda como fallback por si el catálogo tampoco lo encuentra.
                    _capa0_fallback = capa0_result
                else:
                    # Alimento de una sola palabra: CAPA 0 es suficientemente precisa
                    pre_extraccion = capa0_result

        # Porción de lata (capa especial antes del catálogo)
        if not pre_extraccion and _msg_tiene_porcion_lata(mensaje):
            pre_extraccion = self._capa_lata(mensaje, db)

        # CAPA 1: catálogo platos (incluye CAPA 1.5 — plato_constructor)
        if not pre_extraccion:
            intento_platos = await self._capa1_platos(mensaje, perfil, db)
            if intento_platos:
                if intento_platos.get("skip_duplicate"):
                    return {
                        "success": True,
                        "tipo_detectado": "comida",
                        "alimentos": [intento_platos.get("nombre")],
                        "advertencia_prohibido": None,
                        "alerta_macros": None,
                        "balance_actualizado": {},
                        "datos": {},
                        "mensaje": (
                            f"🧾 Parece que ya registré \"{intento_platos.get('nombre')}\" hace poco. "
                            "Si fue otra porción, dime por ejemplo: \"comí 2\"."
                        ),
                    }
                pre_extraccion = intento_platos
            elif _capa0_fallback:
                # CAPA 1 no encontró el plato multi-palabra → usar estimación CAPA 0
                logger.info(
                    "CAPA 1 no resolvió '%s' — usando estimación CAPA 0 como fallback",
                    _nombre_c0,
                )
                pre_extraccion = _capa0_fallback

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
            from app.services.nlp_food_extractor import NLPFoodExtractor, contiene_modificador_ficticio
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

            if extractor._es_negacion(mensaje):
                return {
                    "_final": True, "success": True,
                    "tipo_detectado": "ninguno", "alimentos": [], "datos": {},
                    "mensaje": "Entendido, no registré ningún alimento. Si comiste algo, cuéntame 😊",
                }

            resultado = await extractor.extraer(mensaje)

            if resultado and resultado.calorias_total > 0:
                adv = resultado.advertencia
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

        msg_l          = (mensaje or "").lower()
        explicit_repeat = any(x in msg_l for x in ("otra vez", "de nuevo", "nuevamente"))

        _SQL = (
            "SELECT p.id, p.nombre,"
            " SUM(a.calorias_100g * pi2.gramos / 100.0),"
            " SUM(a.proteina_100g * pi2.gramos / 100.0),"
            " SUM(a.carbohidratos_100g * pi2.gramos / 100.0),"
            " SUM(a.grasas_100g * pi2.gramos / 100.0)"
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
                if _score_hist >= 0.80:
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
                        return {
                            "calorias":        round(float(_row_hist[2] or 0), 1),
                            "proteinas_g":     round(float(_row_hist[3] or 0), 1),
                            "carbohidratos_g": round(float(_row_hist[4] or 0), 1),
                            "grasas_g":        round(float(_row_hist[5] or 0), 1),
                            "fibra_g": 0, "azucar_g": 0, "sodio_mg": 0,
                            "es_comida": True, "es_ejercicio": False,
                            "alimentos_detectados": [_row_hist[1]],
                            "ejercicios_detectados": [],
                            "calidad_nutricional": "Alta",
                            "origen": "historial_recomendacion",
                        }
        except Exception as _eh:
            logger.debug("Fast-path historial error (no crítico): %s", _eh)


        matched: List[tuple] = []
        # REGLA 1: trackear ítems que CAPA 1 no pudo resolver (evita pérdida silenciosa)
        _no_resueltos_c1: List[str] = []

        # REGLA 2: prefijos de recipiente que deben quitarse antes de buscar en BD
        _RE_PREFIJO_RECIPIENTE = re.compile(
            r'^(?:plato\s+de|porci[oó]n\s+de|medio\s+plato\s+de|media\s+porci[oó]n\s+de)\s+',
            re.IGNORECASE,
        )

        for it in items:
            m = re.match(
                r"^\s*((?:\d+(?:[.,]\d+)?)|uno|una|un|dos|tres|cuatro|cinco|medio|media)\s+(.*)$",
                it, re.IGNORECASE,
            )
            qty, name_part = 1.0, it
            if m:
                qty, name_part = _parse_qty(m.group(1)), m.group(2).strip()

            # REGLA 2: eliminar prefijo de recipiente antes de buscar
            name_part = _RE_PREFIJO_RECIPIENTE.sub("", name_part).strip()

            nn = _norm_plato(name_part)
            if not nn or len(nn) < 4:
                continue

            row = db.execute(_sql(_SQL), {"q": nn}).fetchone()
            if not row:
                # Fallback similitud coseno sobre los últimos 250 platos
                cands = db.query(Plato.id, Plato.nombre, Plato.nombre_normalizado).order_by(Plato.id.desc()).limit(250).all()
                best_id, best_score = None, 0.0
                for pid, pnombre, pnn in cands:
                    rn = _norm_plato(pnn or pnombre or "")
                    score = difflib.SequenceMatcher(a=nn, b=rn).ratio()
                    if score > best_score:
                        best_score, best_id = score, pid
                if best_id and best_score >= 0.92:
                    row = db.execute(_sql(
                        "SELECT p.id, p.nombre,"
                        " SUM(a.calorias_100g * pi2.gramos / 100.0),"
                        " SUM(a.proteina_100g * pi2.gramos / 100.0),"
                        " SUM(a.carbohidratos_100g * pi2.gramos / 100.0),"
                        " SUM(a.grasas_100g * pi2.gramos / 100.0)"
                        " FROM platos p JOIN plato_ingredientes pi2 ON pi2.plato_id = p.id"
                        " JOIN alimentos a ON a.id = pi2.alimento_id WHERE p.id = :pid"
                        " GROUP BY p.id, p.nombre"
                    ), {"pid": best_id}).fetchone()
            if row:
                matched.append((row, qty))
            else:
                # REGLA 1: ítem no resuelto por CAPA 1 — registrar para no perderlo silenciosamente
                _no_resueltos_c1.append(name_part)

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
                # Solo intentar si el texto limpio tiene >= 3 palabras y contiene "con"
                if " con " in _msg_clean and len(_msg_clean.split()) >= 3:
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
                            _cand_norm = _norm_plato(_cand)
                            _no_resueltos_c1 = [
                                n for n in _no_resueltos_c1
                                if _norm_plato(n) != _cand_norm
                            ]
                    except Exception as _e15:
                        logger.error("Capa1.5: error construyendo '%s': %s", _cand, _e15)

            if not matched:
                return None

        # Guard anti-duplicado 10 min
        try:
            last = (
                db.query(PreferenciaAlimento)
                .filter(PreferenciaAlimento.client_id == perfil.id)
                .order_by(PreferenciaAlimento.ultima_vez.desc()).first()
            )
            if last and last.ultima_vez:
                from datetime import timezone as _tz
                now = datetime.now()
                try:
                    now = datetime.now(_tz.utc) if getattr(last.ultima_vez, "tzinfo", None) else now
                except Exception:
                    pass
                delta_min = (now - last.ultima_vez).total_seconds() / 60.0
                if delta_min <= 10:
                    is_same = any(_norm_plato(r[1] or "") == _norm_plato(last.alimento or "") for r, _ in matched)
                    qty_any = any(q > 1.01 for _, q in matched)
                    if is_same and not qty_any and not explicit_repeat:
                        return {"skip_duplicate": True, "nombre": last.alimento}
        except Exception:
            pass

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

        desglose = []
        desglose_total = ""
        if len(matched) == 1:
            from app.services.asistente_nutricion import _cargar_ingredientes_bd
            try:
                desglose = _cargar_ingredientes_bd(db, matched[0][0][0])
                desglose_total = (
                    f"Total: {round(kcal, 1)} kcal"
                    f" | P:{round(p_g, 1)}g | C:{round(c_g, 1)}g | G:{round(g_g, 1)}g"
                )
            except Exception:
                pass

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
            resp = await ia_engine.consultar_groq(prompt, sistema="Eres nutricionista. Solo JSON, sin texto extra.")
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
