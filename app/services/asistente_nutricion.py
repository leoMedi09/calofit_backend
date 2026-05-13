"""
Lógica de nutrición del asistente (tarjetas de comida, macros, fuzzy de comidas recientes).

Separado de ``asistente_ejercicio.py`` para poder cambiar comidas sin tocar entrenamiento.
"""
from __future__ import annotations

import asyncio
import difflib
import json
import re
import unicodedata
import urllib.parse
import urllib.request
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import func as sql_func
from sqlalchemy.orm import Session

from app.core.cache import add_user_recent_meal, get_user_recent_meals, set_consulta_cached
from app.core.logging_config import get_logger
from app.core.macros_diarios import macros_desde_calorias_pct_clasico
from app.core.utils import parsear_macros_de_texto, inferir_momento_dia_peru
from app.models.preferencias import PreferenciaAlimento
from app.models.alimento import Alimento
from app.services.nutricional_result import validar_macros_atwater
from app.services.calculador_dieta import CalculadorDietaAutomatica

logger = get_logger("nutricion")


# ─── Normalización de texto (sin tildes, minúsculas, sin símbolos) ────────────
def _norm(texto: str) -> str:
    """Normaliza texto: minúsculas, sin tildes, sin caracteres especiales."""
    s = (texto or "").strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    return re.sub(r"\s{2,}", " ", s).strip()


# ─── Helpers para componentes_bd ─────────────────────────────────────────────

def _norm_ing(s: str) -> str:
    s = unicodedata.normalize("NFD", s.lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9 ]", " ", s).strip()


def coherencia_proteina_platos(nombre_query_norm: str, nombre_candidato_norm: str) -> bool:
    """
    Evita que la similitud de texto empareje platos con proteína distinta.
    Cubre: pescado, pollo, pavo, pato, res/lomo, cerdo, cordero.

    Regla crítica: "arroz con pato" ≠ "arroz con pollo" aunque el score
    de similitud sea alto. El sistema debe activar plato_constructor en su lugar.
    """
    q = (nombre_query_norm or "").lower()
    c = (nombre_candidato_norm or "").lower()
    if not q or not c:
        return True

    # ── Detectar proteína principal en query y candidato ─────────────────────
    _FISH = (
        "pescado", "salmon", "salmón", "atun", "atún", "tilapia", "merluza",
        "bacalao", "trucha", "lenguado", "caballa", "lisa", "mero", "tollo",
        "anchoveta", "bonito", "salpreso",
    )
    # Platos cuya proteína base es pescado aunque no figure explícitamente en el nombre
    _FISH_IMPLICIT = ("ferrenafana", "ferreñafana", "chinguirito", "tiradito")
    fish_q = any(w in q for w in _FISH) or any(w in q for w in _FISH_IMPLICIT)
    fish_c = any(w in c for w in _FISH) or any(w in c for w in _FISH_IMPLICIT)

    pollo_q = "pollo" in q or ("pechuga" in q and not fish_q)
    pollo_c = "pollo" in c or ("pechuga" in c and not fish_c)

    pavo_q = "pavo" in q
    pavo_c = "pavo" in c

    # pato: proteína distinta de pollo — bloquear cruce entre sí
    pato_q = "pato" in q
    pato_c = "pato" in c

    res_q = any(w in q for w in ("res", "lomo", "ternera", "bistec", "carne de res"))
    res_c = any(w in c for w in ("res", "lomo", "ternera", "bistec", "carne de res"))

    cerdo_q = any(w in q for w in ("cerdo", "chancho", "chicharron", "chicharrón", "tocino"))
    cerdo_c = any(w in c for w in ("cerdo", "chancho", "chicharron", "chicharrón", "tocino"))

    # ── Reglas de incompatibilidad ────────────────────────────────────────────
    # Pescado ≠ carnes
    if fish_q and (pollo_c or pavo_c or pato_c or cerdo_c or res_c) and not fish_c:
        return False
    if fish_c and (pollo_q or pavo_q or pato_q or cerdo_q or res_q) and not fish_q:
        return False

    # Pollo ≠ pavo, pato, res, cerdo
    if pollo_q and (pavo_c or pato_c) and not pollo_c:
        return False
    if pollo_c and (pavo_q or pato_q) and not pollo_c:
        return False

    # Pavo ≠ pollo, pato
    if pavo_q and (pollo_c or pato_c) and not pavo_c:
        return False
    if pavo_c and (pollo_q or pato_q) and not pavo_c:
        return False

    # Pato ≠ pollo, pavo, res, cerdo  ← REGLA NUEVA (raíz del bug Arroz con Pato → Pollo)
    if pato_q and (pollo_c or pavo_c or res_c or cerdo_c) and not pato_c:
        return False
    if pato_c and (pollo_q or pavo_q or res_q or cerdo_q) and not pato_q:
        return False

    return True


_ING_STOPWORDS = {"blanco", "fresco", "fresca", "cocido", "cocida", "natural",
                  "ligero", "ligera", "variado", "variada", "mixta", "integral"}

_GRAMOS_POR_UNIDAD = {
    "cucharada": 15.0, "cucharadita": 5.0, "taza": 240.0,
    "vaso": 250.0, "rebanada": 35.0, "loncha": 25.0,
}
_GRAMOS_POR_ITEM = {
    "huevo": 55.0, "huevos": 55.0, "aguacate": 150.0, "palta": 150.0,
    "naranja": 130.0, "manzana": 150.0, "platano": 120.0, "banana": 120.0,
    "lechuga": 80.0, "limon": 60.0,
}


_ABREV_MEDIDA = {
    "cda": 15.0, "cdas": 15.0, "c.d.a": 15.0,
    "cdta": 5.0, "cdtas": 5.0, "c.d.t.a": 5.0,
    "taza": 240.0, "tazas": 240.0,
    "vaso": 250.0, "vasos": 250.0,
    "ml": 1.0,
}


def _es_nombre_condimento_traza(nombre_norm: str) -> bool:
    """
    Sal, pimienta y especias típicas en pizca / cdta: sin aporte nutricional
    relevante en la app (no usar kcal/100g de polvo seco al bulto).
    """
    if not nombre_norm:
        return False
    # ^sal\\b evita 'salmon', 'salchicha'
    if re.match(r"^sal\b", nombre_norm):
        return True
    if re.match(r"^pimienta\b", nombre_norm):
        return True
    if re.match(r"^oregano\b", nombre_norm):
        return True
    if re.match(r"^comino\b", nombre_norm):
        return True
    if re.match(r"^(hoja de )?laurel\b", nombre_norm):
        return True
    return False


def _parse_ing_gramos(texto: str) -> Optional[Tuple[str, float]]:
    """Extrae (nombre_norm, gramos) de una cadena de ingrediente del LLM."""

    def _emit(nombre_norm: str, gramos_val: float) -> Optional[Tuple[str, float]]:
        if _es_nombre_condimento_traza(nombre_norm):
            return None
        return nombre_norm, gramos_val

    t = texto.strip()
    # NNNg nombre (kcal)
    m = re.match(r"(\d+(?:\.\d+)?)\s*g\s+(.+?)(?:\s*\(|$)", t, re.IGNORECASE)
    if m:
        return _emit(_norm_ing(m.group(2).strip()), float(m.group(1)))
    # N abrev nombre   ← fix: "1 cda aceite de oliva" → 15g
    m = re.match(
        r"(\d+(?:\.\d+)?)\s+(cdas?|cdtas?|c\.d\.a\.?|c\.d\.t\.a\.?|tazas?|vasos?|ml)\s+(?:de\s+)?(.+?)(?:\s*\(|$)",
        t, re.IGNORECASE
    )
    if m:
        cant  = float(m.group(1))
        abrev = m.group(2).lower().rstrip(".")
        nombre = _norm_ing(m.group(3).strip())
        gramos = _ABREV_MEDIDA.get(abrev, 15.0)
        return _emit(nombre, cant * gramos)
    # N unidad de nombre (palabras completas)
    m = re.match(
        r"(\d+(?:\.\d+)?)\s+(cucharadas?|cucharaditas?|tazas?|vasos?|rebanadas?|lonchas?)\s+(?:de\s+)?(.+?)(?:\s*\(|$)",
        t, re.IGNORECASE
    )
    if m:
        cant = float(m.group(1))
        unidad = _norm_ing(m.group(2))
        nombre = _norm_ing(m.group(3).strip())
        for k, g in _GRAMOS_POR_UNIDAD.items():
            if k in unidad:
                return _emit(nombre, cant * g)
        return _emit(nombre, cant * 15.0)
    # N nombre (Xg, Y kcal)
    m = re.match(r"(\d+(?:\.\d+)?)\s+(.+?)\s*\((\d+)\s*g[,\s]", t, re.IGNORECASE)
    if m:
        return _emit(_norm_ing(m.group(2).strip()), float(m.group(3)))
    # N nombre  (fallback: cantidad × peso unitario o 100g)
    m = re.match(r"(\d+(?:\.\d+)?)\s+(.+?)(?:\s*\(|$)", t, re.IGNORECASE)
    if m:
        cant   = float(m.group(1))
        nombre = _norm_ing(m.group(2).strip())
        # Rechazar si el "nombre" parece una abreviación de medida suelta
        if nombre in _ABREV_MEDIDA:
            return None
        for item, grms in _GRAMOS_POR_ITEM.items():
            if item in nombre:
                return _emit(nombre, cant * grms)
        return _emit(nombre, cant * 100.0)
    return None


def _resolver_alimento_en_bd(db: Session, nombre_norm: str):
    """Busca un alimento por nombre o alias en la BD. Devuelve objeto Alimento o None."""
    from app.models.alimento import Alimento
    from app.models.alimento_alias import AlimentoAlias

    # Exact match
    a = db.query(Alimento).filter(Alimento.nombre_normalizado == nombre_norm).first()
    if a:
        return a
    # Alias exact
    alias = db.query(AlimentoAlias).filter(AlimentoAlias.alias == nombre_norm).first()
    if alias:
        return db.query(Alimento).filter(Alimento.id == alias.alimento_id).first()
    # Contains — prefiere el candidato con mayor similitud al nombre buscado
    words = [w for w in nombre_norm.split() if len(w) >= 5 and w not in _ING_STOPWORDS]
    for w in words:
        candidates = (
            db.query(Alimento)
            .filter(Alimento.nombre_normalizado.like(f"%{w}%"))
            .limit(15)
            .all()
        )
        if candidates:
            best = max(
                candidates,
                key=lambda c: difflib.SequenceMatcher(
                    None, nombre_norm, c.nombre_normalizado or ""
                ).ratio(),
            )
            return best
        alias_candidates = (
            db.query(AlimentoAlias)
            .filter(AlimentoAlias.alias_normalizado.like(f"%{w}%"))
            .limit(10)
            .all()
        )
        if alias_candidates:
            best_alias = max(
                alias_candidates,
                key=lambda al: difflib.SequenceMatcher(
                    None, nombre_norm, al.alias_normalizado or ""
                ).ratio(),
            )
            return db.query(Alimento).filter(Alimento.id == best_alias.alimento_id).first()
    return None




# ─── Fallback USDA + Groq para ingredientes desconocidos ─────────────────────

_USDA_API_KEY = "DEMO_KEY"
try:
    from app.core.config import settings as _cfg
    _USDA_API_KEY = getattr(_cfg, "USDA_API_KEY", "DEMO_KEY") or "DEMO_KEY"
except Exception:
    pass


def _buscar_usda_sync(nombre_en: str) -> Optional[dict]:
    """Consulta USDA FoodData Central y devuelve macros por 100g o None."""
    if len(nombre_en.split()) > 5:
        return None
    params = urllib.parse.urlencode({
        "query":    nombre_en,
        "api_key":  _USDA_API_KEY,
        "pageSize": 1,
        "dataType": "Foundation,SR Legacy",
    })
    url = f"https://api.nal.usda.gov/fdc/v1/foods/search?{params}"
    NUTRIENT_IDS = {1008: "calorias_100g", 1003: "proteina_100g",
                    1005: "carbohidratos_100g", 1004: "grasas_100g",
                    1079: "fibra_100g", 2000: "azucar_100g"}
    try:
        with urllib.request.urlopen(url, timeout=8) as resp:
            data = json.loads(resp.read().decode())
        foods = data.get("foods", [])
        if not foods:
            return None
        food = foods[0]
        n_lower = nombre_en.lower()
        desc    = food.get("description", "").lower()
        if not any(p in desc for p in n_lower.split() if len(p) >= 4):
            return None
        macros: dict = {}
        for n in food.get("foodNutrients", []):
            nid = n.get("nutrientId")
            if nid in NUTRIENT_IDS:
                macros[NUTRIENT_IDS[nid]] = round(float(n.get("value", 0)), 2)
        return macros if "calorias_100g" in macros else None
    except Exception:
        return None


def _buscar_fatsecret_sync(nombre_es: str) -> Optional[dict]:
    """Consulta FatSecret con porcion_g=100 para obtener macros base por 100g."""
    try:
        from app.services.fatsecret_client import get_fatsecret_client
        fs = get_fatsecret_client()
        if not fs:
            return None
        raw = fs.lookup_macros(nombre_es, porcion_g=100)
        if not raw or raw.get("calorias", 0) <= 0:
            return None
        return {
            "calorias_100g":      round(float(raw["calorias"]), 2),
            "proteina_100g":      round(float(raw["proteinas"]), 2),
            "carbohidratos_100g": round(float(raw["carbohidratos"]), 2),
            "grasas_100g":        round(float(raw["grasas"]), 2),
        }
    except Exception:
        return None


def _buscar_colision_local(
    db: Session, nombre_norm: str
) -> Optional[Tuple[Any, float]]:
    """
    Detecta si existe un alimento similar en BD antes de insertar duplicado.
    Retorna (Alimento, similitud) o None.
    Limitado a candidatos con misma primera palabra para eficiencia.
    """
    first_word = nombre_norm.split()[0] if nombre_norm else ""
    if len(first_word) < 3:
        return None
    candidates = (
        db.query(Alimento)
        .filter(Alimento.nombre_normalizado.like(f"{first_word}%"))
        .limit(30)
        .all()
    )
    best, best_score = None, 0.0
    for c in candidates:
        score = difflib.SequenceMatcher(
            None, nombre_norm, c.nombre_normalizado or ""
        ).ratio()
        if score > best_score:
            best_score, best = score, c
    if best and best_score >= 0.75:
        return (best, best_score)
    return None


async def _buscar_o_crear_alimento_async(
    db: Session, nombre_norm: str, nombre_es: str
) -> Optional[Alimento]:
    """
    Busca un alimento en BD; si no existe, intenta USDA y luego Groq.
    Guarda el resultado en `alimentos` para consultas futuras.
    Retorna el objeto Alimento o None si todo falla.
    """
    # FIREWALL ANTI-RECURSIVO: gestiona únicamente la tabla `alimentos`.
    # Nunca llama a plato_constructor.py ni crea registros en `platos`.
    # 1. Buscar en BD
    alim = _resolver_alimento_en_bd(db, nombre_norm)
    if alim:
        return alim

    # 1b. Condimentos en porción típica (pizca/cdta): sin carga USDA/Groq
    if _es_nombre_condimento_traza(nombre_norm):
        if re.match(r"^sal\b", nombre_norm):
            sal_row = (
                db.query(Alimento)
                .filter(Alimento.nombre_normalizado == "sal comun")
                .first()
            )
            if sal_row:
                return sal_row
        traza_row = (
            db.query(Alimento)
            .filter(Alimento.nombre_normalizado == "especias condimento traza")
            .first()
        )
        if traza_row:
            return traza_row

    # 2. USDA (en hilo para no bloquear el event loop)
    macros = await asyncio.to_thread(_buscar_usda_sync, nombre_es)
    fuente = "USDA (auto-aprendido)"

    # 2.5. FatSecret si USDA no devolvió resultados
    if not macros:
        macros = await asyncio.to_thread(_buscar_fatsecret_sync, nombre_es)
        if macros:
            fuente = "FatSecret (auto-aprendido)"

    # 3. Groq si USDA y FatSecret fallaron
    if not macros:
        try:
            from app.services.ia_service import ia_engine
            prompt = (
                f"Eres un nutricionista experto. Dame los macronutrientes por 100g de: '{nombre_es}'.\n"
                f"Responde SOLO JSON válido:\n"
                f'{{"calorias_100g":<n>,"proteina_100g":<n>,"carbohidratos_100g":<n>,"grasas_100g":<n>}}'
            )
            resp = await ia_engine._llamar_groq(prompt=prompt, max_tokens=120, temp=0.1)
            resp = re.sub(r"```(?:json)?", "", resp).strip().strip("`")
            m = re.search(r"\{.*\}", resp, re.DOTALL)
            if m:
                data = json.loads(m.group(0))
                if float(data.get("calorias_100g", 0)) > 0:
                    macros = {
                        "calorias_100g":      round(float(data.get("calorias_100g", 0)), 1),
                        "proteina_100g":      round(float(data.get("proteina_100g", 0)), 1),
                        "carbohidratos_100g": round(float(data.get("carbohidratos_100g", 0)), 1),
                        "grasas_100g":        round(float(data.get("grasas_100g", 0)), 1),
                        "fibra_100g": 0.0, "azucar_100g": 0.0,
                    }
                    fuente = "Groq (estimado)"
        except Exception as e:
            logger.error("Groq fallback falló para '%s': %s", nombre_es, e)

    if not macros:
        return None

    # 4. Verificar colisión antes de insertar para no crear duplicados
    colision = _buscar_colision_local(db, nombre_norm)
    if colision:
        alim_existente, sim = colision
        if sim >= 0.85:
            return alim_existente
        elif sim >= 0.70:
            try:
                from app.models.alimento_alias import AlimentoAlias
                alias = AlimentoAlias(
                    alimento_id=alim_existente.id,
                    alias=nombre_es[:255],
                    alias_normalizado=nombre_norm[:255],
                )
                db.add(alias)
                db.commit()
                logger.info(
                    "Alias creado '%s' → '%s' (sim=%.2f)",
                    nombre_norm, alim_existente.nombre_normalizado, sim,
                )
            except Exception:
                db.rollback()
            return alim_existente

    # 4b. Guardar en alimentos para próximas consultas.
    # Alimentos estimados por Groq se marcan como no confiables y pendientes de validación
    # para que un administrador pueda revisarlos antes de confiar en ellos.
    _es_groq = fuente == "Groq (estimado)"
    try:
        _ok, _motivo = validar_macros_atwater(
            macros.get("calorias_100g", 0),
            macros.get("proteina_100g", 0),
            macros.get("carbohidratos_100g", 0),
            macros.get("grasas_100g", 0),
        )
        if not _ok:
            logger.warning(
                "Alimento '%s' (fuente=%s) descartado por Atwater — %s",
                nombre_es, fuente, _motivo,
            )
            return None
        nuevo = Alimento(
            nombre=nombre_es[:200],
            nombre_normalizado=nombre_norm[:200],
            calorias_100g      = macros.get("calorias_100g", 0),
            proteina_100g      = macros.get("proteina_100g", 0),
            carbohidratos_100g = macros.get("carbohidratos_100g", 0),
            grasas_100g        = macros.get("grasas_100g", 0),
            fibra_100g         = macros.get("fibra_100g", 0),
            azucar_100g        = macros.get("azucar_100g", 0),
            categoria="Otros",
            fuente=fuente,
            es_confiable        = not _es_groq,
            pendiente_validacion= _es_groq,
        )
        db.add(nuevo)
        db.commit()
        db.refresh(nuevo)
        logger.info(
            "Nuevo alimento guardado desde %s: '%s'%s",
            fuente, nombre_es,
            " [PENDIENTE VALIDACIÓN]" if _es_groq else "",
        )
        return nuevo
    except Exception as e:
        db.rollback()
        logger.error("Error guardando alimento '%s': %s", nombre_es, e)
        return None


async def _construir_componentes_bd_async(
    db: Session, ingredientes: List[str]
) -> Tuple[Optional[list], Optional[dict]]:
    """
    Como _construir_componentes_bd pero con fallback USDA → Groq para
    ingredientes no encontrados en la BD local.
    """
    componentes = []
    total = {"calorias": 0.0, "proteinas_g": 0.0, "carbohidratos_g": 0.0, "grasas_g": 0.0}

    for ing_str in ingredientes:
        parsed = _parse_ing_gramos(str(ing_str))
        if not parsed:
            continue
        nombre_norm, gramos = parsed

        # Recuperar nombre "legible" quitando el formato "Xg ... (Y kcal)"
        nombre_es_raw = re.sub(r"\(.*?\)", "", str(ing_str)).strip()
        nombre_es_raw = re.sub(r"^\d+(?:\.\d+)?\s*g\s+", "", nombre_es_raw).strip()
        nombre_es_raw = re.sub(r"^\d+(?:\.\d+)?\s+(?:cucharadas?|cucharaditas?|tazas?|vasos?)\s+(?:de\s+)?", "", nombre_es_raw, flags=re.IGNORECASE).strip()
        nombre_es = nombre_es_raw or nombre_norm

        alim = await _buscar_o_crear_alimento_async(db, nombre_norm, nombre_es)
        if alim:
            factor = gramos / 100.0
            componentes.append({
                "alimento_id": alim.id,
                "nombre":      alim.nombre,
                "gramos":      gramos,
                "kcal":        round(alim.calorias_100g * factor, 1),
            })
            total["calorias"]        += round(alim.calorias_100g * factor, 1)
            total["proteinas_g"]     += round(alim.proteina_100g * factor, 1)
            total["carbohidratos_g"] += round(alim.carbohidratos_100g * factor, 1)
            total["grasas_g"]        += round(alim.grasas_100g * factor, 1)
        else:
            print(f"[Nutricion] Ingrediente no resuelto (incluso con fallback): '{nombre_norm}'")

    if not componentes:
        return None, None

    recalc = {k: round(v, 1) for k, v in total.items()}
    return componentes, recalc


_RE_LINEA_PARECE_INGREDIENTE = re.compile(
    r"(?i)(?:\d+[\d.,]*\s*(g|gr|gramos?|ml\b|cdas?|c\.?d\.?a\.?|tazas?|latas?|rebanad|rodaj|unid(ades?)?|pizca)\b|"
    r"\(\s*\d+[\d.,]*\s*kcal|kcal\s*\)|\bprote[íi]n)"
)

_RE_LINEA_MACROS = re.compile(r"(?i)\b(?:P|C|G|Cal)\s*:\s*[\d.,]+")


def _es_linea_macros(v: str) -> bool:
    t = (v or "").strip()
    if not t:
        return False
    return bool(_RE_LINEA_MACROS.search(t)) and (
        " | " in t or t.count(":") >= 2 or t.lower().startswith(("p:", "c:", "g:", "cal:"))
    )


def _normalizar_lista_texto(v: Any) -> List[str]:
    """
    Normaliza listas que el LLM/parser puede devolver como:
    - list[str]
    - string multi-línea con viñetas
    - None
    """
    if not v:
        return []
    if isinstance(v, list):
        out: List[str] = []
        for x in v:
            t = str(x or "").strip()
            if not t:
                continue
            t = re.sub(r"^(\s*[-\*•]\s?|\s*\d+[\.\)]\s?)", "", t).strip()
            if _es_linea_macros(t):
                continue
            if t:
                out.append(t)
        return out
    if isinstance(v, str):
        lines = [ln.strip() for ln in v.splitlines() if ln.strip()]
        out = []
        for ln in lines:
            t = re.sub(r"^(\s*[-\*•]\s?|\s*\d+[\.\)]\s?)", "", ln).strip()
            if _es_linea_macros(t):
                continue
            if t:
                out.append(t)
        return out
    # fallback conservador
    return [str(v).strip()] if str(v).strip() else []


def _reparar_ingredientes_desde_preparacion(seccion: Dict[str, Any]) -> None:
    """
    Si ingredientes está vacío pero preparación contiene líneas con cantidades/kcal,
    mover esas líneas a ingredientes (para persistencia y UI).
    """
    if seccion.get("tipo") != "comida":
        return
    ing = _normalizar_lista_texto(seccion.get("ingredientes"))
    prep = _normalizar_lista_texto(seccion.get("preparacion"))
    if ing or not prep:
        seccion["ingredientes"] = ing
        seccion["preparacion"] = prep
        return
    nuevos_ing: List[str] = []
    nuevos_prep: List[str] = []
    for linea in prep:
        t = (linea or "").strip()
        if not t:
            continue
        if _es_linea_macros(t):
            nuevos_prep.append(t)
            continue
        if _RE_LINEA_PARECE_INGREDIENTE.search(t) or (
            len(t) <= 120 and re.search(r"(?i)\b\d+[\d.,]*\s*(g|gr|ml)\b", t)
        ):
            nuevos_ing.append(linea)
        else:
            nuevos_prep.append(linea)
    if nuevos_ing:
        seccion["ingredientes"] = nuevos_ing
        seccion["preparacion"] = nuevos_prep
    else:
        seccion["ingredientes"] = ing
        seccion["preparacion"] = prep


def _norm_nombre_plato(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\[.*?\]", "", s)
    s = s.split("[")[0].strip()
    s = re.sub(r"[^a-z0-9áéíóúüñ\s]", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# Sufijos de acompañamiento que no forman parte del nombre del plato principal
# Ej: "lomo saltado con un vaso de agua" → "lomo saltado"
_RE_SUFIJO_ACOMPANAMIENTO = re.compile(
    r"\s+con\s+(un|una|dos|tres)?\s*"
    r"(vaso|taza|copa|vaso de agua|vaso de leche|jugo|refresco|gaseosa|limonada|"
    r"agua|infusion|te|cafe)\b.*$",
    re.IGNORECASE,
)


_RE_VERBO_LOG = re.compile(
    r"^(?:com[ií]|almorcé|almorce|desayuné|desayune|cené|cene|"
    r"tomé|tome|beb[ií]|meriendé|registra\s+que\s+com[ií]|"
    r"registra\s+que\s+almorcé|registra\s+que\s+desayuné|"
    r"registra\s+que\s+tomé|registra\s+que\s*)\s+",
    re.IGNORECASE,
)


def _limpiar_nombre_plato_bd(nombre: str) -> str:
    """
    1) Elimina verbos de LOG al inicio ('Comi X' → 'X', 'Desayune Y' → 'Y').
    2) Elimina acompañamientos de bebida al final ('con un vaso de agua').
    """
    # Quitar prefijo de verbo LOG
    limpio = _RE_VERBO_LOG.sub("", nombre).strip()
    # Quitar sufijo de bebida
    limpio = _RE_SUFIJO_ACOMPANAMIENTO.sub("", limpio).strip()
    return limpio if limpio else nombre


def _buscar_reciente_por_nombre(user_id: Any, nombre_plato: str) -> Optional[Dict[str, Any]]:
    """
    Reutiliza macros si el plato ya se recomendó recientemente.
    Evita que el LLM re-genere el mismo plato con kcal/macros distintos.
    """
    nombre = _norm_nombre_plato(nombre_plato)
    if not nombre:
        return None
    recientes = get_user_recent_meals(user_id) or []
    best = None
    best_score = 0.0
    for m in recientes:
        n2 = _norm_nombre_plato(str(m.get("nombre") or ""))
        if not n2:
            continue
        if n2 == nombre:
            return m
        score = difflib.SequenceMatcher(a=nombre, b=n2).ratio()
        if score > best_score:
            best_score = score
            best = m
    # Threshold alto: solo reutilizar cuando realmente es el mismo plato.
    # Umbral coherente con _buscar_plato_bd_por_nombre
    if best and best_score >= 0.82:
        return best
    return None


# Tabla de conversión medida doméstica → gramos aproximados
_MEDIDA_GRAMOS: dict = {
    "cdta":         5,    # cucharadita  (~5 ml)
    "cucharadita":  5,
    "cta":          5,
    "cda":          15,   # cucharada    (~15 ml)
    "cucharada":    15,
    "taza":         240,  # 1 taza estándar
    "vaso":         240,
    "ml":           1,
}

# Soporta números enteros, decimales Y fracciones comunes (1/4, 1/2, 3/4, 1/3)
_RE_MEDIDA = re.compile(
    r"^(\d+(?:[.,]\d+)?|\d+/\d+)\s*"
    r"(cdta|cucharadita|cta|cda|cucharada|taza|vaso|ml)\b",
    re.IGNORECASE,
)

# Gramos por pieza de alimentos unitarios comunes en cocina peruana
# Fuente: MINSA / INS/CENAN porciones estándar
_GRAMOS_POR_UNIDAD: dict[str, float] = {
    # Frutas
    "aguacate":   200.0,  # palta/aguacate mediano
    "palta":      200.0,
    "platano":    120.0,  # plátano maduro sin cáscara
    "banana":     120.0,
    "manzana":    150.0,
    "naranja":    130.0,
    "mandarina":   80.0,
    "limon":       50.0,
    "lima":        50.0,
    "pera":       150.0,
    "durazno":    130.0,
    "melocoton":  130.0,
    "mango":      200.0,
    "papaya":     200.0,  # 1 trozo estándar
    "sandia":     300.0,  # 1 trozo
    "maracuya":    80.0,
    "granadilla":  80.0,
    "lucuma":     100.0,
    "chirimoya":  150.0,
    "kiwi":        75.0,
    "fresa":       12.0,  # 1 fresa
    # Vegetales
    "tomate":     100.0,
    "cebolla":     80.0,
    "papa":        150.0,
    "camote":      150.0,
    "zanahoria":   80.0,
    "brocoli":    100.0,  # 1 porción / floretes
    "coliflor":   100.0,
    "pepino":     200.0,
    "pimiento":   100.0,
    "choclo":     150.0,  # 1 mazorca desgranada
    "choclo desgranado": 100.0,
    "berenjena":  150.0,
    "zapallo":    150.0,  # 1 trozo
    "yuca":       150.0,
    "lechuga":      5.0,  # 1 hoja
    "espinaca":     5.0,  # 1 hoja
    # Protéicas
    "huevo":       55.0,
    "huevos":      55.0,  # por unidad
    # Panes
    "rebanada":    35.0,
    "rebanadas":   35.0,
    "loncha":      25.0,
    "lonchas":     25.0,
    "tostada":     30.0,  # 1 tostada fina
    "bollo":       50.0,
    "pan":         50.0,  # 1 unidad panecillo
}

# Pieza: rebanadas, lonchas, huevos Y ahora unidades de alimentos comunes
_RE_PIEZA = re.compile(
    r"^(\d+(?:[.,]\d+)?|\d+/\d+)\s*"
    r"(rebanadas?|lonchas?|huevos?|huevo|tostadas?|bollos?|piezas?|unidades?)\b\s+",
    re.IGNORECASE,
)

# Patrón para detectar "N alimento" donde alimento es una pieza unitaria conocida
# Ej: "1 aguacate", "2 tomates", "1 plátano"
_RE_UNIDAD_ALIMENTO = re.compile(
    r"^(\d+(?:[.,]\d+)?|\d+/\d+)\s+(.+?)(?:\s*\(|\s*$)",
    re.IGNORECASE,
)


def _parsear_fraccion(s: str) -> float:
    """Convierte '1/4', '1/2', '3/4', '2' a float. Seguro para cualquier input."""
    s = s.strip().replace(",", ".")
    if "/" in s:
        partes = s.split("/", 1)
        try:
            return float(partes[0]) / float(partes[1])
        except (ValueError, ZeroDivisionError):
            return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def _gramos_por_pieza(unidad: str) -> float:
    u = (unidad or "").lower().strip()
    # Buscar en tabla extendida
    if u in _GRAMOS_POR_UNIDAD:
        return _GRAMOS_POR_UNIDAD[u]
    # Prefijos (rebanada/s, loncha/s...)
    for key, val in _GRAMOS_POR_UNIDAD.items():
        if u.startswith(key[:7]):  # match parcial para plural
            return val
    return 0.0


def _agregar_equivalencia_gramos(ing: str) -> str:
    """
    Si el ingrediente usa una medida doméstica o unidad de pieza,
    añade la equivalencia en gramos entre paréntesis.

    Formatos soportados:
      "1 cdta aceite de oliva (40 kcal)"       → "1 cdta aceite de oliva (~5g | 40 kcal)"
      "1/4 taza queso fresco (100 kcal)"        → "1/4 taza queso fresco (~60g | 100 kcal)"
      "2 rebanadas pan integral (183 kcal)"     → "2 rebanadas pan integral (~70g | 183 kcal)"
      "1 aguacate (140 kcal)"                   → "1 aguacate (~200g | 140 kcal)"
      "2 tomates (45 kcal)"                     → "2 tomates (~200g | 45 kcal)"
    Los ingredientes que ya usan gramos ("150g ...") no se modifican.
    """
    s = ing.strip()

    # Saltar si ya tiene gramos explícitos ("150g ...")
    if re.match(r"^\d+(?:[.,]\d+)?\s*g\b", s, re.IGNORECASE):
        return ing

    # ── 1) Medidas domésticas: cdta / cucharada / taza / vaso / ml ────────────
    m = _RE_MEDIDA.match(s)
    if m:
        cantidad = _parsear_fraccion(m.group(1))
        unidad   = m.group(2).lower()
        g_unit   = _MEDIDA_GRAMOS.get(unidad)
        if g_unit:
            g_total = round(cantidad * g_unit)
            equiv   = f"~{g_total}g"
            if "(" in s:
                return s.replace("(", f"({equiv} | ", 1)
            return f"{s} ({equiv})"

    # ── 2) Piezas nominadas: rebanadas / lonchas / huevos / tostadas ──────────
    m_pz = _RE_PIEZA.match(s)
    if m_pz:
        cant  = _parsear_fraccion(m_pz.group(1))
        g_por = _gramos_por_pieza(m_pz.group(2))
        if g_por > 0:
            g_tot = round(cant * g_por)
            equiv = f"~{g_tot}g"
            if "(" in s:
                return s.replace("(", f"({equiv} | ", 1)
            return f"{s} ({equiv})"

    # ── 3) Alimentos unitarios: "1 aguacate", "2 tomates", "1 plátano" ────────
    # Solo aplica cuando la cantidad es 1-10 unidades (evita "200 pollo" falso)
    m_u = _RE_UNIDAD_ALIMENTO.match(s)
    if m_u:
        cant_raw  = m_u.group(1)
        alim_raw  = m_u.group(2).strip().lower()
        # Probar: exacto → sin 's' → sin 'es'
        g_por = (
            _GRAMOS_POR_UNIDAD.get(alim_raw)
            or _GRAMOS_POR_UNIDAD.get(alim_raw.rstrip("s"))
            or _GRAMOS_POR_UNIDAD.get(re.sub(r"es$", "", alim_raw))
        )
        if g_por:
            cant  = _parsear_fraccion(cant_raw)
            if 0 < cant <= 10:  # guardia: no confundir "200 pollo" con unidades
                g_tot = round(cant * g_por)
                equiv = f"~{g_tot}g"
                if "(" in s:
                    return s.replace("(", f"({equiv} | ", 1)
                return f"{s} ({equiv})"

    return ing


_RE_ING_GRAMOS  = re.compile(r"^(\d+(?:[.,]\d+)?)\s*g\s+(.+?)(?:\s*\(.*\))?$", re.IGNORECASE)
_RE_ING_KCAL    = re.compile(r"\(\s*([\d.,]+)\s*kcal\s*\)", re.IGNORECASE)


def _recalcular_ing_desde_bd(db: Session, ing: str) -> str:
    """
    Para un string de ingrediente generado por LLM, busca el alimento en BD
    y recalcula las kcal reales según los gramos indicados.

    Soporta:
      "200g pechuga de pollo (240 kcal)"  → "200g pechuga de pollo (330 kcal)"
      "1 cucharada aceite de oliva (120 kcal)" → "1 cucharada aceite de oliva (~15g | 132.6 kcal)"
    """
    from sqlalchemy import text as _text

    # ── Determinar gramos y nombre del ingrediente ──────────────────────
    ing_clean = ing.strip()

    # Formato: "200g pechuga de pollo ..."
    m_g = _RE_ING_GRAMOS.match(ing_clean)
    if m_g:
        gramos     = float(m_g.group(1).replace(",", "."))
        nombre_ing = m_g.group(2).strip()
        prefix     = f"{int(gramos) if gramos == int(gramos) else gramos}g"
    else:
        m_pz = _RE_PIEZA.match(ing_clean)
        if m_pz:
            gramos     = float(m_pz.group(1).replace(",", ".")) * _gramos_por_pieza(m_pz.group(2))
            nombre_ing = ing_clean[m_pz.end():].strip()
            nombre_ing = _RE_ING_KCAL.sub("", nombre_ing).strip()
            g_str      = str(int(gramos)) if gramos == int(gramos) else str(round(gramos, 1))
            prefix     = f"{m_pz.group(1)} {m_pz.group(2)} (~{g_str}g)"
        else:
            # Formato: "1 cucharada / 2 cdta ..."
            m_med = _RE_MEDIDA.match(ing_clean)
            if not m_med:
                return _agregar_equivalencia_gramos(ing_clean)
            cantidad   = float(m_med.group(1).replace(",", "."))
            unidad     = m_med.group(2).lower()
            g_unit     = _MEDIDA_GRAMOS.get(unidad, 0)
            if not g_unit:
                return _agregar_equivalencia_gramos(ing_clean)
            gramos     = cantidad * g_unit
            # Quitar la parte de medida para quedarnos con el nombre
            nombre_ing = ing_clean[m_med.end():].strip()
            # Quitar "(X kcal)" del nombre si hubiera
            nombre_ing = _RE_ING_KCAL.sub("", nombre_ing).strip()
            g_str      = str(int(gramos)) if gramos == int(gramos) else str(round(gramos, 1))
            prefix     = f"{m_med.group(0)} (~{g_str}g)"

    # Quitar "(X kcal)" del nombre si viene pegado
    nombre_ing = _RE_ING_KCAL.sub("", nombre_ing).strip()

    nombre_norm_ing = _norm(nombre_ing)
    if _es_nombre_condimento_traza(nombre_norm_ing):
        return f"{prefix} {nombre_ing} (0 kcal)"

    # ── Buscar alimento en BD (nombre exacto → alias → contains) ─────────
    row = db.execute(_text(
        "SELECT a.calorias_100g FROM alimentos a"
        " WHERE a.nombre_normalizado = :n LIMIT 1"
    ), {"n": nombre_norm_ing}).fetchone()

    if not row:
        # Intentar via alias
        row = db.execute(_text(
            "SELECT a.calorias_100g FROM alimento_alias al"
            " JOIN alimentos a ON a.id = al.alimento_id"
            " WHERE al.alias_normalizado = :n LIMIT 1"
        ), {"n": nombre_norm_ing}).fetchone()

    if not row:
        # Buscar por contains (máximo 1 palabra clave larga)
        palabras = [p for p in nombre_norm_ing.split() if len(p) > 4]
        for p in palabras[:2]:
            row = db.execute(_text(
                "SELECT a.calorias_100g FROM alimentos a"
                " WHERE a.nombre_normalizado LIKE :p LIMIT 1"
            ), {"p": f"%{p}%"}).fetchone()
            if row:
                break

    if not row:
        # No encontrado → solo agregar equivalencia gramos si aplica
        return _agregar_equivalencia_gramos(ing_clean)

    kcal_100g = float(row[0] or 0)
    kcal_real = kcal_100g * gramos / 100.0
    kcal_str  = str(int(kcal_real)) if kcal_real == int(kcal_real) else str(round(kcal_real, 1))

    return f"{prefix} {nombre_ing} ({kcal_str} kcal)"


async def _recalcular_ing_async(db: Session, ing: str) -> str:
    """
    Versión async de _recalcular_ing_desde_bd.
    Si el alimento no está en BD, lo busca vía USDA/Groq y lo guarda
    para que la misma solicitud muestre kcal correctas desde el primer pedido.
    """
    from sqlalchemy import text as _text

    ing_clean = ing.strip()

    # Determinar gramos y nombre
    m_g = _RE_ING_GRAMOS.match(ing_clean)
    if m_g:
        gramos     = float(m_g.group(1).replace(",", "."))
        nombre_ing = m_g.group(2).strip()
        prefix     = f"{int(gramos) if gramos == int(gramos) else gramos}g"
    else:
        m_pz = _RE_PIEZA.match(ing_clean)
        if m_pz:
            gramos     = float(m_pz.group(1).replace(",", ".")) * _gramos_por_pieza(m_pz.group(2))
            nombre_ing = ing_clean[m_pz.end():].strip()
            nombre_ing = _RE_ING_KCAL.sub("", nombre_ing).strip()
            g_str      = str(int(gramos)) if gramos == int(gramos) else str(round(gramos, 1))
            prefix     = f"{m_pz.group(1)} {m_pz.group(2)} (~{g_str}g)"
        else:
            m_med = _RE_MEDIDA.match(ing_clean)
            if not m_med:
                return _agregar_equivalencia_gramos(ing_clean)
            cantidad = float(m_med.group(1).replace(",", "."))
            unidad   = m_med.group(2).lower()
            g_unit   = _MEDIDA_GRAMOS.get(unidad, 0)
            if not g_unit:
                return _agregar_equivalencia_gramos(ing_clean)
            gramos     = cantidad * g_unit
            nombre_ing = ing_clean[m_med.end():].strip()
            nombre_ing = _RE_ING_KCAL.sub("", nombre_ing).strip()
            g_str      = str(int(gramos)) if gramos == int(gramos) else str(round(gramos, 1))
            prefix     = f"{m_med.group(0)} (~{g_str}g)"

    nombre_ing     = _RE_ING_KCAL.sub("", nombre_ing).strip()
    nombre_norm_ing = _norm(nombre_ing)
    if _es_nombre_condimento_traza(nombre_norm_ing):
        return f"{prefix} {nombre_ing} (0 kcal)"

    # Buscar en BD (exacto → alias → contains)
    kcal_100g: Optional[float] = None
    row = db.execute(_text(
        "SELECT a.calorias_100g FROM alimentos a WHERE a.nombre_normalizado = :n LIMIT 1"
    ), {"n": nombre_norm_ing}).fetchone()
    if row:
        kcal_100g = float(row[0] or 0)

    if kcal_100g is None:
        row = db.execute(_text(
            "SELECT a.calorias_100g FROM alimento_alias al"
            " JOIN alimentos a ON a.id = al.alimento_id"
            " WHERE al.alias_normalizado = :n LIMIT 1"
        ), {"n": nombre_norm_ing}).fetchone()
        if row:
            kcal_100g = float(row[0] or 0)

    if kcal_100g is None:
        palabras = [p for p in nombre_norm_ing.split() if len(p) > 4]
        for p in palabras[:2]:
            row = db.execute(_text(
                "SELECT a.calorias_100g FROM alimentos a"
                " WHERE a.nombre_normalizado LIKE :p LIMIT 1"
            ), {"p": f"%{p}%"}).fetchone()
            if row:
                kcal_100g = float(row[0] or 0)
                break

    # Si no encontrado → USDA/Groq → guardar en alimentos
    if kcal_100g is None:
        alim = await _buscar_o_crear_alimento_async(db, nombre_norm_ing, nombre_ing)
        if alim:
            kcal_100g = float(alim.calorias_100g or 0)

    if kcal_100g is None:
        return _agregar_equivalencia_gramos(ing_clean)

    kcal_real = kcal_100g * gramos / 100.0
    kcal_str  = str(int(kcal_real)) if kcal_real == int(kcal_real) else str(round(kcal_real, 1))
    return f"{prefix} {nombre_ing} ({kcal_str} kcal)"


def _cargar_ingredientes_bd(db: Session, plato_id: int) -> list:
    """
    Devuelve ingredientes de un plato con macros completos calculados desde
    la cadena platos → plato_ingredientes → alimentos (sumatoria matemática).

    Formato de salida (compatible con Flutter _ingredientLine widget):
        "Arroz Blanco 150g (546 kcal)"
    El widget parsea el último (X kcal) como badge naranja y muestra el nombre limpio.
    """
    from sqlalchemy import text as _text
    rows = db.execute(_text(
        "SELECT a.nombre, pi2.gramos,"
        " ROUND((a.calorias_100g      * pi2.gramos / 100.0)::numeric, 1) AS kcal,"
        " ROUND((a.proteina_100g      * pi2.gramos / 100.0)::numeric, 1) AS prot,"
        " ROUND((a.carbohidratos_100g * pi2.gramos / 100.0)::numeric, 1) AS carb,"
        " ROUND((a.grasas_100g        * pi2.gramos / 100.0)::numeric, 1) AS gras"
        " FROM plato_ingredientes pi2"
        " JOIN alimentos a ON a.id = pi2.alimento_id"
        " WHERE pi2.plato_id = :pid"
        " ORDER BY kcal DESC"
    ), {"pid": plato_id}).fetchall()
    result = []
    for nombre_ing, gramos, kcal, prot, carb, gras in rows:
        kcal_v = float(kcal or 0)
        prot_v = round(float(prot or 0), 1)
        carb_v = round(float(carb or 0), 1)
        gras_v = round(float(gras or 0), 1)
        # Formato kcal: entero si es exacto, 1 decimal si no
        kcal_s = str(int(kcal_v)) if kcal_v == int(kcal_v) else str(round(kcal_v, 1))
        # Gramaje limpio (sin decimales si es entero)
        gramos_s = str(int(gramos)) if float(gramos) == int(float(gramos)) else str(round(float(gramos), 1))
        # Formato Flutter-compatible: "Nombre Xg (Y kcal)"
        # El widget _ingredientLine() extrae el badge de kcal del último paréntesis (Y kcal)
        result.append(f"{nombre_ing} {gramos_s}g ({kcal_s} kcal)")
    return result


# ─── Reglas esenciales/prohibidos para platos recuperados de BD ───────────────
# Opera sobre los nombres normalizados de los ingredientes reales en BD.
# Si el plato falla, se descarta y se fuerza re-creación vía LLM.
_PLATOS_ESENCIALES_BD: list[dict] = [
    {
        "keywords":    ("ceviche", "cebiche"),
        "obligatorio_todos": [
            # grupo 1: proteina marina
            ("pescado", "lisa", "caballa", "mero", "tollo", "camaron",
             "langostino", "pulpo", "calamar", "anchoveta", "bonito",
             "trucha", "salmon", "atun"),
            # grupo 2: ácido
            ("limon", "lima", "citrico"),
            # grupo 3: cebolla (ESENCIAL — es la trinidad del ceviche peruano)
            ("cebolla",),
        ],
        "prohibidos": (
            "aceite de oliva", "aceite vegetal", "aceite",
            "mantequilla", "crema", "mayonesa", "leche",
            "queso", "yogurt",
            "pescado blanco cocido", "salmon cocido", "trucha cocida",
        ),
    },
    {
        "keywords":    ("tiradito",),
        "obligatorio_todos": [
            ("pescado", "lisa", "caballa", "mero", "lenguado", "trucha",
             "salmon", "atun", "bonito"),
            ("limon", "lima"),
        ],
        "prohibidos": (
            "aceite de oliva", "mantequilla", "crema", "mayonesa",
        ),
    },
    {
        "keywords":    ("tortilla de huevo", "tortilla"),
        "obligatorio_todos": [("huevo",)],
        "prohibidos":  (),
    },
]


def _validar_plato_bd_esenciales(plato_id: int, nombre_norm: str, db) -> tuple[bool, str]:
    """
    Verifica que un plato recuperado de BD tenga los ingredientes esenciales
    y no tenga ingredientes prohibidos según su tipo culinario.

    Retorna (True, "") si el plato es válido.
    Retorna (False, motivo) si debe descartarse y re-crearse.
    """
    from sqlalchemy import text as _text
    rows = db.execute(_text(
        "SELECT a.nombre_normalizado FROM plato_ingredientes pi2"
        " JOIN alimentos a ON a.id = pi2.alimento_id"
        " WHERE pi2.plato_id = :pid"
    ), {"pid": plato_id}).fetchall()
    ings_norms = [r[0] or "" for r in rows]

    for regla in _PLATOS_ESENCIALES_BD:
        if not any(kw in nombre_norm for kw in regla["keywords"]):
            continue

        # 1) Ingredientes esenciales obligatorios
        for grupo in regla.get("obligatorio_todos", []):
            if not any(any(req in ing for req in grupo) for ing in ings_norms):
                return False, (
                    f"ceviche/tiradito en BD sin ingrediente esencial "
                    f"({', '.join(grupo[:3])}...) — plato id={plato_id} descartado"
                )

        # 2) Ingredientes prohibidos
        for prohibido in regla.get("prohibidos", ()):
            if any(prohibido in ing for ing in ings_norms):
                return False, (
                    f"plato id={plato_id} tiene ingrediente prohibido "
                    f"'{prohibido}' para tipo '{regla['keywords'][0]}'"
                )

    return True, ""


def _buscar_plato_bd_por_nombre(db: Session, nombre_norm: str) -> Optional[Dict[str, Any]]:
    """
    Busca plato en `platos` calculando macros desde plato_ingredientes × alimentos:
    - match exacto por nombre_normalizado
    - fallback por similaridad si no hay match exacto
    Los ingredientes devueltos incluyen gramos y kcal reales (no LLM).
    """
    from sqlalchemy import text as _text
    from app.models.plato import Plato

    # Nota: p.preparacion y p.nota son columnas JSON → no se pueden usar en GROUP BY.
    # Se agrupan solo por p.id/p.nombre y se cargan por separado.
    _SQL_MACROS = (
        "SELECT p.id, p.nombre,"
        " SUM(a.calorias_100g      * pi2.gramos / 100.0),"
        " SUM(a.proteina_100g      * pi2.gramos / 100.0),"
        " SUM(a.carbohidratos_100g * pi2.gramos / 100.0),"
        " SUM(a.grasas_100g        * pi2.gramos / 100.0)"
        " FROM platos p"
        " JOIN plato_ingredientes pi2 ON pi2.plato_id = p.id"
        " JOIN alimentos a ON a.id = pi2.alimento_id"
        " WHERE p.nombre_normalizado = :q"
        " GROUP BY p.id, p.nombre"
        " LIMIT 1"
    )
    row = db.execute(_text(_SQL_MACROS), {"q": nombre_norm}).fetchone()
    if row:
        plato_id = row[0]
        # Validar esenciales/prohibidos antes de servir el plato
        ok, motivo = _validar_plato_bd_esenciales(plato_id, nombre_norm, db)
        if not ok:
            logger.warning("[bd_esenciales] %s", motivo)
            # Eliminar plato corrupto para no servirlo de nuevo
            try:
                from sqlalchemy import text as _del
                db.execute(_del("DELETE FROM plato_ingredientes WHERE plato_id=:p"), {"p": plato_id})
                db.execute(_del("DELETE FROM platos WHERE id=:p"), {"p": plato_id})
                db.commit()
                logger.warning("[bd_esenciales] Plato id=%d eliminado de BD", plato_id)
            except Exception as e:
                logger.error("[bd_esenciales] Error eliminando plato: %s", e)
            return None
        plato_meta = (
            db.query(Plato.preparacion, Plato.nota)
            .filter(Plato.id == plato_id)
            .first()
        )
        return {
            "plato_id":        plato_id,
            "calorias":        float(row[2] or 0),
            "proteinas_g":     float(row[3] or 0),
            "carbohidratos_g": float(row[4] or 0),
            "grasas_g":        float(row[5] or 0),
            "ingredientes":    _cargar_ingredientes_bd(db, plato_id),
            "preparacion":     list(plato_meta[0] or []) if plato_meta else [],
            "nota":            plato_meta[1] if plato_meta else None,
        }

    # Fallback por similaridad (top 300 más recientes)
    cand = (
        db.query(Plato.id, Plato.nombre_normalizado, Plato.preparacion, Plato.nota)
        .order_by(Plato.id.desc())
        .limit(300)
        .all()
    )
    best_id    = None
    best_score = 0.0
    best_prep  = None
    best_nota  = None
    for plato_id, nn, prep, nota in cand:
        if not nn:
            continue
        if not coherencia_proteina_platos(nombre_norm, str(nn)):
            continue
        score = difflib.SequenceMatcher(a=nombre_norm, b=str(nn)).ratio()
        if score > best_score:
            best_score = score
            best_id    = plato_id
            best_prep  = prep
            best_nota  = nota

    if best_id and best_score >= 0.82:
        # Validar esenciales/prohibidos en el candidato fuzzy también
        ok, motivo = _validar_plato_bd_esenciales(best_id, nombre_norm, db)
        if not ok:
            logger.warning("[bd_esenciales_fuzzy] %s", motivo)
            try:
                from sqlalchemy import text as _del
                db.execute(_del("DELETE FROM plato_ingredientes WHERE plato_id=:p"), {"p": best_id})
                db.execute(_del("DELETE FROM platos WHERE id=:p"), {"p": best_id})
                db.commit()
            except Exception:
                pass
            return None
        row2 = db.execute(_text(
            "SELECT SUM(a.calorias_100g * pi2.gramos / 100.0),"
            " SUM(a.proteina_100g * pi2.gramos / 100.0),"
            " SUM(a.carbohidratos_100g * pi2.gramos / 100.0),"
            " SUM(a.grasas_100g * pi2.gramos / 100.0)"
            " FROM plato_ingredientes pi2"
            " JOIN alimentos a ON a.id = pi2.alimento_id"
            " WHERE pi2.plato_id = :pid"
        ), {"pid": best_id}).fetchone()
        return {
            "plato_id":        best_id,
            "calorias":        float(row2[0] or 0) if row2 else 0.0,
            "proteinas_g":     float(row2[1] or 0) if row2 else 0.0,
            "carbohidratos_g": float(row2[2] or 0) if row2 else 0.0,
            "grasas_g":        float(row2[3] or 0) if row2 else 0.0,
            "ingredientes":    _cargar_ingredientes_bd(db, best_id),
            "preparacion":     list(best_prep or []),
            "nota":            best_nota,
        }
    return None


def verificar_conflicto_macros(
    progreso: Any,
    plan_hoy_data: Dict[str, Any],
    _perfil: Any,
) -> Optional[str]:
    """
    Verifica si el progreso actual supera los límites diarios del plan.
    Retorna un string de advertencia o None si todo está bien.
    """
    alertas: List[str] = []
    consumidas = progreso.calorias_consumidas or 0
    meta_cal = plan_hoy_data.get("calorias_dia", 0) or 0

    if meta_cal > 0 and consumidas > meta_cal:
        exceso = consumidas - meta_cal
        alertas.append(
            f"🔥 Calorías: llevas {consumidas:.0f}/{meta_cal:.0f} kcal (+{exceso:.0f} de exceso)"
        )

    prot_consumidas = progreso.proteinas_consumidas or 0
    prot_meta = plan_hoy_data.get("proteinas_g", 0) or 0
    if prot_meta > 0 and prot_consumidas > prot_meta * 1.15:
        alertas.append(f"🥚 Proteínas: {prot_consumidas:.0f}g de {prot_meta:.0f}g meta")

    carbs_consumidos = progreso.carbohidratos_consumidos or 0
    carbs_meta = plan_hoy_data.get("carbohidratos_g", 0) or 0
    if carbs_meta > 0 and carbs_consumidos > carbs_meta * 1.15:
        alertas.append(f"🍞 Carbohidratos: {carbs_consumidos:.0f}g de {carbs_meta:.0f}g meta")

    grasas_consumidas = progreso.grasas_consumidas or 0
    grasas_meta = plan_hoy_data.get("grasas_g", 0) or 0
    if grasas_meta > 0 and grasas_consumidas > grasas_meta * 1.15:
        alertas.append(f"🥑 Grasas: {grasas_consumidas:.0f}g de {grasas_meta:.0f}g meta")

    if not alertas:
        return None

    detalle = "\n".join(alertas)
    return (
        f"⚠️ Has superado algunos límites de tu plan de hoy:\n"
        f"{detalle}\n"
        f"💡 Considera compensar con actividad física o elegir opciones más ligeras en tu próxima comida."
    )


_RE_NOMBRE_GENERICO = re.compile(
    r"(?i)^(?:sugerencia|opci[oó]n|plato|comida|receta|alternativa)\s*\d*\.?\s*$"
)

# ─── Filtro calórico por momento del día ─────────────────────────────────────
_TDEE_FALLBACK_KCAL: float = 2000.0


def _inferir_momento_actual() -> str:
    return inferir_momento_dia_peru()


def _obtener_pct_por_objetivo(goal: str) -> dict:
    """Devuelve el mapa de % TDEE máximo por momento según el objetivo del usuario."""
    g = (goal or "").lower()
    if "agresivo" in g:
        return {"desayuno": 0.25, "almuerzo": 0.40, "cena": 0.25, "merienda": 0.10, "snack": 0.10}
    elif "definicion" in g or "definición" in g:
        return {"desayuno": 0.30, "almuerzo": 0.40, "cena": 0.30, "merienda": 0.10, "snack": 0.10}
    elif "volumen" in g or "masa" in g:
        return {"desayuno": 0.35, "almuerzo": 0.40, "cena": 0.35, "merienda": 0.15, "snack": 0.15}
    else:
        return {"desayuno": 0.30, "almuerzo": 0.40, "cena": 0.30, "merienda": 0.12, "snack": 0.15}


def _calcular_tdee_perfil(perfil: Any) -> float:
    """
    Devuelve el TDEE estimado con Harris-Benedict.
    Fallback a _TDEE_FALLBACK_KCAL si faltan campos o hay error.
    """
    try:
        peso   = getattr(perfil, "weight", None)
        altura = getattr(perfil, "height", None)
        genero = getattr(perfil, "gender", None)
        birth  = getattr(perfil, "birth_date", None)
        nivel  = getattr(perfil, "activity_level", None) or "Moderado"
        goal   = getattr(perfil, "goal", None) or "Mantener peso"

        if not all([peso, altura, genero, birth]):
            return _TDEE_FALLBACK_KCAL
        if float(peso) < 30 or float(altura) < 100:
            return _TDEE_FALLBACK_KCAL

        edad = (datetime.now().date() - birth).days // 365
        if not (5 < edad < 120):
            return _TDEE_FALLBACK_KCAL

        rec = CalculadorDietaAutomatica.calcular_recomendacion_dieta(
            float(peso), float(altura), edad, genero, nivel, goal
        )
        tdee = float(rec.calorias_diarias)
        return tdee if tdee > 0 else _TDEE_FALLBACK_KCAL
    except Exception as _e:
        logger.warning("[_calcular_tdee_perfil] fallback %.0f kcal: %s", _TDEE_FALLBACK_KCAL, _e)
        return _TDEE_FALLBACK_KCAL


def _verificar_rango_calorico(kcal: float, momento: str, tdee: float, goal: str = "") -> tuple:
    """
    Devuelve (rechazar: bool, motivo: str).
    Solo rechaza por exceso calórico; nunca rechaza por kcal baja.
    """
    if kcal <= 0:
        return False, ""

    # Hard cap absoluto para desayuno
    if momento == "desayuno" and kcal > 700:
        return True, f"hard cap desayuno: {kcal:.0f} kcal > 700 kcal"

    # Límite universal de seguridad
    if kcal > 0.50 * tdee:
        return True, f"{kcal:.0f} kcal > 50% TDEE ({0.50 * tdee:.0f} kcal)"

    # Límite dinámico por objetivo
    pct_max = _obtener_pct_por_objetivo(goal).get(momento, 0.40)
    if kcal > pct_max * tdee:
        return True, (
            f"{momento}: {kcal:.0f} kcal > {int(pct_max * 100)}% TDEE "
            f"({pct_max * tdee:.0f} kcal, objetivo='{goal or 'mantener'}')"
        )

    return False, ""


# ─── CAMBIO 4: Bloqueo semántico por momento del día ─────────────────────────
# Platos que no deben recomendarse en ciertos momentos por razones de digestión
# y protocolo nutricional peruano (ceviche es almuerzo, no cena).
_PLATOS_NOCTURNOS_PROHIBIDOS: list[str] = [
    "ceviche",
    "cebiche",
    "tiradito",
    "chicharron de pescado",
    "chicharron de camaron",
    "jalea",
]


def _validar_momento_semantico(nombre_norm: str, momento: str) -> tuple[bool, str]:
    """
    CAMBIO 4 — Valida que el plato sea apropiado para el momento del día.

    Reglas actuales:
    - cena: ceviche, tiradito, jaleas y chicharrones de mar NO recomendados
      (digestión pesada, platos fríos/crudos de mar no aptos para la noche).

    Convención: retorna (True, motivo) si RECHAZADO; (False, "") si PERMITIDO.
    (Igual que _verificar_rango_calorico y el resto del pipeline.)
    """
    if not momento or not nombre_norm:
        return False, ""

    if momento == "cena":
        for plato_prohibido in _PLATOS_NOCTURNOS_PROHIBIDOS:
            if plato_prohibido in nombre_norm:
                return True, (
                    f"'{plato_prohibido}' no recomendado en cena — "
                    f"plato frío/crudo marino, mejor en almuerzo"
                )

    return False, ""


def _extraer_platos_del_mensaje(mensaje: str) -> List[str]:
    """
    Extrae nombres candidatos de platos desde el mensaje original.
    Quita verbos de LOG ("comi", "desayune") y de pregunta ("dame", "recomiéndame").
    """
    if not mensaje:
        return []
    low = mensaje.lower().strip()
    # Quitar verbos de LOG y pregunta al inicio
    for pref in (
        # LOG
        "comi ", "comí ", "almorcé ", "almorce ", "desayuné ", "desayune ",
        "cené ", "cene ", "tomé ", "tome ", "bebí ", "bebi ", "meriendé ",
        "almorce con ", "desayune con ",
        # Pregunta / info
        "dame informacion de", "dame información de", "dame info de",
        "dame", "recomiendame", "recomiéndame", "cuéntame sobre", "cuentame sobre",
        "como se prepara", "cómo se prepara", "que tiene", "qué tiene",
        # Registrar que...
        "registra que comi ", "registra que comí ", "registra que almorce ",
        "registra que almorcé ", "registra que desayune ", "registra que desayuné ",
        "registra que tome ", "registra que tomé ",
        "registra que ",
    ):
        if low.startswith(pref):
            low = low[len(pref):].strip()
            break
    # Quitar sufijos de contexto temporal al final
    for suf in (
        " en el almuerzo", " en el desayuno", " en la cena", " en la merienda",
        " al almuerzo", " al desayuno", " a la cena",
        " con ingredientes y preparacion", " con ingredientes y preparación",
        " con ingredientes", " con preparacion", " con preparación",
        " y preparacion", " y preparación",
    ):
        if low.endswith(suf):
            low = low[: -len(suf)].strip()
    return [low.strip()] if low.strip() else []


# ─── FASE 4: Helpers de flujo invertido (BD construye, LLM solo propone) ──────

async def _resolver_y_construir_desde_bd(
    db: Session,
    nombre_plato: str,
    ingredientes_llm: List[str],
) -> Optional[Dict[str, Any]]:
    """
    FASE 4 — Inversión del flujo: dado nombre e ingredientes propuestos por el LLM,
    resuelve cada ingrediente en BD y calcula macros reales (Trinidad Nutricional).
    Nunca confía en macros del LLM — los recalcula siempre desde alimentos.

    Returns dict con calorias/proteinas_g/carbohidratos_g/grasas_g/ingredientes
    o None si no se pueden resolver ≥2 ingredientes válidos.
    """
    parsed: List[tuple] = []  # [(Alimento, gramos)]
    for ing_str in (ingredientes_llm or []):
        result = _parse_ing_gramos(str(ing_str))
        if not result:
            continue
        nombre_norm_ing, gramos = result
        if gramos <= 0:
            continue
        alim = _resolver_alimento_en_bd(db, nombre_norm_ing)
        if alim:
            parsed.append((alim, gramos))

    if len(parsed) < 2:
        logger.debug(
            "[FASE4] _resolver_bd: solo %d ingrediente(s) resueltos en BD para '%s'",
            len(parsed), nombre_plato,
        )
        return None

    # FASE 4.2 — Compatibilidad entre ingredientes (antes de calcular macros)
    try:
        from app.services.plato_constructor import _validar_compatibilidad_ingredientes
        _ok_compat, _mot_compat = _validar_compatibilidad_ingredientes(parsed)
        if not _ok_compat:
            logger.warning(
                "[FASE4.2] _resolver_bd: '%s' rechazado — %s",
                nombre_plato, _mot_compat,
            )
            return None
    except Exception as _e_compat:
        logger.debug("[FASE4.2] _resolver_bd no verificado para '%s': %s", nombre_plato, _e_compat)

    # Trinity nutricional: macros SOLO desde BD (fuente única de verdad)
    kcal = sum(float(a.calorias_100g or 0) * g / 100 for a, g in parsed)
    prot = sum(float(a.proteina_100g or 0) * g / 100 for a, g in parsed)
    carb = sum(float(a.carbohidratos_100g or 0) * g / 100 for a, g in parsed)
    gras = sum(float(a.grasas_100g or 0) * g / 100 for a, g in parsed)

    if kcal <= 0:
        logger.debug("[FASE4] _resolver_bd: kcal=0 para '%s'", nombre_plato)
        return None

    # Atwater (log-only, no bloqueante — evita rechazar por gap pequeño)
    _ok_atw, _mot_atw = validar_macros_atwater(kcal, prot, carb, gras)
    if not _ok_atw:
        logger.warning(
            "[FASE4] 0e Atwater gap en '%s': %s — macros BD aceptados",
            nombre_plato, _mot_atw,
        )

    # Construir strings de ingredientes en formato Trinity
    ings_str = [
        f"{alim.nombre} {gramos:.0f}g → "
        f"{round(float(alim.calorias_100g or 0) * gramos / 100, 1)} kcal | "
        f"P:{round(float(alim.proteina_100g or 0) * gramos / 100, 1)}g "
        f"C:{round(float(alim.carbohidratos_100g or 0) * gramos / 100, 1)}g "
        f"G:{round(float(alim.grasas_100g or 0) * gramos / 100, 1)}g"
        for alim, gramos in parsed
    ]

    return {
        "calorias":        round(kcal, 1),
        "proteinas_g":     round(prot, 1),
        "carbohidratos_g": round(carb, 1),
        "grasas_g":        round(gras, 1),
        "ingredientes":    ings_str,
        "_fuente":         "bd_resolucion_directa",
    }


# ─── FASE 4.3: Motor de selección inteligente ────────────────────────────────

_INTENCION_ALTA_PROTEINA: frozenset[str] = frozenset({
    "alto en proteina", "alta proteina", "mas proteina", "proteina alta",
    "ganar masa", "masa muscular", "musculo", "hipertrofia",
    "proteico", "proteica", "subir masa",
})
_INTENCION_BAJO_CALORIAS: frozenset[str] = frozenset({
    "bajo en calorias", "pocas calorias", "ligero", "liviano", "light",
    "dieta", "bajar de peso", "perder peso", "deficit", "adelgazar",
    "algo ligero", "poco calorico",
})
_INTENCION_RAPIDO: frozenset[str] = frozenset({
    "rapido", "rapida", "facil", "sencillo", "simple", "sin cocinar",
    "facil de preparar",
})

# Plato base contextual para smart fallback (CAMBIO 7)
_PLATOS_BASE_CONTEXTUAL: Dict[str, Dict[str, str]] = {
    "alto_proteina": {
        "desayuno": "huevos revueltos con avena integral",
        "almuerzo": "pechuga de pollo con arroz y ensalada",
        "cena":     "pechuga de pollo a la plancha con verduras",
        "snack":    "atun con galletas integrales",
    },
    "bajo_calorias": {
        "desayuno": "avena con platano",
        "almuerzo": "ensalada de pollo con verduras mixtas",
        "cena":     "sopa de pollo con verduras",
        "snack":    "manzana con yogurt natural",
    },
    "default": {
        "desayuno": "avena con platano y leche",
        "almuerzo": "arroz con pollo y ensalada verde",
        "cena":     "sopa de pollo con verduras",
        "snack":    "fruta fresca con yogurt",
    },
}

# Palabras clave por proteína principal (para control de variedad)
_PROTEINAS_VARIEDAD_KEYS: Dict[str, List[str]] = {
    "pollo":    ["pollo", "pechuga", "muslo"],
    "pescado":  ["pescado", "caballa", "lisa", "mero", "tollo", "salmon",
                 "trucha", "atun", "bonito", "salpreso"],
    "res":      ["res", "lomo", "carne", "bistec", "vacuno"],
    "cerdo":    ["cerdo", "chancho", "tocino"],
    "mariscos": ["camaron", "langostino", "pulpo", "calamar"],
    "huevo":    ["huevo"],
    "legumbre": ["lenteja", "frijol", "garbanzo", "arveja"],
}


def _interpretar_intencion_usuario(mensaje: str) -> Dict[str, Any]:
    """
    FASE 4.3 — Detecta intención nutricional del usuario en el mensaje.
    Retorna dict con flags: alto_proteina, bajo_calorias, rapido.
    """
    msg_norm = _norm(mensaje or "")
    return {
        "alto_proteina": any(kw in msg_norm for kw in _INTENCION_ALTA_PROTEINA),
        "bajo_calorias": any(kw in msg_norm for kw in _INTENCION_BAJO_CALORIAS),
        "rapido":        any(kw in msg_norm for kw in _INTENCION_RAPIDO),
    }


def _detectar_proteina_plato(nombre_norm: str, ings_strs: List[str]) -> str:
    """
    FASE 4.3 — Detecta la proteína principal de un plato para control de variedad.
    Retorna uno de los keys de _PROTEINAS_VARIEDAD_KEYS, o 'otro'.
    """
    texto = nombre_norm + " " + " ".join(_norm(str(s)) for s in (ings_strs or []))
    for proteina, keywords in _PROTEINAS_VARIEDAD_KEYS.items():
        if any(kw in texto for kw in keywords):
            return proteina
    return "otro"


def _elegir_plato_base_contextual(intencion: Dict[str, Any], momento: str) -> str:
    """CAMBIO 7 — Nombre de plato base apropiado para intención + momento."""
    if intencion.get("alto_proteina"):
        grupo = "alto_proteina"
    elif intencion.get("bajo_calorias"):
        grupo = "bajo_calorias"
    else:
        grupo = "default"
    return _PLATOS_BASE_CONTEXTUAL[grupo].get(
        momento,
        _PLATOS_BASE_CONTEXTUAL[grupo]["almuerzo"],
    )


def _score_plato(
    plato_data: Dict[str, Any],
    perfil: Any,
    intencion: Dict[str, Any],
    momento: str,
) -> float:
    """
    FASE 4.3 — Scoring de plato para rankeo inteligente.

    Criterios (acumulativos):
    1. Proteína alta (si se pidió): +2.0 / bonus suave siempre +0.5
    2. Objetivo usuario (perder/volumen): ajuste kcal ±1.0
    3. Momento del día: penalizar pesado en cena / desayuno
    4. Intención bajo calorías: ajuste ±1.5
    5. Completitud P+C+G: +0.3

    Retorna float; platos con kcal=0 → −10.0 (excluidos del pool).
    """
    kcal = float(plato_data.get("calorias") or 0)
    prot = float(plato_data.get("proteinas_g") or 0)
    carb = float(plato_data.get("carbohidratos_g") or 0)
    gras = float(plato_data.get("grasas_g") or 0)

    if kcal <= 0:
        return -10.0

    score = 0.0

    # 1) Proteína
    if intencion.get("alto_proteina"):
        score += 2.0 if prot >= 25 else (-0.5 if prot < 15 else 0.0)
    elif prot >= 20:
        score += 0.5

    # 2) Objetivo
    _goal = _norm(getattr(perfil, "goal", "") or "")
    if any(w in _goal for w in ("perder", "deficit", "definicion", "adelgazar")):
        score += (1.0 if kcal < 450 else (-1.0 if kcal > 700 else 0.0))
    elif any(w in _goal for w in ("volumen", "masa", "ganar", "hipertrofia")):
        score += (1.0 if kcal > 600 else 0.0)

    # 3) Momento
    if momento == "cena":
        score += (-1.5 if kcal > 550 else (0.5 if kcal < 400 else 0.0))
    elif momento == "desayuno":
        score += (-1.0 if kcal > 500 else 0.0)

    # 4) Intención bajo calorías
    if intencion.get("bajo_calorias"):
        score += (1.5 if kcal < 400 else (-1.0 if kcal > 600 else 0.0))

    # 5) Completitud nutricional
    if prot > 0 and carb > 0 and gras > 0:
        score += 0.3

    return score


async def _fallback_garantizado_bd(
    db: Session,
    perfil: Any,
    momento: str,
    tdee: float,
    intencion: Optional[Dict[str, Any]] = None,
    historial_nombres: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    FASE 4.3 — Selector inteligente garantizado.

    1. Filtra por momento (nocturnal hard filter — CAMBIO 1).
    2. Consulta candidatos en BD por rango kcal ajustado por objetivo.
    3. Score cada candidato (_score_plato — CAMBIO 3).
    4. Ordena por score y selecciona top-2 con variedad de proteína (CAMBIO 5).
    5. Smart fallback si BD vacía: crea plato base contextual (CAMBIO 7).

    Retorna ≤2 secciones de tipo 'comida' listas para la respuesta.
    """
    from sqlalchemy import text as _text

    _intencion = intencion or {}
    _excluir   = [_norm(n) for n in (historial_nombres or []) if n]

    _RANGO_MOMENTO: Dict[str, tuple] = {
        "desayuno": (200.0,  500.0),
        "almuerzo": (400.0,  850.0),
        "cena":     (200.0,  550.0),
        "snack":    ( 80.0,  300.0),
    }
    min_k, max_k = _RANGO_MOMENTO.get(momento, (150.0, 1000.0))
    _goal   = getattr(perfil, "goal", "") or ""
    _pct    = _obtener_pct_por_objetivo(_goal).get(momento, 0.40)
    max_k   = min(max_k, _pct * tdee)
    # Si pide alto proteína, ampliar rango para no quedar sin candidatos
    if _intencion.get("alto_proteina"):
        max_k = max(max_k, min(750.0, tdee * 0.45))

    # ── Consultar candidatos (más pool para scoring) ─────────────────────────
    try:
        rows = db.execute(_text(
            "SELECT p.id, p.nombre,"
            " SUM(a.calorias_100g * pi2.gramos / 100.0) AS kcal_total"
            " FROM platos p"
            " JOIN plato_ingredientes pi2 ON pi2.plato_id = p.id"
            " JOIN alimentos a ON a.id = pi2.alimento_id"
            " WHERE p.nombre IS NOT NULL"
            " GROUP BY p.id, p.nombre"
            " HAVING SUM(a.calorias_100g * pi2.gramos / 100.0) BETWEEN :min_k AND :max_k"
            " ORDER BY RANDOM() LIMIT 15"
        ), {"min_k": min_k, "max_k": max_k}).fetchall()
    except Exception as _e_q:
        logger.error("[FASE4.3] fallback BD query falló: %s", _e_q)
        rows = []

    # ── Enriquecer, filtrar y puntuar candidatos ─────────────────────────────
    candidatos: List[tuple] = []  # (score, plato_data, nombre_str, pid)
    for row in rows:
        _pid, _nombre = row[0], row[1]
        if not _nombre:
            continue

        # CAMBIO 1: filtro nocturno (mismo criterio que el pipeline principal)
        _rechazar_sem, _ = _validar_momento_semantico(_norm(_nombre), momento)
        if _rechazar_sem:
            logger.debug("[FASE4.3] '%s' excluido por momento='%s'", _nombre, momento)
            continue

        # Excluir nombres similares al historial reciente
        _nn_cand = _norm(_nombre)
        if any(
            difflib.SequenceMatcher(None, _nn_cand, excl).ratio() > 0.75
            for excl in _excluir
        ):
            continue

        _nn = _norm_nombre_plato(str(_nombre))
        try:
            plato_data = _buscar_plato_bd_por_nombre(db, _nn)
        except Exception:
            continue
        if not plato_data or float(plato_data.get("calorias") or 0) <= 0:
            continue

        _score = _score_plato(plato_data, perfil, _intencion, momento)
        candidatos.append((_score, plato_data, _nombre, _pid))

    # CAMBIO 4: ordenar por score
    candidatos.sort(key=lambda x: x[0], reverse=True)

    # ── Seleccionar top-2 con variedad de proteína (CAMBIO 5) ────────────────
    fallback_secs: List[Dict[str, Any]] = []
    _proteinas_usadas: set[str] = set()

    for _score, plato_data, _nombre, _pid in candidatos:
        if len(fallback_secs) >= 2:
            break
        _prot_key = _detectar_proteina_plato(
            _norm(_nombre), plato_data.get("ingredientes") or []
        )
        # Permitir "otro" siempre; para proteínas conocidas exigir variedad
        if _prot_key != "otro" and _prot_key in _proteinas_usadas:
            continue
        _proteinas_usadas.add(_prot_key)

        _cid   = str(uuid.uuid4())
        _kcal  = float(plato_data.get("calorias") or 0)
        _pay   = {
            "calorias":        _kcal,
            "proteinas_g":     float(plato_data.get("proteinas_g") or 0),
            "carbohidratos_g": float(plato_data.get("carbohidratos_g") or 0),
            "grasas_g":        float(plato_data.get("grasas_g") or 0),
            "nombre":          str(_nombre),
            "ingredientes":    plato_data.get("ingredientes") or [],
        }
        set_consulta_cached(_cid, _pay)
        _mac = (
            f"Cal: {_pay['calorias']}kcal | P: {_pay['proteinas_g']}g | "
            f"C: {_pay['carbohidratos_g']}g | G: {_pay['grasas_g']}g"
        )
        fallback_secs.append({
            "tipo":                "comida",
            "nombre":              str(_nombre),
            "macros":              _mac,
            "macros_cache":        _mac,
            "macros_normalizados": {
                "kcal":            _kcal,
                "proteinas_g":     _pay["proteinas_g"],
                "carbohidratos_g": _pay["carbohidratos_g"],
                "grasas_g":        _pay["grasas_g"],
            },
            "ingredientes":        _pay["ingredientes"],
            "preparacion":         plato_data.get("preparacion") or [],
            "consulta_id":         _cid,
            "_fallback_bd":        True,
        })
        logger.info(
            "[FASE4.3] fallback: '%s' (score=%.2f, %.0f kcal, prot=%s)",
            _nombre, _score, _kcal, _prot_key,
        )

    # CAMBIO 7: Smart fallback — si BD no devolvió nada, crear plato base contextual
    if not fallback_secs:
        _plato_base = _elegir_plato_base_contextual(_intencion, momento)
        logger.warning(
            "[FASE4.3] BD vacía (momento=%s, rango=%.0f-%.0f kcal) — "
            "smart fallback: '%s'",
            momento, min_k, max_k, _plato_base,
        )
        try:
            from app.services.plato_constructor import crear_plato_dinamico as _cpd
            _nuevo = await _cpd(db, _plato_base)
            if _nuevo:
                _data_b = _buscar_plato_bd_por_nombre(db, _nuevo.nombre_normalizado)
                if _data_b and float(_data_b.get("calorias") or 0) > 0:
                    _cid2  = str(uuid.uuid4())
                    _kcal2 = float(_data_b.get("calorias") or 0)
                    _pay2  = {
                        "calorias":        _kcal2,
                        "proteinas_g":     float(_data_b.get("proteinas_g") or 0),
                        "carbohidratos_g": float(_data_b.get("carbohidratos_g") or 0),
                        "grasas_g":        float(_data_b.get("grasas_g") or 0),
                        "nombre":          _nuevo.nombre,
                        "ingredientes":    _data_b.get("ingredientes") or [],
                    }
                    set_consulta_cached(_cid2, _pay2)
                    _mac2 = (
                        f"Cal: {_pay2['calorias']}kcal | P: {_pay2['proteinas_g']}g | "
                        f"C: {_pay2['carbohidratos_g']}g | G: {_pay2['grasas_g']}g"
                    )
                    fallback_secs.append({
                        "tipo":                "comida",
                        "nombre":              _nuevo.nombre,
                        "macros":              _mac2,
                        "macros_cache":        _mac2,
                        "macros_normalizados": {
                            "kcal":            _kcal2,
                            "proteinas_g":     _pay2["proteinas_g"],
                            "carbohidratos_g": _pay2["carbohidratos_g"],
                            "grasas_g":        _pay2["grasas_g"],
                        },
                        "ingredientes":        _pay2["ingredientes"],
                        "preparacion":         _data_b.get("preparacion") or [],
                        "consulta_id":         _cid2,
                        "_fallback_bd":        True,
                        "_smart_fallback":     True,
                    })
                    logger.info(
                        "[FASE4.3] Smart fallback: '%s' creado (%.0f kcal)",
                        _nuevo.nombre, _kcal2,
                    )
        except Exception as _e_sf:
            logger.error("[FASE4.3] Smart fallback falló para '%s': %s", _plato_base, _e_sf)

    if not fallback_secs:
        logger.error(
            "[FASE4.3] Fallback completamente vacío — momento=%s", momento
        )
    return fallback_secs


def _postprocesar_secciones(respuesta_estructurada: Dict[str, Any]) -> None:
    """
    FASE 4 — Post-proceso obligatorio antes de enviar a Flutter.
    Elimina secciones de comida con kcal<=0, macros_normalizados ausentes
    o marcadas con _rechazar=True. Preserva secciones no-comida.
    """
    secciones_orig = respuesta_estructurada.get("secciones") or []
    limpias = []
    for s in secciones_orig:
        if s.get("tipo") != "comida":
            limpias.append(s)
            continue
        if s.get("_rechazar"):
            continue
        mn = s.get("macros_normalizados") or {}
        if float(mn.get("kcal") or 0) <= 0:
            logger.warning(
                "[FASE4] post-proceso: '%s' eliminada — kcal=0",
                s.get("nombre") or "?",
            )
            continue
        limpias.append(s)
    respuesta_estructurada["secciones"] = limpias


# ─────────────────────────────────────────────────────────────────────────────


async def procesar_secciones_comida(
    respuesta_estructurada: Dict[str, Any],
    perfil: Any,
    db: Optional[Session] = None,
    mensaje_original: Optional[str] = None,
) -> None:
    """Cachea secciones de comida para consistencia de registro (mutación in-place)."""
    # Leer perfil.id UNA SOLA VEZ antes de cualquier operación DB.
    # Después de db.commit() / db.rollback() SQLAlchemy expira el objeto
    # y cada acceso a .id dispararía un SELECT; si la transacción está
    # abortada ese SELECT falla con InFailedSqlTransaction.
    _perfil_id   = perfil.id
    _perfil_goal = getattr(perfil, "goal", None)

    _platos_del_mensaje = _extraer_platos_del_mensaje(mensaje_original or "")

    # ── Filtro calórico: TDEE y momento calculados una sola vez por llamada ────
    _tdee_filtro    = _calcular_tdee_perfil(perfil)
    _momento_filtro = _inferir_momento_actual()
    _secciones_rechazadas_log: List[str] = []

    # FASE 4.3: Detectar intención nutricional del usuario (para scoring en fallback)
    _intencion_usuario = _interpretar_intencion_usuario(mensaje_original or "")

    for seccion in respuesta_estructurada.get("secciones", []):
        if seccion.get("tipo") != "comida":
            continue

        # Asegurar que ingredientes/preparación sean listas y reparar ingredientes vacíos.
        _reparar_ingredientes_desde_preparacion(seccion)

        # Bloqueo de ingredientes raros/no deseados en Perú (hard guard, aunque el LLM insista).
        nombre_sec = str(seccion.get("nombre") or "")
        ing_src = seccion.get("ingredientes") or []
        ing_filtrados = []
        for ing in ing_src:
            s = str(ing)
            low = s.lower()
            if "estrag" in low or "tarragon" in low:
                # Reemplazo conservador por hierbas típicas (sin alterar kcal si había).
                s = re.sub(r"(?i)estrag[oó]n|tarragon", "perejil", s)
            ing_filtrados.append(s)
        seccion["ingredientes"] = ing_filtrados
        # También limpiar el nombre del plato si se coló.
        if "estrag" in nombre_sec.lower() or "tarragon" in nombre_sec.lower():
            seccion["nombre"] = re.sub(r"(?i)con\s+(estrag[oó]n|tarragon)", "con perejil", nombre_sec).strip()

        ing_list = seccion.get("ingredientes") or []
        cal_suma = 0.0
        prot_suma = 0.0
        for ing in ing_list:
            m_cal = re.search(r"(\d+(?:\.\d+)?)\s*kcal", ing, re.IGNORECASE)
            m_prot = re.search(r"(\d+(?:\.\d+)?)\s*g\s*prote", ing, re.IGNORECASE)
            if m_cal:
                cal_suma += float(m_cal.group(1))
            if m_prot:
                prot_suma += float(m_prot.group(1))

        nombre_bruto_pre = seccion.get("nombre") or "Comida"
        nombre_norm_pre = _norm_nombre_plato(str(nombre_bruto_pre))

        # 0) DB-first para platos: si ya existe, congelar macros (anti-inconsistencia).
        macros_parsed: Optional[Dict[str, Any]] = None
        if db and nombre_norm_pre:
            try:
                macros_parsed = _buscar_plato_bd_por_nombre(db, nombre_norm_pre)
            except Exception:
                try:
                    db.rollback()
                except Exception:
                    pass
                macros_parsed = None

        # 0b) Rescue: si nombre es genérico ("Sugerencia 1") y hay platos del mensaje original,
        #     intentar buscar por esos nombres en BD y también renombrar la sección.
        if not macros_parsed and db and _RE_NOMBRE_GENERICO.match(nombre_norm_pre):
            for _plato_msg in _platos_del_mensaje:
                _nn_msg = _norm_nombre_plato(_plato_msg)
                if not _nn_msg:
                    continue
                try:
                    _found = _buscar_plato_bd_por_nombre(db, _nn_msg)
                except Exception:
                    try:
                        db.rollback()
                    except Exception:
                        pass
                    _found = None
                if _found:
                    macros_parsed = _found
                    # Renombrar la sección con el nombre real del mensaje
                    seccion["nombre"] = _plato_msg.title()
                    nombre_bruto_pre = seccion["nombre"]
                    nombre_norm_pre = _nn_msg
                    break

        # 0c) Rescue adicional: si el LLM generó un nombre diferente al que dijo el usuario
        #     y ese nombre no está en BD, buscar los nombres del mensaje original en BD.
        #     Ejemplo: user dice "avena con platano" → LLM dice "Tostada de Avena con Plátano" (no en BD)
        if not macros_parsed and db and _platos_del_mensaje:
            for _plato_msg in _platos_del_mensaje:
                _nn_msg = _norm_nombre_plato(_plato_msg)
                if not _nn_msg or _nn_msg == nombre_norm_pre:
                    continue
                try:
                    _found = _buscar_plato_bd_por_nombre(db, _nn_msg)
                except Exception:
                    try:
                        db.rollback()
                    except Exception:
                        pass
                    _found = None
                if _found:
                    macros_parsed = _found
                    seccion["nombre"] = _plato_msg.title()
                    nombre_bruto_pre = seccion["nombre"]
                    nombre_norm_pre = _nn_msg
                    break
        # 0d) Sugerencias proactivas: si el plato no está en BD y tiene ≥2 palabras,
        #     construirlo dinámicamente para que Flutter siempre muestre ingredientes reales.
        if not macros_parsed and db and len(nombre_norm_pre.split()) >= 2:
            try:
                from app.services.plato_constructor import crear_plato_dinamico as _cpd
                _plato_nuevo = await _cpd(db, str(nombre_bruto_pre))
                if _plato_nuevo:
                    macros_parsed = _buscar_plato_bd_por_nombre(
                        db, _plato_nuevo.nombre_normalizado
                    )
            except Exception as _e_cpd:
                logger.error("[procesar_secciones] crear_plato_dinamico falló para '%s': %s", nombre_bruto_pre, _e_cpd)

        # 0e) FASE 4: si crear_plato_dinamico falló y hay ingredientes del LLM,
        #     resolver los ingredientes directamente en BD (Trinity Nutricional).
        #     Invierte el flujo: BD calcula macros, LLM solo propone nombres/gramos.
        if not macros_parsed and db and ing_list:
            try:
                _bd_resolv = await _resolver_y_construir_desde_bd(
                    db, str(nombre_bruto_pre), ing_list
                )
                if _bd_resolv:
                    macros_parsed = _bd_resolv
                    if _bd_resolv.get("ingredientes"):
                        seccion["ingredientes"] = _bd_resolv["ingredientes"]
                    logger.info(
                        "[FASE4] 0e: macros desde BD para '%s' → %.0f kcal "
                        "(%d ingredientes resueltos)",
                        nombre_bruto_pre,
                        _bd_resolv.get("calorias", 0),
                        len(_bd_resolv.get("ingredientes") or []),
                    )
            except Exception as _e_0e:
                logger.warning("[FASE4] 0e falló para '%s': %s", nombre_bruto_pre, _e_0e)

        # BLOQUE 4: Filtro semántico para platos servidos desde BD antes de recomendar.
        # Si el plato tiene conflicto nombre↔ingredientes (ej. ceviche con queso guardado
        # antes de FASE 3), se descarta y cae al fallback LLM/estimación.
        if macros_parsed and macros_parsed.get("plato_id") is not None:
            try:
                from app.services.plato_constructor import validar_semantica_plato as _val_sem
                _ings_bd = macros_parsed.get("ingredientes") or []
                _ings_nombres = []
                for _ing_str in _ings_bd:
                    _m_ing = re.match(r"^(.+?)\s+\d+g\s+→", str(_ing_str))
                    if _m_ing:
                        _ings_nombres.append(_m_ing.group(1))
                _ok_bd, _motivo_bd = _val_sem(str(nombre_bruto_pre), _ings_nombres)
                if not _ok_bd:
                    logger.warning(
                        "Plato BD id=%s '%s' filtrado por semántica: %s",
                        macros_parsed["plato_id"], nombre_bruto_pre, _motivo_bd,
                    )
                    macros_parsed = None
            except Exception:
                pass  # filtro no bloqueante — no interrumpe el flujo

        # CAMBIO 4 — Segunda capa: coherencia nombre↔ingrediente principal en platos BD.
        # Cubre casos donde _PLANTILLAS_PLATOS no tiene regla (ej. "pescado" genérico
        # resuelto con pollo en la BD por un bug anterior de persistencia).
        if macros_parsed and macros_parsed.get("plato_id") is not None:
            try:
                from app.services.plato_constructor import (
                    _validar_coherencia_nombre_ingredientes as _val_coh,
                    _norm as _norm_plato,
                )
                _ings_bd2 = macros_parsed.get("ingredientes") or []
                _ings_nombres2 = []
                for _ing_str2 in _ings_bd2:
                    _s2 = str(_ing_str2)
                    # Intentar varios formatos de ingrediente:
                    # Formato 1: "Nombre 150g → xxx kcal"
                    _m2 = re.match(r"^(.+?)\s+\d+g\s+[→–-]", _s2)
                    if _m2:
                        _ings_nombres2.append(_m2.group(1).strip())
                        continue
                    # Formato 2: "Nombre 150g (xxx kcal)"
                    _m2b = re.match(r"^(.+?)\s+\d+g", _s2)
                    if _m2b:
                        _ings_nombres2.append(_m2b.group(1).strip())
                        continue
                    # Formato 3: string simple sin gramos
                    if len(_s2.strip()) > 2:
                        _ings_nombres2.append(_s2.strip())

                # GUARD CRITICO: si no pudimos extraer ningún nombre de ingrediente
                # NO descartar el plato — es un fallo de parseo, no datos corruptos.
                # Marcar solo en debug para no silenciar bugs reales.
                if not _ings_nombres2:
                    logger.debug(
                        "CAMBIO 4: plato BD '%s' con %d ingredientes pero formato no parseable — "
                        "skip coherencia (plato preservado)",
                        nombre_bruto_pre, len(_ings_bd2),
                    )
                else:
                    class _AlimProxy:
                        def __init__(self, nombre: str):
                            self.nombre = nombre

                    _resueltos_proxy = [(_AlimProxy(n), 100) for n in _ings_nombres2]
                    _nombre_bd_norm = _norm_plato(str(nombre_bruto_pre))
                    _ok_coh2, _motivo_coh2 = _val_coh(_nombre_bd_norm, _resueltos_proxy)
                    if not _ok_coh2:
                        logger.warning(
                            "Plato BD id=%s '%s' descartado por incoherencia nombre↔ingredientes: %s",
                            macros_parsed["plato_id"], nombre_bruto_pre, _motivo_coh2,
                        )
                        macros_parsed = None

                    # CAMBIO 4c — Validar ingredientes mencionados en el nombre del plato BD.
                    # Cubre: "Tortilla de Huevo con Perejil" servida desde BD con ings
                    # Huevo+Aceite+Pan (sin perejil). El fuzzy match (sim≥0.82) reutiliza
                    # un plato parecido pero con nombre diferente — los ingredientes del
                    # nombre nuevo no coinciden con los del plato viejo de BD.
                    if macros_parsed:
                        try:
                            from app.services.plato_constructor import (
                                _validar_ingredientes_en_nombre as _val_ien,
                            )
                            _ok_ien, _motivo_ien = _val_ien(_nombre_bd_norm, _resueltos_proxy)
                            if not _ok_ien:
                                logger.warning(
                                    "Plato BD id=%s '%s' descartado — nombre menciona "
                                    "ingredientes ausentes en BD: %s",
                                    macros_parsed["plato_id"], nombre_bruto_pre, _motivo_ien,
                                )
                                macros_parsed = None
                        except Exception as _e_ien:
                            logger.debug("CAMBIO 4c no bloqueante falló: %s", _e_ien)

                # CAMBIO 4b — Bloquear kcal=0 en platos BD (dato corrupto/incompleto)
                if macros_parsed:
                    _kcal_bd = float(macros_parsed.get("calorias") or 0)
                    if _kcal_bd <= 0:
                        logger.warning(
                            "Plato BD id=%s '%s' descartado: kcal=0 — ingredientes sin macros en BD",
                            macros_parsed["plato_id"], nombre_bruto_pre,
                        )
                        macros_parsed = None
            except Exception as _e_coh4:
                logger.debug("CAMBIO 4 no bloqueante falló para '%s': %s", nombre_bruto_pre, _e_coh4)


        # 1) Fallback: si no hay plato en BD, reusar recientes o parsear string LLM.
        reciente = None
        if not macros_parsed:
            reciente = _buscar_reciente_por_nombre(_perfil_id, str(nombre_bruto_pre))
            if reciente:
                macros_parsed = {
                    "calorias": float(reciente.get("calorias") or 0),
                    "proteinas_g": float(reciente.get("proteinas_g") or 0),
                    "carbohidratos_g": float(reciente.get("carbohidratos_g") or 0),
                    "grasas_g": float(reciente.get("grasas_g") or 0),
                }
            else:
                macros_parsed = parsear_macros_de_texto(
                    seccion.get("macros") or "",
                    _perfil_goal,
                )
                # Atwater sanity check sobre macros estimadas por LLM (log-only, no bloqueante).
                # Rechazar aquí mostraría ceros en Flutter, que es peor UX que el estimado.
                if macros_parsed:
                    _c_est = float(macros_parsed.get("calorias") or 0)
                    _p_est = float(macros_parsed.get("proteinas_g") or 0)
                    _cb_est = float(macros_parsed.get("carbohidratos_g") or 0)
                    _g_est = float(macros_parsed.get("grasas_g") or 0)
                    _ok_est, _mot_est = validar_macros_atwater(_c_est, _p_est, _cb_est, _g_est)
                    if not _ok_est:
                        logger.warning(
                            "[LLM macros] '%s' estimación incoherente — %s",
                            nombre_bruto_pre, _mot_est,
                        )
        else:
            # Si viene desde BD, siempre usar ingredientes y preparación de BD.
            # Los datos del LLM (kcal inventadas, pasos sin sentido) se descartan.
            ing_bd = macros_parsed.get("ingredientes") or []
            if ing_bd:
                seccion["ingredientes"] = ing_bd
            # Preparación: preferir siempre la BD sobre la del LLM
            prep_bd = macros_parsed.get("preparacion") or []
            if prep_bd:
                seccion["preparacion"] = prep_bd
            elif not (seccion.get("preparacion") or []):
                seccion["preparacion"] = []
            if not (seccion.get("nota") or "").strip() and macros_parsed.get("nota"):
                seccion["nota"] = macros_parsed.get("nota") or ""
        cal_stats = float((macros_parsed or {}).get("calorias") or 0)

        # Si el plato viene de la BD (tiene plato_id), NUNCA sobreescribir con
        # macros calculadas del texto LLM. La BD es la fuente de verdad.
        _viene_de_bd = macros_parsed is not None and macros_parsed.get("plato_id") is not None

        # ── GUARD CRÍTICO: plato descartado por validación (CAMBIO 4 u otros) ─
        # Si macros_parsed fue anulado a None por alguna validación (incoherencia
        # nombre↔ingredientes, kcal=0 en BD, semántica, etc.) Y no hay reciente
        # de caché, el plato no tiene datos válidos para mostrar. Saltar ANTES
        # de que el fallback "elif not macros_parsed → kcal=0" lo reinstancie
        # con datos en cero, lo que causa filtros falsos y logs con motivo vacío.
        if macros_parsed is None and not reciente and cal_suma <= 0:
            logger.info(
                "[pipeline] '%s' descartado — sin datos válidos de macros ni ingredientes",
                nombre_bruto_pre,
            )
            _secciones_rechazadas_log.append(str(nombre_bruto_pre))
            seccion["_rechazar"] = True
            continue

        usar_macros_desde_ingredientes = False
        if (not reciente) and (not _viene_de_bd) and cal_suma > 50:
            if not macros_parsed or cal_stats <= 0:
                usar_macros_desde_ingredientes = True
            elif abs(cal_suma - cal_stats) / max(cal_suma, cal_stats, 1.0) > 0.12:
                usar_macros_desde_ingredientes = True

        if (not reciente) and usar_macros_desde_ingredientes:
            gras_estimada = cal_suma * 0.25 / 9
            carb_estimada = (cal_suma - (prot_suma * 4) - (gras_estimada * 9)) / 4
            macros_parsed = {
                "calorias":        round(cal_suma, 1),
                "proteinas_g":     round(prot_suma, 1),
                "carbohidratos_g": round(max(0, carb_estimada), 1),
                "grasas_g":        round(gras_estimada, 1),
            }
        elif not macros_parsed:
            macros_parsed = {
                "calorias": 0,
                "proteinas_g": 0,
                "carbohidratos_g": 0,
                "grasas_g": 0,
            }

        cal_m = float(macros_parsed.get("calorias") or 0)
        p = float(macros_parsed.get("proteinas_g") or 0)
        c = float(macros_parsed.get("carbohidratos_g") or 0)
        g = float(macros_parsed.get("grasas_g") or 0)

        # Si falta alguna macro (P/C/G) pero hay kcal, completar lo faltante de forma consistente
        # (evita chips incompletos en Flutter cuando el LLM entrega solo 1–2 macros).
        # No aplicar si los macros ya vienen verificados desde la BD.
        if (not reciente) and (not _viene_de_bd) and cal_m > 30:
            missing = sum(1 for v in (p, c, g) if v <= 0.0)
            if missing >= 1 and (p > 0.0 or c > 0.0 or g > 0.0):
                est = macros_desde_calorias_pct_clasico(
                    cal_m, _perfil_goal
                )
                # Mantener los no-cero del modelo, rellenar los cero.
                p2 = p if p > 0 else float(est["proteinas_g"])
                c2 = c if c > 0 else float(est["carbohidratos_g"])
                g2 = g if g > 0 else float(est["grasas_g"])
                # Re-escalar a kcal objetivo usando Atwater.
                atw = 4.0 * p2 + 4.0 * c2 + 9.0 * g2
                if atw > 1.0:
                    sc = cal_m / atw
                    macros_parsed = {
                        "calorias": round(cal_m, 1),
                        "proteinas_g": round(max(0.0, p2 * sc), 1),
                        "carbohidratos_g": round(max(0.0, c2 * sc), 1),
                        "grasas_g": round(max(0.0, g2 * sc), 1),
                    }
                    cal_m = float(macros_parsed.get("calorias") or cal_m)
                    p = float(macros_parsed.get("proteinas_g") or 0)
                    c = float(macros_parsed.get("carbohidratos_g") or 0)
                    g = float(macros_parsed.get("grasas_g") or 0)
        if (not reciente) and (not _viene_de_bd) and cal_m > 30:
            atw = 4.0 * p + 4.0 * c + 9.0 * g
            ref_cal = cal_m
            if cal_suma > 50 and abs(cal_suma - cal_m) / max(cal_suma, cal_m, 1.0) <= 0.1:
                ref_cal = cal_suma
            if atw > 1.0 and ref_cal > 1.0 and abs(atw - ref_cal) / ref_cal > 0.11:
                denom = 4.0 * (p + c) + 9.0 * g
                if denom > 1.0:
                    sc = ref_cal / atw
                    macros_parsed = {
                        "calorias": round(ref_cal, 1),
                        "proteinas_g": round(max(0.0, p * sc), 1),
                        "carbohidratos_g": round(max(0.0, c * sc), 1),
                        "grasas_g": round(max(0.0, g * sc), 1),
                    }
                else:
                    est = macros_desde_calorias_pct_clasico(
                        ref_cal, _perfil_goal
                    )
                    macros_parsed = {
                        "calorias": round(ref_cal, 1),
                        "proteinas_g": round(est["proteinas_g"], 1),
                        "carbohidratos_g": round(est["carbohidratos_g"], 1),
                        "grasas_g": round(est["grasas_g"], 1),
                    }

        cal_fin = float(macros_parsed.get("calorias") or 0)
        if (
            (not reciente)
            and (not _viene_de_bd)
            and (not usar_macros_desde_ingredientes)
            and cal_suma > 50
            and cal_fin > 50
            and abs(cal_suma - cal_fin) / max(cal_suma, cal_fin, 1.0) <= 0.1
            and prot_suma < 5.0
        ):
            est = macros_desde_calorias_pct_clasico(
                cal_suma, _perfil_goal
            )
            macros_parsed = {
                "calorias": round(cal_suma, 1),
                "proteinas_g": round(est["proteinas_g"], 1),
                "carbohidratos_g": round(est["carbohidratos_g"], 1),
                "grasas_g": round(est["grasas_g"], 1),
            }

        nombre_bruto = seccion.get("nombre") or "Comida"
        nombre_limpio_raw = re.sub(r"\[.*?\]", "", nombre_bruto).split("[")[0].strip()
        # Limpiar sufijos de bebida/acompañamiento que no forman parte del plato
        # Ej: "Lomo saltado con un vaso de agua" → "Lomo saltado"
        nombre_limpio = _limpiar_nombre_plato_bd(nombre_limpio_raw)
        # Actualizar el nombre en la sección para que la tarjeta muestre el nombre limpio
        if nombre_limpio != nombre_limpio_raw:
            seccion["nombre"] = nombre_limpio

        # Para platos de BD los ingredientes ya tienen kcal correctas.
        # Para platos del LLM: recalcular kcal desde BD (USDA/Groq si no existe)
        # así incluso el PRIMER pedido muestra kcal correctas.
        ing_raw = seccion.get("ingredientes") or []
        if _viene_de_bd or not db:
            ing_con_gramos = [str(i) for i in ing_raw]
        else:
            ing_con_gramos = []
            for i in ing_raw:
                try:
                    ing_con_gramos.append(
                        await _recalcular_ing_async(db, str(i))
                    )
                except Exception:
                    ing_con_gramos.append(_agregar_equivalencia_gramos(str(i)))

        # Siempre devolver los ingredientes con kcal correctas a la sección.
        # Antes sólo se guardaban en el cache pero no en la respuesta → Flutter
        # mostraba los valores inventados por el LLM.
        seccion["ingredientes"] = ing_con_gramos

        # Si el plato NO viene de BD, recalcular la kcal total sumando las kcal
        # reales de cada ingrediente (más preciso que el estimado del LLM).
        _kcal_final   = float(macros_parsed.get("calorias") or 0)
        _prot_final   = float(macros_parsed.get("proteinas_g") or 0)
        _carb_final   = float(macros_parsed.get("carbohidratos_g") or 0)
        _gras_final   = float(macros_parsed.get("grasas_g") or 0)

        if not _viene_de_bd and ing_con_gramos:
            _sum_kcal = 0.0
            for _ig in ing_con_gramos:
                _m = re.search(r"\((\d+(?:\.\d+)?)\s*kcal\)", _ig, re.IGNORECASE)
                if _m:
                    _sum_kcal += float(_m.group(1))
            if _sum_kcal > 10:
                # Recalcular macros proporcionales a la suma real de kcal
                _old = _kcal_final or _sum_kcal
                _sc = _sum_kcal / _old if _old > 0 else 1.0
                _kcal_final = round(_sum_kcal, 1)
                _prot_final = round(_prot_final * _sc, 1)
                _carb_final = round(_carb_final * _sc, 1)
                _gras_final = round(_gras_final * _sc, 1)

        # Siempre redondear a 1 decimal para evitar artefactos de punto flotante
        # (ej. 21.002999... de sumas SQL se convierte en 21.0)
        _kcal_final = round(_kcal_final, 1)
        _prot_final = round(_prot_final, 1)
        _carb_final = round(_carb_final, 1)
        _gras_final = round(_gras_final, 1)

        # ── Filtro calórico por momento del día ───────────────────────────────
        _modo = respuesta_estructurada.get("modo_funcion", "")
        if _modo != "recomendar_nutricion":
            _rechazar, _motivo_kcal = _verificar_rango_calorico(
                _kcal_final, _momento_filtro, _tdee_filtro, _perfil_goal or ""
            )
            if _rechazar:
                logger.warning(
                    "[filtro_kcal] '%s' descartado — %s (tdee=%.0f)",
                    seccion.get("nombre") or "?", _motivo_kcal, _tdee_filtro,
                )
                _secciones_rechazadas_log.append(seccion.get("nombre") or "?")
                seccion["_rechazar"] = True
                continue   # salta cache, DB y macros_normalizados

            # ── CAMBIO 4: Filtro semántico por momento del día ────────────────────
            _nombre_check = _norm(str(seccion.get("nombre") or ""))
            _rechazar_sem, _motivo_sem_momento = _validar_momento_semantico(
                _nombre_check, _momento_filtro or ""
            )
            if _rechazar_sem:
                logger.warning(
                    "[filtro_momento] '%s' descartado — %s",
                    seccion.get("nombre") or "?", _motivo_sem_momento,
                )
                _secciones_rechazadas_log.append(seccion.get("nombre") or "?")
                seccion["_rechazar"] = True
                continue   # salta cache, DB y macros_normalizados
            # ──────────────────────────────────────────────────────────────────────

        consulta_id = str(uuid.uuid4())
        payload = {
            "calorias":        _kcal_final,
            "proteinas_g":     _prot_final,
            "carbohidratos_g": _carb_final,
            "grasas_g":        _gras_final,
            "nombre":          nombre_limpio,
            "ingredientes":    ing_con_gramos,
        }
        set_consulta_cached(consulta_id, payload)
        seccion["consulta_id"] = consulta_id
        macros_canon = (
            f"Cal: {payload['calorias']}kcal | P: {payload['proteinas_g']}g | "
            f"C: {payload['carbohidratos_g']}g | G: {payload['grasas_g']}g"
        )
        seccion["macros"] = macros_canon
        seccion["macros_cache"] = macros_canon
        seccion["macros_normalizados"] = {
            "kcal":            _kcal_final,
            "proteinas_g":     _prot_final,
            "carbohidratos_g": _carb_final,
            "grasas_g":        _gras_final,
        }
        # Solo guardar en recent_meals si el nombre es un plato real (no genérico).
        # Nombres como "Sugerencia 1" contaminan la caché y causan lookups incorrectos.
        if not _RE_NOMBRE_GENERICO.match(_norm_nombre_plato(nombre_limpio)):
            add_user_recent_meal(_perfil_id, payload)

        # 2) Persistir plato en catálogo + historial de recomendaciones.
        if db and nombre_norm_pre:
            try:
                from app.models.plato import Plato, PlatoIngrediente
                from app.models.historial_recomendacion import HistorialRecomendacion

                nombre_norm_bd = _norm_nombre_plato(nombre_limpio)

                # Buscar si el plato ya existe en el catálogo
                plato_obj = (
                    db.query(Plato)
                    .filter(Plato.nombre_normalizado == nombre_norm_bd)
                    .first()
                )

                if not plato_obj:
                    # Guardar solo si la sección es completa (evita basura en BD)
                    ing_list = seccion.get("ingredientes") or []
                    ing_ok   = isinstance(ing_list, list) and len(ing_list) >= 2
                    prep_ok  = isinstance(seccion.get("preparacion"), list) and len(seccion.get("preparacion") or []) >= 2
                    macros_ok = (
                        float(payload["calorias"] or 0) > 0
                        and float(payload["proteinas_g"] or 0) > 0
                        and float(payload["carbohidratos_g"] or 0) > 0
                        and float(payload["grasas_g"] or 0) > 0
                    )
                    # Validación proteína: si el nombre contiene "pescado"/"salmón" los
                    # ingredientes no deben contener pollo/pavo (y viceversa), para evitar
                    # guardar tarjetas con proteína incorrecta.
                    _nombre_low = nombre_limpio.lower()
                    _ings_low   = " ".join(str(i) for i in ing_list).lower()
                    _prot_ok = True
                    if any(w in _nombre_low for w in ("pescado", "salmón", "salmon", "atún", "atun", "tilapia")):
                        if any(w in _ings_low for w in ("pechuga", "pollo", "pavo", "carne de res", "cerdo")):
                            _prot_ok = False
                            print(f"[Persistencia] Proteína incorrecta en '{nombre_limpio}': se descarta")
                    elif any(w in _nombre_low for w in ("pollo",)):
                        if "pavo" in _ings_low and "pollo" not in _ings_low:
                            _prot_ok = False
                            print(f"[Persistencia] Proteína incorrecta en '{nombre_limpio}': pavo en plato de pollo")
                    if ing_ok and prep_ok and macros_ok and _prot_ok:
                        plato_obj = Plato(
                            nombre=nombre_limpio[:255],
                            nombre_normalizado=nombre_norm_bd[:255],
                            tipo_plato="cualquiera",
                            preparacion=seccion.get("preparacion") or [],
                            nota=(seccion.get("nota") or None),
                            origen="llm",
                        )
                        db.add(plato_obj)
                        db.flush()  # obtener plato_obj.id sin hacer commit

                        # Vincular ingredientes a alimentos reales (USDA/Groq como fallback)
                        comps_bd, _ = await _construir_componentes_bd_async(db, ing_list)
                        if comps_bd:
                            for orden_i, comp in enumerate(comps_bd, start=1):
                                db.add(PlatoIngrediente(
                                    plato_id    = plato_obj.id,
                                    alimento_id = comp["alimento_id"],
                                    gramos      = comp["gramos"],
                                    orden       = orden_i,
                                ))

                # Guardar snapshot en historial (siempre, haya o no plato en catálogo)
                momento = _momento_filtro

                db.add(HistorialRecomendacion(
                    client_id       = _perfil_id,
                    plato_id        = plato_obj.id if plato_obj else None,
                    nombre_plato    = nombre_limpio[:255],
                    calorias        = float(payload["calorias"] or 0),
                    proteinas_g     = float(payload["proteinas_g"] or 0),
                    carbohidratos_g = float(payload["carbohidratos_g"] or 0),
                    grasas_g        = float(payload["grasas_g"] or 0),
                    momento_dia     = momento,
                    fue_consumido   = False,
                ))
                db.commit()
            except Exception:
                # No bloquear la respuesta del asistente si la persistencia falla.
                try:
                    db.rollback()
                except Exception:
                    pass

    # ── Eliminar secciones rechazadas por filtro calórico (capa individual) ──
    if _secciones_rechazadas_log:
        respuesta_estructurada["secciones"] = [
            s for s in respuesta_estructurada.get("secciones", [])
            if not s.get("_rechazar")
        ]
        logger.info(
            "[filtro_kcal] %d sección(es) eliminada(s): %s",
            len(_secciones_rechazadas_log), _secciones_rechazadas_log,
        )

    # ── VALIDACIÓN GLOBAL POR COMIDA (capa de segunda defensa) ───────────────
    # No aplicar si es recomendación, ya que las secciones son opciones excluyentes.
    _modo = respuesta_estructurada.get("modo_funcion", "")
    if _modo != "recomendar_nutricion":
        _secciones_validas = [
            s for s in respuesta_estructurada.get("secciones", [])
            if s.get("tipo") == "comida" and s.get("macros_normalizados")
        ]

        if len(_secciones_validas) > 1:
            _pct_map_global = _obtener_pct_por_objetivo(_perfil_goal or "")
            _pct_max_global = _pct_map_global.get(_momento_filtro, 0.40)
            _limite_global  = _pct_max_global * _tdee_filtro
            _total_kcal     = sum(
                float(s.get("macros_normalizados", {}).get("kcal") or 0)
                for s in _secciones_validas
            )

            logger.info(
                "[filtro_global] momento=%s total=%.0f kcal límite=%.0f kcal "
                "(%.0f%% TDEE=%.0f) objetivo='%s'",
                _momento_filtro, _total_kcal, _limite_global,
                _pct_max_global * 100, _tdee_filtro, _perfil_goal or "mantener",
            )

            # Cambio 4: eliminar plato más calórico si el total excede el límite
            if _total_kcal > _limite_global and len(_secciones_validas) > 1:
                _secciones_validas.sort(
                    key=lambda s: float(s.get("macros_normalizados", {}).get("kcal") or 0),
                    reverse=True,
                )
                _elim = _secciones_validas.pop(0)
                _kcal_elim = float(_elim.get("macros_normalizados", {}).get("kcal") or 0)
                logger.warning(
                    "[filtro_global] total %.0f kcal > límite %.0f → eliminado '%s' "
                    "(%.0f kcal, %.0f%% del total, momento=%s)",
                    _total_kcal, _limite_global,
                    _elim.get("nombre") or "?",
                    _kcal_elim,
                    (_kcal_elim / _total_kcal * 100) if _total_kcal > 0 else 0,
                    _momento_filtro,
                )
                _total_kcal -= _kcal_elim
                respuesta_estructurada["secciones"] = _secciones_validas

            # Cambio 5: regla de dominancia — plato >70% del total con múltiples secciones
            if len(_secciones_validas) > 1 and _total_kcal > 0:
                for _s in list(_secciones_validas):
                    _kcal_s = float(_s.get("macros_normalizados", {}).get("kcal") or 0)
                    if _kcal_s / _total_kcal > 0.70:
                        logger.warning(
                            "[filtro_dominancia] '%s' (%.0f kcal = %.0f%% del total %.0f) "
                            "eliminado por dominancia (momento=%s)",
                            _s.get("nombre") or "?",
                            _kcal_s,
                            _kcal_s / _total_kcal * 100,
                            _total_kcal,
                            _momento_filtro,
                    )
                    _secciones_validas.remove(_s)
                    break  # solo eliminar el dominante más calórico por llamada
            respuesta_estructurada["secciones"] = _secciones_validas

    # ── FASE 4: Garantía "nunca respuesta vacía" ──────────────────────────────
    # Activa si todas las secciones de comida fueron rechazadas o tienen kcal=0.
    _comidas_con_datos = [
        s for s in respuesta_estructurada.get("secciones", [])
        if s.get("tipo") == "comida"
        and not s.get("_rechazar")
        and float((s.get("macros_normalizados") or {}).get("kcal") or 0) > 0
    ]
    _modo = respuesta_estructurada.get("modo_funcion", "")
    if not _comidas_con_datos and db and _modo == "recomendar_nutricion":
        logger.warning(
            "[FASE4] Todas las comidas rechazadas o kcal=0 — "
            "activando fallback garantizado (momento=%s, tdee=%.0f)",
            _momento_filtro, _tdee_filtro,
        )
        try:
            _fb_secs = await _fallback_garantizado_bd(
                db, perfil, _momento_filtro, _tdee_filtro,
                intencion=_intencion_usuario,
                historial_nombres=_secciones_rechazadas_log,
            )
            if _fb_secs:
                respuesta_estructurada["secciones"].extend(_fb_secs)
                logger.info(
                    "[FASE4.3] Fallback garantizado: %d plato(s) seleccionados "
                    "(intencion=%s)",
                    len(_fb_secs), _intencion_usuario,
                )
            else:
                logger.error(
                    "[FASE4.3] Fallback completamente vacío — momento=%s",
                    _momento_filtro,
                )
        except Exception as _e_fb:
            logger.error("[FASE4.3] Fallback garantizado falló: %s", _e_fb)

    # CAMBIO 4: Ordenar secciones de comida válidas por score de intención
    _secs_comida_ok = [
        s for s in respuesta_estructurada.get("secciones", [])
        if s.get("tipo") == "comida" and not s.get("_rechazar")
        and float((s.get("macros_normalizados") or {}).get("kcal") or 0) > 0
    ]
    if len(_secs_comida_ok) > 1:
        _secs_comida_ok.sort(
            key=lambda s: _score_plato(
                {
                    "calorias":        float((s.get("macros_normalizados") or {}).get("kcal") or 0),
                    "proteinas_g":     float((s.get("macros_normalizados") or {}).get("proteinas_g") or 0),
                    "carbohidratos_g": float((s.get("macros_normalizados") or {}).get("carbohidratos_g") or 0),
                    "grasas_g":        float((s.get("macros_normalizados") or {}).get("grasas_g") or 0),
                },
                perfil, _intencion_usuario, _momento_filtro,
            ),
            reverse=True,
        )
        _otras_secs = [
            s for s in respuesta_estructurada.get("secciones", [])
            if s.get("tipo") != "comida"
        ]
        respuesta_estructurada["secciones"] = _otras_secs + _secs_comida_ok

    # ── FASE 4: Post-proceso — eliminar comidas con datos corruptos ───────────
    _postprocesar_secciones(respuesta_estructurada)


def fuzzy_match_comidas_recientes(mensaje: str, perfil: Any) -> Optional[Dict[str, Any]]:
    """Busca coincidencias de comidas recientes por similaridad."""
    msg_texto = mensaje.lower().strip()
    comidas_recientes = get_user_recent_meals(perfil.id)
    if not comidas_recientes:
        return None

    search_candidates = []
    for m in comidas_recientes:
        for ing in m.get("ingredientes", []):
            ing_match = re.search(r"de\s+(.*?)\s*\(", ing, re.IGNORECASE)
            if not ing_match:
                ing_match = re.search(r"\*\s*\d+.*?\s+(.*?)\s*\(", ing)
            if ing_match:
                nombre_ing = ing_match.group(1).strip().lower()
                search_candidates.append((nombre_ing, m, ing))
        search_candidates.append((m["nombre"].lower(), m, None))

    msg_limpio = msg_texto
    for ruido in [
        "registra que me ", "registra que ", "registra me ", "registra ",
        "comí ", "comi ", "cómo ", "como ", "cené ", "almorcé ", "desayuné ",
        "cene ", "almorce ", "desayune ", "he comido ", "he cenado ",
        "un ", "una ", "unos ", "unas ",
    ]:
        msg_limpio = msg_limpio.replace(ruido, "")
    msg_limpio = msg_limpio.strip()

    nombres_para_difflib = [c[0] for c in search_candidates]
    coincidencias = difflib.get_close_matches(msg_limpio, nombres_para_difflib, n=1, cutoff=0.75)

    if not coincidencias:
        return None

    match_nombre = coincidencias[0]
    for c_nombre, m_payload, item_str in search_candidates:
        if c_nombre == match_nombre:
            if item_str:
                cals_match = re.search(r"\((\d+)\s*kcal", item_str)
                p_match = re.search(r"P:\s*(\d+[,.]?\d*)g", item_str)
                c_match = re.search(r"C:\s*(\d+[,.]?\d*)g", item_str)
                g_match = re.search(r"G:\s*(\d+[,.]?\d*)g", item_str)
                if cals_match:
                    try:
                        return {
                            "nombre": match_nombre.capitalize(),
                            "calorias": float(cals_match.group(1)),
                            "proteinas_g": float(p_match.group(1).replace(",", ".")) if p_match else 0,
                            "carbohidratos_g": float(c_match.group(1).replace(",", ".")) if c_match else 0,
                            "grasas_g": float(g_match.group(1).replace(",", ".")) if g_match else 0,
                        }
                    except Exception:
                        pass
            return m_payload
    return None


def advertencia_alimentos_prohibidos(perfil: Any, alimentos_detectados: List[str]) -> Optional[str]:
    """
    Si algún alimento detectado coincide con la lista del nutricionista, devuelve texto de aviso.
    """
    prohibidos = [f.lower().strip() for f in (getattr(perfil, "forbidden_foods", None) or [])]
    if not prohibidos or not alimentos_detectados:
        return None
    coincidencias: List[str] = []
    for alimento in alimentos_detectados:
        al_low = alimento.lower().strip()
        for prohib in prohibidos:
            if prohib in al_low or al_low in prohib:
                coincidencias.append(alimento)
                break
    if not coincidencias:
        return None
    nombres = ", ".join(coincidencias)
    return (
        f"⚠️ Atención: '{nombres}' está en tu lista de alimentos prohibidos "
        f"definida por tu nutricionista. Se registró, pero te recomiendo "
        f"consultarlo con tu profesional de salud."
    )


def registrar_comida_desde_payload_tarjeta(
    payload: Dict[str, Any],
    perfil: Any,
    progreso: Any,
    db: Session,
) -> Dict[str, Any]:
    """
    Suma calorías/macros al progreso y hace upsert de PreferenciaAlimento (tarjeta o confirmación).
    No hace commit.
    """
    calorias = float(payload.get("calorias", 0) or 0)
    proteinas_g = float(payload.get("proteinas_g", 0) or 0)
    carbohidratos_g = float(payload.get("carbohidratos_g", 0) or 0)
    grasas_g = float(payload.get("grasas_g", 0) or 0)
    nombre = (payload.get("nombre") or "Comida").strip()

    progreso.calorias_consumidas = (progreso.calorias_consumidas or 0) + calorias
    progreso.proteinas_consumidas = (progreso.proteinas_consumidas or 0) + proteinas_g
    progreso.carbohidratos_consumidos = (progreso.carbohidratos_consumidos or 0) + carbohidratos_g
    progreso.grasas_consumidas = (progreso.grasas_consumidas or 0) + grasas_g

    pref = (
        db.query(PreferenciaAlimento)
        .filter(
            PreferenciaAlimento.client_id == perfil.id,
            sql_func.lower(PreferenciaAlimento.alimento) == nombre.lower(),
        )
        .first()
    )
    if pref:
        pref.frecuencia += 1
        pref.ultima_vez = datetime.now()
        pref.calorias = calorias
        pref.proteinas = proteinas_g
        pref.carbohidratos = carbohidratos_g
        pref.grasas = grasas_g
    else:
        db.add(
            PreferenciaAlimento(
                client_id=perfil.id,
                alimento=nombre.lower(),
                frecuencia=1,
                puntuacion=1.0,
                ultima_vez=datetime.now(),
                calorias=calorias,
                proteinas=proteinas_g,
                carbohidratos=carbohidratos_g,
                grasas=grasas_g,
            )
        )

    return {
        "tipo_detectado": "comida",
        "nombre": nombre,
        "calorias": calorias,
        "proteinas_g": proteinas_g,
        "carbohidratos_g": carbohidratos_g,
        "grasas_g": grasas_g,
    }


def aplicar_extraccion_nlp_comida_a_progreso(extraccion: Dict[str, Any], progreso: Any) -> None:
    """Suma al progreso los macros de una extracción NLP marcada como comida."""
    if not extraccion.get("es_comida"):
        return
    calorias = extraccion.get("calorias", 0) or 0
    progreso.calorias_consumidas = (progreso.calorias_consumidas or 0) + calorias
    progreso.proteinas_consumidas = (progreso.proteinas_consumidas or 0) + (
        extraccion.get("proteinas_g", 0) or 0
    )
    progreso.carbohidratos_consumidos = (progreso.carbohidratos_consumidos or 0) + (
        extraccion.get("carbohidratos_g", 0) or 0
    )
    progreso.grasas_consumidas = (progreso.grasas_consumidas or 0) + (extraccion.get("grasas_g", 0) or 0)


def registrar_preferencias_alimentos(extraccion: Dict[str, Any], perfil: Any, db: Session) -> None:
    """Auto-aprendizaje: frecuencia, puntuación y última vez; macros de la fila = último registro."""
    alimentos_raw = extraccion.get("alimentos_detectados", [])
    alimentos = [a.split("[")[0].strip() for a in alimentos_raw if str(a).strip()]
    if not alimentos:
        alimentos = ["comida"]
    extraccion["alimentos_detectados"] = alimentos

    cals = float(extraccion.get("calorias") or 0)
    pr = float(extraccion.get("proteinas_g") or 0)
    cb = float(extraccion.get("carbohidratos_g") or 0)
    gr = float(extraccion.get("grasas_g") or 0)
    n = max(1, len(alimentos))
    sc, sp, s_cb, s_gr = cals / n, pr / n, cb / n, gr / n

    ahora = datetime.now()

    for alimento in alimentos:
        a_low = (alimento or "comida").lower().strip()[:200]
        pref = (
            db.query(PreferenciaAlimento)
            .filter(
                PreferenciaAlimento.client_id == perfil.id,
                sql_func.lower(PreferenciaAlimento.alimento) == a_low,
            )
            .first()
        )

        if pref:
            pref.frecuencia = (pref.frecuencia or 0) + 1
            pref.ultima_vez = ahora
            pref.puntuacion = min(5.0, (pref.puntuacion or 1.0) + 0.1)
            pref.calorias = sc
            pref.proteinas = sp
            pref.carbohidratos = s_cb
            pref.grasas = s_gr
        else:
            db.add(
                PreferenciaAlimento(
                    client_id=perfil.id,
                    alimento=a_low,
                    frecuencia=1,
                    puntuacion=1.0,
                    ultima_vez=ahora,
                    calorias=sc,
                    proteinas=sp,
                    carbohidratos=s_cb,
                    grasas=s_gr,
                )
            )
