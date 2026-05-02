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
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.core.utils import get_peru_date
from app.models.alimento import Alimento
from app.models.alimento_unidad import AlimentoUnidad  # noqa: F401 (used in _get_porcion_estandar)
from app.models.historial import ProgresoCalorias
from app.models.preferencias import PreferenciaAlimento
from app.services.alimentos_db_service import AlimentosDBService, _norm as _norm_al
from app.services.asistente_nutricion import (
    advertencia_alimentos_prohibidos,
    aplicar_extraccion_nlp_comida_a_progreso,
    registrar_preferencias_alimentos,
    verificar_conflicto_macros,
)


# ── Helpers de normalización (usados también en asistente_service) ───────────

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
    s = (s or "").strip().lower()
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


def _inferir_momento_dia(mensaje: str) -> Optional[str]:
    """Infiere el momento del día a partir del texto del mensaje."""
    m = (mensaje or "").lower()
    if any(w in m for w in ["desayun", "mañana", "breakfast"]):
        return "desayuno"
    if any(w in m for w in ["almorz", "almuerz", "almorzar", "mediodía", "mediodia"]):
        return "almuerzo"
    if any(w in m for w in ["cenar", "cené", "cene", "noche"]):
        return "cena"
    if any(w in m for w in ["merienda", "snack", "colación", "tarde"]):
        return "merienda"
    return None


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


# Regex: detecta prefijos de gramaje o recipiente que NO deben activar Capa 1.5
# Ejemplos bloqueados: "220g de pollo", "50kg arroz", "un vaso de limonada"
_CAPA15_SKIP_RE = re.compile(
    r'^\d+(?:[.,]\d+)?\s*(?:g|gr|kg|ml|l|litros?|vasos?|tazas?|copas?)\b'
    r'|^(?:un|una)\s+(?:vaso|taza|copa|botella|lata|plato)\b',
    re.IGNORECASE,
)


def _es_candidato_plato_capa15(item: str) -> bool:
    """True si el item parece un plato complejo (no una cantidad ni recipiente simple)."""
    return len(item.split()) >= 2 and not _CAPA15_SKIP_RE.match(item)


def _split_items_from_message(msg: str) -> List[str]:
    t = (msg or "").lower()
    t = re.sub(r"(?i)^\s*(he\s+comido|com[ií]\s*|me\s+com[ií]\s*|hoy\s+com[ií]\s*)\s*", "", t).strip()
    t = re.sub(r"(?i)\b(otra\s+vez|de\s+nuevo|nuevamente)\b", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return [p.strip() for p in re.split(r"\s+y\s+|,|;", t) if p.strip()]


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

        # CAPA 0: NLPFoodExtractor
        if not _parece_ejercicio:
            capa0_result = await self._capa0_nlp(mensaje, msg_lower, _parece_comida, ia_engine, db)
            if capa0_result.get("_final"):
                # guard de negación o alimento desconocido — responder directamente
                return {k: v for k, v in capa0_result.items() if k != "_final"}
            if capa0_result.get("calorias", 0) > 0:
                pre_extraccion = capa0_result

        # Porción de lata (capa especial antes del catálogo)
        if not pre_extraccion and _msg_tiene_porcion_lata(mensaje):
            pre_extraccion = self._capa_lata(mensaje, db)

        # CAPA 1: catálogo platos
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
        # Intentar extraer nombre de alimento del propio mensaje
        m = re.search(
            r"(?i)(?:de\s+|del?\s+|cuánto\s+(?:de\s+)?)?([a-záéíóúüñ][\w\s]{2,25}?)(?:\s*[,.]|$)",
            mensaje,
        )
        nombre_alim = m.group(1).strip() if m else "alimento"
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

        hoy     = get_peru_date()
        progreso = db.query(ProgresoCalorias).filter(
            ProgresoCalorias.client_id == perfil.id, ProgresoCalorias.fecha == hoy
        ).first()
        if not progreso:
            progreso = ProgresoCalorias(client_id=perfil.id, fecha=hoy)
            db.add(progreso)

        extraccion = {
            "es_comida": True, "es_ejercicio": False,
            "calorias": round(kcal, 1), "proteinas_g": round(p, 1),
            "carbohidratos_g": round(c, 1), "grasas_g": round(g, 1),
            "fibra_g": 0, "azucar_g": 0, "sodio_mg": 0,
            "alimentos_detectados": [nombre], "ejercicios_detectados": [],
            "calidad_nutricional": "Alta", "porcion_g": porcion_g, "origen": "manual",
        }
        aplicar_extraccion_nlp_comida_a_progreso(extraccion, progreso)
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
            from app.services.nlp_food_extractor import NLPFoodExtractor
            extractor = NLPFoodExtractor(ia_engine, db)

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
            print(f"[RegistroComida] Error CAPA 0: {e}")
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

        matched: List[tuple] = []
        for it in items:
            m = re.match(
                r"^\s*((?:\d+(?:[.,]\d+)?)|uno|una|un|dos|tres|cuatro|cinco|medio|media)\s+(.*)$",
                it, re.IGNORECASE,
            )
            qty, name_part = 1.0, it
            if m:
                qty, name_part = _parse_qty(m.group(1)), m.group(2).strip()

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

        if not matched:
            # ── Capa 1.5: construcción dinámica de platos ──────────────────────
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
                    print(f"[Capa1.5] Error construyendo plato dinámico: {e}")

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
                    except Exception as _e15:
                        print(f"[Capa1.5] Error construyendo '{_cand}': {_e15}")

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
            kcal += float(row[2] or 0) * qty
            p_g  += float(row[3] or 0) * qty
            c_g  += float(row[4] or 0) * qty
            g_g  += float(row[5] or 0) * qty
            nombres.append(row[1])

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

        return {
            "es_comida": True, "es_ejercicio": False,
            "calorias": round(kcal, 1), "proteinas_g": round(p_g, 1),
            "carbohidratos_g": round(c_g, 1), "grasas_g": round(g_g, 1),
            "fibra_g": 0, "azucar_g": 0, "sodio_mg": 0,
            "alimentos_detectados": nombres, "ejercicios_detectados": [],
            "calidad_nutricional": "Alta", "origen": "platos",
            "desglose_ingredientes": desglose,
            "desglose_total": desglose_total,
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

        if not extraccion or not extraccion.get("calorias"):
            return {
                "success": False, "tipo_detectado": "ninguno",
                "alimentos": [], "datos": {},
                "mensaje": "No pude identificar el alimento. ¿Puedes ser más específico? 🍽️",
            }

        hoy      = get_peru_date()
        progreso = db.query(ProgresoCalorias).filter(
            ProgresoCalorias.client_id == perfil.id, ProgresoCalorias.fecha == hoy
        ).first()
        if not progreso:
            progreso = ProgresoCalorias(client_id=perfil.id, fecha=hoy)
            db.add(progreso)

        adv_prohibido = advertencia_alimentos_prohibidos(extraccion, perfil)
        aplicar_extraccion_nlp_comida_a_progreso(extraccion, progreso)
        registrar_preferencias_alimentos(extraccion, perfil, db)
        db.commit()

        nombres     = extraccion.get("alimentos_detectados", [])
        nombre_str  = ", ".join(nombres) if nombres else "tu registro"
        alerta_macros = verificar_conflicto_macros(progreso, plan_hoy_data, perfil)

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
        warn = extraccion.get("_warn_cantidad") or extraccion.get("advertencia")
        if warn and ("kcal" in str(warn) or "correcto" in str(warn).lower()):
            msg_final += f"\n\n⚠️ {warn}"

        return {
            "success": True, "tipo_detectado": "comida",
            "alimentos": nombres,
            "advertencia_prohibido": adv_prohibido,
            "alerta_macros": alerta_macros,
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
        """CAPA 5 (último recurso): Llama-3 estima macros para ingredientes simples.
        Solo si las 4 capas anteriores fallaron. Persiste con fuente=llm para trazabilidad."""
        try:
            import json
            prompt = (
                f"El usuario dijo: '{mensaje}'. Identifica el ALIMENTO SIMPLE (no platos complejos). "
                "Devuelve SOLO JSON: {\"nombre\":\"...\",\"calorias\":0,\"proteinas_g\":0,"
                "\"carbohidratos_g\":0,\"grasas_g\":0,\"porcion_g\":100}"
            )
            resp = await ia_engine.consultar_groq(prompt, sistema="Eres nutricionista. Solo JSON, sin texto extra.")
            data = json.loads(resp)
            if data.get("calorias", 0) <= 0:
                return None

            nombre   = data.get("nombre", "alimento desconocido")
            porcion  = float(data.get("porcion_g") or 100)
            f        = 100.0 / max(1.0, porcion)
            nn       = _norm_al(nombre)

            existing = db.query(Alimento).filter(Alimento.nombre_normalizado == nn).first()
            if not existing:
                db.add(Alimento(
                    nombre=nombre[:255], nombre_normalizado=nn[:255],
                    calorias_100g=round(data["calorias"] * f, 2),
                    proteina_100g=round(data.get("proteinas_g", 0) * f, 2),
                    carbohidratos_100g=round(data.get("carbohidratos_g", 0) * f, 2),
                    grasas_100g=round(data.get("grasas_g", 0) * f, 2),
                    fuente="llm",
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
            print(f"[RegistroComida] CAPA 5 LLM error: {e}")
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
