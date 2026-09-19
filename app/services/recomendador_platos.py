"""
Motor de recomendaciones confiables de platos.

Estrategia:
  1. BD primero — platos ya validados con macros reales (confianza alta)
  2. KNN como candidatos — ingredientes que el LLM convierte en platos coherentes
  3. Filtro de calidad — validación semántica + nutricional antes de mostrar
  4. Variedad garantizada — rotación por día + horario + historial reciente

Nunca muestra un plato con macros 0 o combinaciones incoherentes.
"""
from __future__ import annotations

import hashlib
import logging
import random
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.orm import Session
import asyncio

from app.core.utils import get_peru_date
from app.models.historial_recomendacion import HistorialRecomendacion
from app.services.nutrition.plate.plate_builder import PlatoBuilder

logger = logging.getLogger(__name__)

_CONDICION_TOKENS: dict[str, set[str]] = {
    "Vegano": {
        "pollo", "pechuga", "muslo", "gallina", "pato", "pavo", "cabrito",
        "cerdo", "chancho", "res", "carne", "bistec", "lomo", "ternera",
        "chicharron", "chicharrón", "jamon", "jamón", "salchicha",
        "pescado", "salmon", "salmón", "atun", "atún", "trucha", "caballa",
        "corvina", "cachema", "lisa", "mero", "tollo", "anchoveta",
        "mariscos", "camaron", "camarón", "langostino", "pulpo", "calamar",
        "leche", "queso", "yogur", "yogurt", "mantequilla",
        "crema de leche", "crema agria", "crema chantilly",
        "manteca", "quesillo", "huevo",
    },
    "Vegetariano": {
        "pollo", "pechuga", "muslo", "gallina", "pato", "pavo", "cabrito",
        "cerdo", "chancho", "res", "carne", "bistec", "lomo", "ternera",
        "chicharron", "chicharrón", "jamon", "jamón", "salchicha",
        "pescado", "salmon", "salmón", "atun", "atún", "trucha", "caballa",
        "corvina", "cachema", "lisa", "mero", "tollo", "anchoveta",
        "mariscos", "camaron", "camarón", "langostino", "pulpo", "calamar",
    },
    "Intolerancia a la Lactosa": {
        "leche", "queso", "yogur", "yogurt", "mantequilla",
        "crema de leche", "crema agria", "crema chantilly",
        "manteca", "quesillo", "lactosa",
    },
    "Celíaco": {
        "trigo", "avena", "cebada", "centeno", "gluten",
        "pan", "pasta", "fideos", "tallarin", "tallarín", "spaghetti",
        "galleta", "harina", "cuscuz", "cuscús",
    },
    "Diabetes": {
        "azucar", "azúcar", "miel", "mermelada", "jarabe",
        "gaseosa", "chicha", "refresco", "jugo azucarado",
        "chocolate", "caramelo", "helado", "torta", "pastel",
        "galleta", "donuts", "churro", "suspiro",
        "picarones", "mazamorra", "alfajor", "alfajores",
        "tres leches", "cocada", "turron", "turrón", "keke", "queque",
    },
}


_DIETA_KEYWORDS_MENSAJE: dict[str, tuple[str, ...]] = {
    "Vegano": ("vegano", "vegana", "veganismo"),
    "Vegetariano": ("vegetariano", "vegetariana"),
    "Intolerancia a la Lactosa": (
        "intolerancia a la lactosa", "intolerante a la lactosa", "soy lactosa",
    ),
    "Celíaco": ("celiaco", "celíaco", "celiaca", "celíaca"),
    "Diabetes": ("diabetes", "diabetico", "diabético", "diabetica", "diabética"),
}


def _detectar_dieta_en_mensaje(mensaje: str) -> list[str]:
    """Detecta restricciones dietéticas dichas en el mensaje actual (ej. "Soy
    vegano"), no solo las guardadas en el perfil. Sin esto, un usuario que
    declara una restricción al vuelo (sin tenerla guardada) podía recibir un
    plato que la viola — el guard solo miraba `medical_conditions`."""
    t = (mensaje or "").lower()
    return [cond for cond, kws in _DIETA_KEYWORDS_MENSAJE.items() if any(k in t for k in kws)]


def _tokens_prohibidos(condiciones: list[str]) -> set[str]:
    """Devuelve el conjunto de tokens prohibidos para una lista de condiciones."""
    tokens: set[str] = set()
    for cond in (condiciones or []):
        tokens.update(_CONDICION_TOKENS.get(cond, set()))
    return tokens


def _plato_es_apto(nombre: str, ingredientes_str: str, tokens: set[str]) -> bool:
    """True si el plato no contiene ningún token prohibido."""
    if not tokens:
        return True
    texto = (nombre + " " + ingredientes_str).lower()
    return not any(t in texto for t in tokens)


_MIN_CONFIANZA = 60

_HISTORIAL_DIAS = 3

_POOL_SIZE = 30

_RANGOS_MOMENTO: dict[str, tuple[float, float]] = {
    "desayuno":   (150.0,  500.0),
    "almuerzo":   (400.0,  950.0),
    "cena":       (120.0,  520.0),
    "snack":      ( 60.0,  300.0),
    "merienda":   ( 60.0,  300.0),
    "cualquiera": (  0.0, 1200.0),
}

_INGREDIENTE_SINONIMOS: dict[str, list[str]] = {
    "mariscos":   ["mariscos", "camaron", "camarón", "langostino", "langosta",
                   "pulpo", "calamar", "almeja", "mejillon", "choro", "cangrejo", "concha"],
    "salmon":     ["salmon", "salmón"],
    "atun":       ["atun", "atún"],
    "trucha":     ["trucha"],
    "caballa":    ["caballa"],
    "corvina":    ["corvina"],
    "cerdo":      ["cerdo", "chancho", "porcino", "chicharron"],
    "res":        ["res", "ternera", "bistec", "lomo fino", "carne de res"],
    "cabrito":    ["cabrito", "cabrilla"],
    "pato":       ["pato", "pato seco"],
    "palta":      ["palta", "aguacate"],
    "platano":    ["platano", "plátano"],
    "lucuma":     ["lucuma", "lúcuma"],
    "frejol":     ["frejol", "frijol", "frejoles", "frijoles"],
    "lenteja":    ["lenteja", "lentejas", "lentejón"],
    "arveja":     ["arveja", "arvejas", "alverjita"],
    "garbanzo":   ["garbanzo", "garbanzos"],
    "haba":       ["haba", "habas"],
    "quinua":     ["quinua", "quinoa"],
    "pasta":      ["pasta", "fideos", "spaghetti", "tallarín", "tallarin", "tallarines"],
    "camote":     ["camote", "boniato"],
    "choclo":     ["choclo", "maiz", "maíz", "elote"],
    "mani":       ["mani", "maní", "mani pelado"],
    "fruto_seco": ["almendra", "nuez", "pecana", "pecanas"],
    "semilla":    ["chia", "chía", "linaza", "ajonjoli"],
    "lacteos":    ["mantequilla", "mantequilla sin sal", "crema de leche"],
    "yogur":      ["yogur", "yogurt", "yoghurt"],
}


def _tiene_ingrediente(nombre: str, ingredientes_str: str, ing_clave: str) -> bool:
    """Devuelve True si el plato contiene el ingrediente (con expansión semántica)."""
    sinonimos = _INGREDIENTE_SINONIMOS.get(ing_clave, [ing_clave])
    texto = (nombre + " " + ingredientes_str).lower()
    return any(s in texto for s in sinonimos)


_KEYWORDS_SOLO_ALMUERZO = frozenset({
    "arroz con pato", "arroz con cabrito", "arroz con pollo",
    "lomo saltado", "seco de res", "seco de cabrito", "seco de pollo",
    "aji de gallina", "ají de gallina",
    "pollo a la brasa", "chicharron de cerdo", "chicharrón",
    "tallarin saltado", "tallarín saltado", "sopa seca",
    "carapulcra", "pepian", "pepián",
    "jalea", "sudado de pescado", "caldo de gallina",
    "cebiche", "ceviche", "tiradito",
    "causa ferreñafana", "causa rellena",
})

_KEYWORDS_LIGEROS = frozenset({
    "sopa", "crema de", "caldo", "ensalada", "tostada", "batido",
    "fruta", "yogurt", "avena", "granola",
})


def _es_plato_apto_para_momento(nombre: str, kcal: float, momento: str) -> bool:
    """
    Devuelve True si el plato es apropiado para el momento del día dado.
    Combina rangos calóricos + keywords de nombre.
    """
    nombre_n = nombre.lower().strip()
    momento_n = (momento or "cualquiera").lower()

    es_solo_almuerzo = any(kw in nombre_n for kw in _KEYWORDS_SOLO_ALMUERZO)
    if es_solo_almuerzo and momento_n not in ("almuerzo", "cualquiera"):
        return False

    es_ligero = any(kw in nombre_n for kw in _KEYWORDS_LIGEROS)
    if es_ligero and momento_n == "almuerzo" and kcal < 300:
        return False

    kcal_min, kcal_max = _RANGOS_MOMENTO.get(momento_n, (0.0, 1200.0))
    if kcal > kcal_max * 1.15:
        return False
    if kcal < kcal_min * 0.5:
        return False

    return True


class RecomendadorPlatosConfiables:
    """
    Recomienda platos con valores nutricionales verificados.

    Fuentes en orden de prioridad:
    1. Platos en BD con macros calculadas desde ingredientes reales
    2. KNN (alimentos) → ensamblados en platos coherentes por LLM
    """

    def __init__(self, db: Session, plate_builder: Optional[PlatoBuilder] = None):
        self.db = db
        self.plate_builder = plate_builder


    def recomendar(
        self,
        client_id: int,
        deficit_kcal: float,
        deficit_proteina: float,
        deficit_carb: float,
        deficit_grasas: float,
        momento_dia: str = "cualquiera",
        n: int = 3,
        excluir_nombres: Optional[List[str]] = None,
        ingrediente_clave: Optional[str] = None,
        condiciones_dieta: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retorna N platos recomendados, confiables y variados.

        Args:
            client_id: ID del cliente
            deficit_kcal: calorías que aún faltan en el día
            deficit_proteina: proteínas faltantes (g)
            deficit_carb: carbohidratos faltantes (g)
            deficit_grasas: grasas faltantes (g)
            momento_dia: desayuno|almuerzo|cena|snack|cualquiera
            n: número de platos a retornar
            excluir_nombres: platos a excluir (forbidden + historial)

        Returns:
            Lista de dicts con: nombre, macros, fuente, confianza
        """
        excluir = set(s.lower().strip() for s in (excluir_nombres or []))

        historial = self._historial_reciente(client_id, dias=_HISTORIAL_DIAS)
        excluir.update(historial)

        tokens_prohibidos = _tokens_prohibidos(condiciones_dieta or [])

        _pool_efectivo = _POOL_SIZE * 3 if tokens_prohibidos else _POOL_SIZE

        candidatos_bd = self._candidatos_desde_bd(
            deficit_kcal=deficit_kcal,
            deficit_proteina=deficit_proteina,
            deficit_carb=deficit_carb,
            deficit_grasas=deficit_grasas,
            excluir=excluir,
            momento_dia=momento_dia,
            pool=_pool_efectivo,
            ingrediente_clave=ingrediente_clave,
            tokens_prohibidos=tokens_prohibidos,
        )

        seed = self._seed_del_dia(client_id, deficit_kcal, momento_dia)
        rng = random.Random(seed)

        alta_similitud = [c for c in candidatos_bd if c["score"] >= 80]
        media_similitud = [c for c in candidatos_bd if 60 <= c["score"] < 80]
        baja_similitud = [c for c in candidatos_bd if c["score"] < 60]

        rng.shuffle(alta_similitud)
        rng.shuffle(media_similitud)
        rng.shuffle(baja_similitud)

        mezclados = alta_similitud + media_similitud + baja_similitud

        seleccionados = self._seleccionar_con_diversidad(mezclados, n)

        if len(seleccionados) < n and self.plate_builder:
            faltantes = n - len(seleccionados)
            logger.info(f"BD pobre para este déficit. Generando {faltantes} platos nuevos vía IA...")
            nuevos_platos = self._generar_y_validar_nuevos_platos(
                client_id=client_id,
                deficit_kcal=deficit_kcal,
                deficit_proteina=deficit_proteina,
                deficit_carb=deficit_carb,
                deficit_grasas=deficit_grasas,
                momento_dia=momento_dia,
                n_faltantes=faltantes,
                excluir=excluir,
                ingrediente_clave=ingrediente_clave,
            )
            seleccionados.extend(nuevos_platos)

        for plato in seleccionados:
            self._guardar_recomendacion(client_id, plato)

        logger.info(
            f"Recomendados {len(seleccionados)} platos para cliente {client_id} "
            f"({momento_dia}) — pool={len(candidatos_bd)} candidatos"
        )

        return seleccionados


    def _candidatos_desde_bd(
        self,
        deficit_kcal: float,
        deficit_proteina: float,
        deficit_carb: float,
        deficit_grasas: float,
        excluir: set,
        momento_dia: str,
        pool: int = 30,
        ingrediente_clave: Optional[str] = None,
        tokens_prohibidos: Optional[set] = None,
    ) -> List[Dict[str, Any]]:
        """
        Busca platos en BD con macros reales calculadas desde sus ingredientes.

        Calcula score de similitud al déficit del usuario.
        """
        try:
            rows = self.db.execute(text("""
                SELECT
                    p.id,
                    p.nombre,
                    p.tipo_plato,
                    SUM(a.calorias_100g       * pi.gramos / 100.0) AS kcal,
                    SUM(a.proteina_100g       * pi.gramos / 100.0) AS prot,
                    SUM(a.carbohidratos_100g  * pi.gramos / 100.0) AS carb,
                    SUM(a.grasas_100g         * pi.gramos / 100.0) AS gras,
                    COUNT(pi.id)                                    AS n_ings,
                    COUNT(CASE WHEN a.calorias_100g > 0 THEN 1 END) AS ings_ok,
                    string_agg(
                        pi.gramos::integer::text || 'g ' || a.nombre || ' (' || round((a.calorias_100g * pi.gramos / 100.0)::numeric, 1)::text || ' kcal)',
                        ', '
                    ) AS ingredientes_str
                FROM platos p
                JOIN plato_ingredientes pi ON p.id = pi.plato_id
                JOIN alimentos a ON pi.alimento_id = a.id
                GROUP BY p.id, p.nombre, p.tipo_plato
                HAVING
                    COUNT(pi.id) >= 2
                    AND COUNT(CASE WHEN a.calorias_100g > 0 THEN 1 END) = COUNT(pi.id)
                    AND SUM(a.calorias_100g * pi.gramos / 100.0) > 50
                LIMIT :pool
            """), {"pool": pool * 3}).fetchall()

        except Exception as exc:
            logger.error(f"Error consultando platos BD: {exc}")
            return []

        candidatos = []
        ing_clave_norm = ingrediente_clave.lower().strip() if ingrediente_clave else None

        for row in rows:
            nombre = row[1]
            if nombre.lower().strip() in excluir:
                continue

            ingredientes_str = (row[9] or "").lower()
            if ing_clave_norm:
                if not _tiene_ingrediente(nombre, ingredientes_str, ing_clave_norm):
                    continue

            if tokens_prohibidos and not _plato_es_apto(nombre, ingredientes_str, tokens_prohibidos):
                logger.debug("[Dieta] Plato '%s' descartado por restricción dietética", nombre)
                continue

            kcal = float(row[3] or 0)
            prot = float(row[4] or 0)
            carb = float(row[5] or 0)
            gras = float(row[6] or 0)

            if momento_dia != "cualquiera" and not _es_plato_apto_para_momento(
                nombre, kcal, momento_dia
            ):
                logger.debug(
                    "[Momento] Plato '%s' (%.0f kcal) descartado para '%s'",
                    nombre, kcal, momento_dia,
                )
                continue

            score = self._calcular_score(
                kcal=kcal, prot=prot, carb=carb, gras=gras,
                d_kcal=deficit_kcal, d_prot=deficit_proteina,
                d_carb=deficit_carb, d_gras=deficit_grasas,
                momento_dia=momento_dia,
            )

            if score < _MIN_CONFIANZA:
                continue

            candidatos.append({
                "plato_id": row[0],
                "nombre": nombre,
                "tipo_plato": row[2] or "cualquiera",
                "macros": {
                    "calorias": round(kcal, 1),
                    "proteinas_g": round(prot, 1),
                    "carbohidratos_g": round(carb, 1),
                    "grasas_g": round(gras, 1),
                },
                "n_ingredientes": int(row[7]),
                "ingredientes_str": row[9] or "",
                "fuente": "BD_Verificado",
                "confianza": 95,
                "score": score,
            })

        candidatos.sort(key=lambda x: x["score"], reverse=True)
        return candidatos[:pool]

    def _calcular_score(
        self,
        kcal: float, prot: float, carb: float, gras: float,
        d_kcal: float, d_prot: float,
        d_carb: float, d_gras: float,
        momento_dia: str = "cualquiera",
    ) -> float:
        """
        Score 0–100 que mide cuánto se ajusta el plato al déficit.

        Prioriza:
        - Que las calorías estén en rango del 30%-100% del déficit kcal
        - Que aporte proteína significativa cuando hay déficit proteico
        - Que no exceda exageradamente el déficit calórico
        """
        if d_kcal <= 0:
            d_kcal = 400

        score = 100.0

        ratio_kcal = kcal / d_kcal
        if ratio_kcal < 0.15:
            score -= 40
        elif ratio_kcal < 0.25:
            score -= 20
        elif ratio_kcal <= 1.1:
            score += 5
        elif ratio_kcal <= 1.5:
            score -= 10
        else:
            score -= 30

        if d_prot > 5:
            ratio_prot = prot / d_prot
            if ratio_prot >= 0.8:
                score += 25
            elif ratio_prot >= 0.5:
                score += 15
            elif ratio_prot < 0.2:
                score -= 40
            elif ratio_prot < 0.4:
                score -= 20
            
        if d_carb > 20:
            ratio_carb = carb / d_carb
            if ratio_carb >= 0.8:
                score += 20
            elif ratio_carb >= 0.5:
                score += 10
            elif ratio_carb < 0.2:
                score -= 40
                
        if d_gras > 10:
            ratio_gras = gras / d_gras
            if ratio_gras >= 0.8:
                score += 25
            elif ratio_gras >= 0.5:
                score += 15
            elif ratio_gras < 0.2:
                score -= 40

        if gras > kcal * 0.6 and d_gras < 20:
            score -= 40
        if carb < 2 and prot < 5 and d_carb < 20:
            score -= 40

        if momento_dia and momento_dia != "cualquiera":
            _, kcal_max_momento = _RANGOS_MOMENTO.get(momento_dia.lower(), (0.0, 1200.0))
            if kcal > kcal_max_momento:
                exceso_pct = (kcal - kcal_max_momento) / kcal_max_momento
                penalizacion = min(50.0, exceso_pct * 120)
                score -= penalizacion
                logger.debug(
                    "[Score] '%s' penalizado %.1f pts por exceso calórico para %s (%.0f > %.0f kcal)",
                    "plato", penalizacion, momento_dia, kcal, kcal_max_momento,
                )

        return max(0.0, min(100.0, score))


    def _seleccionar_con_diversidad(
        self,
        candidatos: List[Dict],
        n: int,
    ) -> List[Dict]:
        """
        Selecciona N platos garantizando variedad de tipo.

        No muestra 3 platos idénticos de pollo seguidos.
        """
        seleccionados = []
        nombres_vistos = set()

        _RE_CONECTOR = re.compile(r"\s+(?:con|y|a\s+la?|al|en\s+|de\s+)", re.I)

        def categoria(nombre: str) -> str:
            partes = _RE_CONECTOR.split(nombre.lower(), maxsplit=1)
            primaria = partes[0].strip()
            nombre_full = nombre.lower()

            def _match(kws: list, texto: str) -> bool:
                return any(kw in texto for kw in kws)

            for texto in (primaria, nombre_full):
                if _match(["pollo", "pechuga", "gallina"], texto): return "pollo"
                if _match(["pescado", "caballa", "corvina", "trucha", "lisa",
                           "mero", "tollo", "cachema", "cebiche", "tiradito",
                           "anchoveta", "sudado"], texto): return "pescado"
                if _match(["lomo", "bistec", "ternera"], texto): return "res"
                if _match([" res ", "carne de res"], " " + texto + " "): return "res"
                if _match(["cerdo", "chancho", "chicharron"], texto): return "cerdo"
                if _match(["pato", "pavo", "cabrito"], texto): return "ave"
                if "huevo" in texto: return "huevo"
                if _match(["sopa", "caldo", "crema de"], texto): return "sopa"
                if _match(["lenteja", "garbanzo", "frejol", "frijol", "haba",
                           "arveja", "pallare", "tofu", "soya"], texto): return "legumbre"
                if _match(["quinua", "quinoa"], texto): return "quinua"
                if "arroz" in texto: return "arroz"
                if _match(["papa", "camote", "yuca", "causa"], texto): return "tuberculo"
                if _match(["ensalada", "verdura", "vegetal"], texto): return "vegetal"
            return "otro"

        categorias_usadas: Dict[str, int] = {}
        max_por_categoria = max(1, n // 2)

        for candidato in candidatos:
            if len(seleccionados) >= n:
                break

            nombre_lower = candidato["nombre"].lower().strip()
            if nombre_lower in nombres_vistos:
                continue

            cat = categoria(candidato["nombre"])
            if categorias_usadas.get(cat, 0) >= max_por_categoria:
                continue

            nombres_vistos.add(nombre_lower)
            categorias_usadas[cat] = categorias_usadas.get(cat, 0) + 1
            seleccionados.append(candidato)

        if len(seleccionados) < n:
            for candidato in candidatos:
                if len(seleccionados) >= n:
                    break
                nombre_lower = candidato["nombre"].lower().strip()
                if nombre_lower not in nombres_vistos:
                    nombres_vistos.add(nombre_lower)
                    seleccionados.append(candidato)

        return seleccionados


    def _generar_y_validar_nuevos_platos(
        self,
        client_id: int,
        deficit_kcal: float,
        deficit_proteina: float,
        deficit_carb: float,
        deficit_grasas: float,
        momento_dia: str,
        n_faltantes: int,
        excluir: set,
        ingrediente_clave: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Pide al LLM que invente nuevos platos para el déficit exacto,
        los valida con PlateBuilder y si son válidos, los devuelve y quedan
        persistidos en la BD para el futuro.
        """
        try:
            from app.services.ai.llm_service import LLMService
            llm = LLMService()

            extra_ingrediente = ""
            if ingrediente_clave:
                extra_ingrediente = f"\nOBLIGATORIO: Todas las recetas DEBEN contener el ingrediente '{ingrediente_clave}' (como ingrediente principal o base)."

            prompt = (
                f"Actúa como nutricionista de un gimnasio en Lambayeque, Perú. "
                f"Crea {n_faltantes + 2} platos distintos típicos de la cocina peruana norteña para '{momento_dia}'.\n"
                f"Deben tener aproximadamente: {deficit_kcal:.0f} kcal, {deficit_proteina:.0f}g proteína, "
                f"{deficit_carb:.0f}g carbohidratos, {deficit_grasas:.0f}g grasas.{extra_ingrediente}\n"
                "REGLAS OBLIGATORIAS:\n"
                "1. Nombres en español, usando preparaciones peruanas conocidas: "
                "'a la plancha', 'al vapor', 'estofado', 'guisado', 'sudado', 'sancochado', "
                "'salteado', 'a la parrilla', 'con arroz', 'ensalada de', 'sopa de', etc.\n"
                "2. PROHIBIDO usar nombres de platos extranjeros: fideuá, ratatouille, stir-fry, "
                "curry, risotto, pad thai, wok, etc.\n"
                "3. Ingredientes comunes en mercados peruanos (no ingredientes importados raros).\n"
                "4. USA SOLO GRAMOS EXACTOS, sin 'tazas', 'cucharadas' ni unidades.\n\n"
                "Responde ÚNICAMENTE con un arreglo JSON válido:\n"
                "[\n"
                "  {\n"
                "    \"nombre_plato\": \"Pollo a la Plancha con Arroz y Ensalada\",\n"
                "    \"ingredientes\": [\n"
                "      {\"nombre\": \"pechuga de pollo\", \"gramos\": 150},\n"
                "      {\"nombre\": \"arroz blanco\", \"gramos\": 100},\n"
                "      {\"nombre\": \"lechuga\", \"gramos\": 50}\n"
                "    ]\n"
                "  }\n"
                "]"
            )

            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, llm.generar_json(prompt=prompt, max_tokens=1500))
                    propuestas = future.result(timeout=25)
            else:
                propuestas = loop.run_until_complete(llm.generar_json(prompt=prompt, max_tokens=1500))

            if not propuestas or not isinstance(propuestas, list):
                return []
            
            nuevos = []

            for prop in propuestas:
                nombre = prop.get("nombre_plato", "")
                if not nombre or nombre.lower().strip() in excluir:
                    continue

                ings = prop.get("ingredientes", [])
                
                if ingrediente_clave:
                    ing_clave_norm = ingrediente_clave.lower().strip()
                    ings_str_llm = " ".join(i.get("nombre", "") for i in ings)
                    if not _tiene_ingrediente(nombre, ings_str_llm, ing_clave_norm):
                        logger.warning(f"Plato LLM '{nombre}' descartado por no contener '{ingrediente_clave}'.")
                        continue
                
                resultado = self.plate_builder.construir_plato(
                    nombre_plato=nombre,
                    ingredientes=ings,
                    client_id=client_id,
                    tipo_plato=momento_dia,
                )

                if resultado.exito and resultado.confianza_global >= _MIN_CONFIANZA:
                    if isinstance(resultado.macros_totales, dict):
                        kcal = resultado.macros_totales.get('calorias', 0)
                        prot = resultado.macros_totales.get('proteina', 0)
                        carb = resultado.macros_totales.get('carbohidratos', 0)
                        gras = resultado.macros_totales.get('grasas', 0)
                    else:
                        kcal = resultado.macros_totales.calorias
                        prot = resultado.macros_totales.proteina
                        carb = resultado.macros_totales.carbohidratos
                        gras = resultado.macros_totales.grasas

                    score = self._calcular_score(
                        kcal=kcal, prot=prot, carb=carb, gras=gras,
                        d_kcal=deficit_kcal, d_prot=deficit_proteina,
                        d_carb=deficit_carb, d_gras=deficit_grasas,
                    )

                    if score < 40:
                        logger.warning(f"Plato LLM '{resultado.nombre}' descartado por bajo score ({score}) respecto al objetivo.")
                        continue

                    ing_str_list = []
                    for i in resultado.ingredientes:
                        _kcal_i = i.macros_totales.get('calorias', 0) if i.macros_totales else 0
                        ing_str_list.append(f"{i.gramos}g {i.nombre} ({round(_kcal_i, 1)} kcal)")

                    nuevos.append({
                        "plato_id": resultado.plato_id,
                        "nombre": resultado.nombre,
                        "tipo_plato": momento_dia,
                        "macros": {
                            "calorias": round(kcal, 1),
                            "proteinas_g": round(prot, 1),
                            "carbohidratos_g": round(carb, 1),
                            "grasas_g": round(gras, 1),
                        },
                        "n_ingredientes": len(resultado.ingredientes),
                        "ingredientes_str": ", ".join(ing_str_list),
                        "fuente": "IA_Generado_y_Validado",
                        "confianza": resultado.confianza_global,
                        "score": score,
                    })

                    if len(nuevos) >= n_faltantes:
                        break

            logger.info(f"Se lograron crear y validar {len(nuevos)} platos nuevos.")
            return nuevos

        except Exception as e:
            logger.warning(f"Error generando platos nuevos: {e}")
            return []


    def _historial_reciente(self, client_id: int, dias: int = 3) -> set:
        """Retorna nombres de platos recomendados en los últimos N días."""
        try:
            desde = datetime.now() - timedelta(days=dias)
            rows = (
                self.db.query(HistorialRecomendacion.nombre_plato)
                .filter(
                    HistorialRecomendacion.client_id == client_id,
                    HistorialRecomendacion.created_at >= desde,
                )
                .all()
            )
            return {r[0].lower().strip() for r in rows if r[0]}
        except Exception as exc:
            logger.warning(f"Error leyendo historial: {exc}")
            return set()

    def _guardar_recomendacion(self, client_id: int, plato: Dict) -> None:
        """Persiste la recomendación en historial para evitar repeticiones."""
        try:
            macros = plato.get("macros", {})
            self.db.add(HistorialRecomendacion(
                client_id=client_id,
                plato_id=plato.get("plato_id"),
                nombre_plato=plato["nombre"][:200],
                calorias=macros.get("calorias", 0),
                proteinas_g=macros.get("proteinas_g", 0),
                carbohidratos_g=macros.get("carbohidratos_g", 0),
                grasas_g=macros.get("grasas_g", 0),
                momento_dia=plato.get("tipo_plato", "cualquiera"),
                fue_consumido=False,
            ))
            self.db.commit()
        except Exception as exc:
            logger.warning(f"Error guardando historial: {exc}")
            self.db.rollback()

    def _seed_del_dia(self, client_id: int, deficit_kcal: float, momento: str) -> int:
        """Seed reproducible por día+usuario+momento para variedad controlada."""
        hoy = get_peru_date().toordinal()
        raw = f"{hoy}:{client_id}:{int(deficit_kcal)}:{momento}"
        return int(hashlib.md5(raw.encode()).hexdigest()[:8], 16)
