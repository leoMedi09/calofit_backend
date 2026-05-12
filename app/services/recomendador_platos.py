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
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import func, text
from sqlalchemy.orm import Session
import asyncio
import json

from app.core.utils import get_peru_date
from app.models import Plato, PlatoIngrediente, Alimento
from app.models.historial_recomendacion import HistorialRecomendacion
from app.services.nutrition.plate.plate_builder import PlatoBuilder

logger = logging.getLogger(__name__)

# Confianza mínima para mostrar un plato al usuario
_MIN_CONFIANZA = 60

# Ventana de exclusión por historial (días)
_HISTORIAL_DIAS = 3

# Número de candidatos a generar antes de filtrar
_POOL_SIZE = 30

# ─── Rangos calóricos por momento del día ────────────────────────────────────
# (kcal_min, kcal_max) — platos fuera de rango son penalizados o descartados
_RANGOS_MOMENTO: dict[str, tuple[float, float]] = {
    "desayuno":   (150.0,  500.0),
    "almuerzo":   (400.0,  950.0),
    "cena":       (120.0,  520.0),
    "snack":      ( 60.0,  300.0),
    "merienda":   ( 60.0,  300.0),
    "cualquiera": (  0.0, 1200.0),
}

# ─── Platos típicamente pesados → solo almuerzo ───────────────────────────────
# Si el nombre normalizado del plato contiene alguna de estas palabras,
# se excluye automáticamente de cena, desayuno y snack.
_KEYWORDS_SOLO_ALMUERZO = frozenset({
    "arroz con pato", "arroz con cabrito", "lomo saltado", "seco de res",
    "seco de cabrito", "jalea", "pollo a la brasa", "chicharron de cerdo",
    "aji de gallina", "causa ferreñafana", "tallarin saltado", "sopa seca",
    "sudado de pescado", "caldo de gallina", "arroz con pollo",
})

# ─── Platos ligeros → válidos para cena/desayuno/snack pero NO almuerzo ──────
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

    # 1) Platos muy pesados → solo almuerzo
    es_solo_almuerzo = any(kw in nombre_n for kw in _KEYWORDS_SOLO_ALMUERZO)
    if es_solo_almuerzo and momento_n not in ("almuerzo", "cualquiera"):
        return False

    # 2) Platos ligeros → no los recomendamos como almuerzo principal
    es_ligero = any(kw in nombre_n for kw in _KEYWORDS_LIGEROS)
    if es_ligero and momento_n == "almuerzo" and kcal < 300:
        return False  # una sopita sola no cubre el almuerzo

    # 3) Rango calórico duro por momento
    kcal_min, kcal_max = _RANGOS_MOMENTO.get(momento_n, (0.0, 1200.0))
    if kcal > kcal_max * 1.15:   # tolerancia 15% hacia arriba
        return False
    if kcal < kcal_min * 0.5:    # muy por debajo del mínimo
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

    # ─────────────────────────────────────────────────────────────────────────
    # API PÚBLICA
    # ─────────────────────────────────────────────────────────────────────────

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

        # 1. Obtener historial reciente del cliente
        historial = self._historial_reciente(client_id, dias=_HISTORIAL_DIAS)
        excluir.update(historial)

        # 2. Candidatos desde BD (platos con macros reales)
        candidatos_bd = self._candidatos_desde_bd(
            deficit_kcal=deficit_kcal,
            deficit_proteina=deficit_proteina,
            deficit_carb=deficit_carb,
            deficit_grasas=deficit_grasas,
            excluir=excluir,
            momento_dia=momento_dia,
            pool=_POOL_SIZE,
            ingrediente_clave=ingrediente_clave,
        )

        # 3. Mezclar con diversidad (shuffl estable por día)
        seed = self._seed_del_dia(client_id, deficit_kcal, momento_dia)
        rng = random.Random(seed)

        # Separar por similitud alta vs media para garantizar variedad
        alta_similitud = [c for c in candidatos_bd if c["score"] >= 80]
        media_similitud = [c for c in candidatos_bd if 60 <= c["score"] < 80]
        baja_similitud = [c for c in candidatos_bd if c["score"] < 60]

        rng.shuffle(alta_similitud)
        rng.shuffle(media_similitud)
        rng.shuffle(baja_similitud)

        # Pool mezclado: 60% alta + 30% media + 10% baja
        mezclados = alta_similitud + media_similitud + baja_similitud

        # 4. Seleccionar N con diversidad de tipo
        seleccionados = self._seleccionar_con_diversidad(mezclados, n, rng)

        # 5. FALLBACK: Si no hay suficientes platos, usar LLM para crear nuevos
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

        # 6. Guardar en historial para evitar repetición
        for plato in seleccionados:
            self._guardar_recomendacion(client_id, plato)

        logger.info(
            f"Recomendados {len(seleccionados)} platos para cliente {client_id} "
            f"({momento_dia}) — pool={len(candidatos_bd)} candidatos"
        )

        return seleccionados

    # ─────────────────────────────────────────────────────────────────────────
    # CANDIDATOS DESDE BD
    # ─────────────────────────────────────────────────────────────────────────

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
    ) -> List[Dict[str, Any]]:
        """
        Busca platos en BD con macros reales calculadas desde sus ingredientes.

        Calcula score de similitud al déficit del usuario.
        """
        try:
            # Query: platos con todos sus ingredientes teniendo macros válidas
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
                if ing_clave_norm not in nombre.lower() and ing_clave_norm not in ingredientes_str:
                    continue

            kcal = float(row[3] or 0)
            prot = float(row[4] or 0)
            carb = float(row[5] or 0)
            gras = float(row[6] or 0)

            # ── NUEVO: Filtro por momento del día ─────────────────────────────
            # Descarta platos que no son aptos para el horario pedido.
            # Ej: "Arroz con Pollo" (524 kcal) no aparece en cena (máx 520 kcal).
            if momento_dia != "cualquiera" and not _es_plato_apto_para_momento(
                nombre, kcal, momento_dia
            ):
                logger.debug(
                    "[Momento] Plato '%s' (%.0f kcal) descartado para '%s'",
                    nombre, kcal, momento_dia,
                )
                continue

            # Score: qué tan bien cubre el déficit (similitud coseno simplificada)
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

        # Ordenar por score desc
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
            d_kcal = 400  # default razonable

        score = 100.0

        # Penalizar si el plato tiene demasiadas o pocas calorías
        ratio_kcal = kcal / d_kcal
        if ratio_kcal < 0.15:
            score -= 40    # muy pequeño
        elif ratio_kcal < 0.25:
            score -= 20
        elif ratio_kcal <= 1.1:
            score += 5     # buen rango
        elif ratio_kcal <= 1.5:
            score -= 10
        else:
            score -= 30    # excede mucho el déficit

        # Puntuación agresiva de Proteína
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
            
        # Puntuación agresiva de Carbohidratos (si el usuario pidió alto en carbos, d_carb se eleva)
        if d_carb > 20:
            ratio_carb = carb / d_carb
            if ratio_carb >= 0.8:
                score += 20
            elif ratio_carb >= 0.5:
                score += 10
            elif ratio_carb < 0.2:
                score -= 40
                
        # Puntuación agresiva de Grasas (si el usuario pidió keto o alto en grasas, d_gras se eleva)
        if d_gras > 10:
            ratio_gras = gras / d_gras
            if ratio_gras >= 0.8:
                score += 25
            elif ratio_gras >= 0.5:
                score += 15
            elif ratio_gras < 0.2:
                score -= 40

        # Penalizar platos con macros extremas solo si no las pidieron
        if gras > kcal * 0.6 and d_gras < 20:
            score -= 40
        if carb < 2 and prot < 5 and d_carb < 20:
            score -= 40

        # ── Penalización extra por exceso calórico para el momento ────────────
        # Aunque el plato cubra bien el déficit acumulado del día,
        # si excede el techo del horario (ej. 524 kcal en cena → máx 520),
        # lo penalizamos fuertemente para que no gane el ranking.
        if momento_dia and momento_dia != "cualquiera":
            _, kcal_max_momento = _RANGOS_MOMENTO.get(momento_dia.lower(), (0.0, 1200.0))
            if kcal > kcal_max_momento:
                exceso_pct = (kcal - kcal_max_momento) / kcal_max_momento
                penalizacion = min(50.0, exceso_pct * 120)  # hasta -50 pts
                score -= penalizacion
                logger.debug(
                    "[Score] '%s' penalizado %.1f pts por exceso calórico para %s (%.0f > %.0f kcal)",
                    "plato", penalizacion, momento_dia, kcal, kcal_max_momento,
                )

        return max(0.0, min(100.0, score))

    # ─────────────────────────────────────────────────────────────────────────
    # DIVERSIDAD Y SELECCIÓN
    # ─────────────────────────────────────────────────────────────────────────

    def _seleccionar_con_diversidad(
        self,
        candidatos: List[Dict],
        n: int,
        rng: random.Random,
    ) -> List[Dict]:
        """
        Selecciona N platos garantizando variedad de tipo.

        No muestra 3 platos idénticos de pollo seguidos.
        """
        seleccionados = []
        nombres_vistos = set()

        # Agrupar por "tipo" para diversidad
        def categoria(nombre: str) -> str:
            n = nombre.lower()
            for kw in ("pollo", "pescado", "res", "cerdo", "huevo"):
                if kw in n:
                    return kw
            for kw in ("sopa", "caldo", "crema"):
                if kw in n:
                    return "sopa"
            for kw in ("ensalada", "vegetal", "verdura"):
                if kw in n:
                    return "vegetal"
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

        # Si no alcanzamos N, relajar restricción de categoría
        if len(seleccionados) < n:
            for candidato in candidatos:
                if len(seleccionados) >= n:
                    break
                nombre_lower = candidato["nombre"].lower().strip()
                if nombre_lower not in nombres_vistos:
                    nombres_vistos.add(nombre_lower)
                    seleccionados.append(candidato)

        return seleccionados

    # ─────────────────────────────────────────────────────────────────────────
    # GENERACIÓN EN TIEMPO REAL (ENRIQUECIMIENTO DE BD)
    # ─────────────────────────────────────────────────────────────────────────

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
                f"Actúa como chef nutricionista. Crea {n_faltantes + 2} platos distintos para '{momento_dia}'.\n"
                f"Deben sumar aproximadamente: {deficit_kcal:.0f} kcal, {deficit_proteina:.0f}g proteína, "
                f"{deficit_carb:.0f}g carbohidratos, {deficit_grasas:.0f}g grasas.{extra_ingrediente}\n"
                "Usa ingredientes reales y comunes en Perú. No uses medidas como 'tazas' ni 'cucharadas', "
                "USA SOLO GRAMOS EXACTOS (ej. '150').\n\n"
                "Responde ÚNICAMENTE con un arreglo JSON válido con esta estructura:\n"
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

            # Ejecutar LLM síncronamente (en un thread si estamos en async loop)
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
                    has_ingredient = ing_clave_norm in nombre.lower()
                    if not has_ingredient:
                        for ing in ings:
                            if ing_clave_norm in ing.get("nombre", "").lower():
                                has_ingredient = True
                                break
                    if not has_ingredient:
                        logger.warning(f"Plato LLM '{nombre}' descartado por no contener '{ingrediente_clave}'.")
                        continue
                
                # Construir plato (esto lo valida semánticamente y nutricionalmente, y LO GUARDA EN BD)
                resultado = self.plate_builder.construir_plato(
                    nombre_plato=nombre,
                    ingredientes=ings,
                    client_id=client_id,
                    tipo_plato=momento_dia,
                )

                # Si el plato es válido, lo agregamos a la respuesta
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

    # ─────────────────────────────────────────────────────────────────────────
    # HISTORIAL Y PERSISTENCIA
    # ─────────────────────────────────────────────────────────────────────────

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
