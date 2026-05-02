"""
Constructor dinámico de platos: descompone un nombre de plato usando LLM,
crea los registros Plato + PlatoIngrediente y genera los pasos de preparación.

Activado desde asistente_registro_comida.py (Capa 1.5) cuando un plato no existe
en el catálogo local y la consulta parece un plato completo (≥2 palabras).
"""
from __future__ import annotations

import json
import re
import unicodedata
from typing import List, Optional

from sqlalchemy.orm import Session

from app.models.plato import Plato, PlatoIngrediente


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
}

# ─── Filtro de coherencia semántica ──────────────────────────────────────────
# Ingredientes que NO deben aparecer en platos con ciertos patrones de nombre
_CONFLICTOS_SEMANTICOS: list[tuple[set[str], frozenset[str]]] = [
    # Platos de cebiche: sin zanahoria, tomate crudo ni jengibre
    ({"cebiche", "ceviche"},
     frozenset({"zanahoria", "jengibre", "ketchup", "salsa de tomate"})),
    # Panes y tostadas: sin papas, arroz ni pollo entero
    ({"tostada", "pan tostado", "sandwich", "sandw"},
     frozenset({"arroz blanco", "papa cocida", "papa sancochada"})),
    # Sopas/cremas: sin aceitunas ni embutidos fríos
    ({"sopa", "crema de", "caldo"},
     frozenset({"mayonesa", "jamonada", "aceitunas", "jamon"})),
    # Postres: sin ingredientes salados de fondo
    ({"torta", "queque", "bizcocho", "mousse", "flan"},
     frozenset({"ajo", "cebolla", "comino", "oregano"})),
]


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
    Retorna la lista corregida.
    """
    min_kcal, max_kcal = _RANGOS_CALORICOS.get(tipo_plato, (150, 1000))
    kcal_actual = _calcular_kcal_resueltos(resueltos)

    if kcal_actual <= 0:
        return resueltos

    if kcal_actual > max_kcal:
        factor = max_kcal / kcal_actual
        corregidos = [(alim, max(5, round(gramos * factor))) for alim, gramos in resueltos]
        print(f"[PlatoConstructor] Autocorrección gramajes: {kcal_actual:.0f}→{_calcular_kcal_resueltos(corregidos):.0f} kcal (límite {max_kcal})")
        return corregidos

    if kcal_actual < min_kcal and len(resueltos) > 0:
        factor = min_kcal / kcal_actual
        # Solo escalar si el factor no es extremo (máx ×2.5) para no crear porciones absurdas
        if factor <= 2.5:
            corregidos = [(alim, round(gramos * factor)) for alim, gramos in resueltos]
            print(f"[PlatoConstructor] Autocorrección gramajes: {kcal_actual:.0f}→{_calcular_kcal_resueltos(corregidos):.0f} kcal (mínimo {min_kcal})")
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
            print(
                f"[PlatoConstructor] Coherencia semántica: {eliminados} ingrediente(s) "
                f"incompatibles eliminados de '{nombre_plato}'"
            )

    return filtrados


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
            f"Responde SOLO JSON válido (array):\n"
            f'[{{"nombre_es":"<ingrediente>","gramos":<entero>}},...]\n'
            f"Reglas OBLIGATORIAS:\n"
            f"- Total gramos entre 400 y 750 para platos completos, 100-200 para snacks\n"
            f"- Máximo 8 ingredientes\n"
            f"- Nombres genéricos (sin marcas)\n"
            f"- Incluye guarniciones típicas peruanas si aplica\n"
            f"- SIEMPRE expresa cantidades en GRAMOS enteros, NUNCA en unidades genéricas\n"
            f"  Ejemplos de conversión: 1 huevo→55g, 1 plátano→120g, 1 cebolla→80g,\n"
            f"  1 tomate→100g, 1 papa→150g, 1 limón→50g, 1 palta→150g"
        )
        resp = await ia_engine._llamar_groq(prompt=prompt, max_tokens=300, temp=0.05)
        resp = re.sub(r"```(?:json)?", "", resp).strip().strip("`")
        m = re.search(r"\[.*\]", resp, re.DOTALL)
        if not m:
            return []
        items = json.loads(m.group(0))
        return [
            {"nombre_es": str(it.get("nombre_es", "")).strip(),
             "gramos": max(1, int(it.get("gramos", 50)))}
            for it in items
            if it.get("nombre_es") and str(it.get("nombre_es", "")).strip()
        ]
    except Exception as e:
        print(f"[PlatoConstructor] Error descomponiendo '{nombre_plato}': {e}")
        return []


async def _generar_preparacion_llm(
    nombre_plato: str, nombres_ingredientes: List[str]
) -> Optional[list]:
    """
    Genera pasos de preparación como array JSON de strings.
    Retorna lista de strings o None si falla.
    """
    try:
        from app.services.ia_service import ia_engine
        lista = ", ".join(nombres_ingredientes)
        prompt = (
            f"Genera los pasos de preparación del plato \"{nombre_plato}\" "
            f"usando TODOS estos ingredientes: {lista}.\n"
            f"Responde SOLO un array JSON de 4 a 6 strings imperativas en español.\n"
            f"Reglas OBLIGATORIAS:\n"
            f"- Cada paso DEBE mencionar al menos un ingrediente de la lista\n"
            f"- Todos los ingredientes deben aparecer en al menos un paso\n"
            f"- Sin números de paso, sin artículos redundantes\n"
            f'Ejemplo: ["Sancocha las papas en agua con sal.", '
            f'"Sofríe la cebolla y el ajo hasta dorar.", '
            f'"Agrega el pollo y cocina 10 minutos."]'
        )
        resp = await ia_engine._llamar_groq(prompt=prompt, max_tokens=500, temp=0.1)
        resp = unicodedata.normalize("NFC", resp)
        resp = re.sub(r"```(?:json)?", "", resp).strip().strip("`")
        m = re.search(r"\[.*\]", resp, re.DOTALL)
        if not m:
            return None
        pasos = json.loads(m.group(0))
        return [str(p).strip() for p in pasos if str(p).strip()]
    except Exception as e:
        print(f"[PlatoConstructor] Error generando preparacion para '{nombre_plato}': {e}")
        return None


# ─── Trinidad Nutricional (logging, no bloqueante) ───────────────────────────

def _verificar_trinidad_nutricional(plato: Plato) -> None:
    """Log de integridad: confirma que macros = Σ(alimento.macro × gramos/100)."""
    try:
        macros = plato.calcular_macros()
        print(
            f"[Trinidad] Plato '{plato.nombre}' (id={plato.id}): "
            f"kcal={macros['calorias']} prot={macros['proteinas_g']} "
            f"carb={macros['carbohidratos_g']} gras={macros['grasas_g']}"
        )
    except Exception as e:
        print(f"[Trinidad] Error verificando plato id={getattr(plato, 'id', '?')}: {e}")


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

    nombre_norm = _norm(nombre_plato)
    if not nombre_norm or len(nombre_norm) < 3:
        return None

    # Guardia no-alimento: rechazar si contiene tokens claramente no-alimentarios
    _tokens = set(nombre_norm.split())
    if _tokens & _NO_FOOD_TOKENS:
        _rechazados = _tokens & _NO_FOOD_TOKENS
        print(f"[PlatoConstructor] '{nombre_plato}' rechazado — tokens no-alimentarios: {_rechazados}")
        return None

    # Verificar si ya existe un plato muy similar
    existing = (
        db.query(Plato)
        .filter(Plato.nombre_normalizado.like(f"{nombre_norm.split()[0]}%"))
        .limit(20)
        .all()
    )
    for p in existing:
        sim = difflib.SequenceMatcher(None, nombre_norm, p.nombre_normalizado or "").ratio()
        if sim >= 0.85:
            print(f"[PlatoConstructor] Plato similar encontrado: '{p.nombre}' (sim={sim:.2f})")
            return p

    # Descomponer ingredientes con LLM
    ingredientes_raw = await _descomponer_plato_llm(nombre_plato)
    if not ingredientes_raw:
        print(f"[PlatoConstructor] LLM no devolvió ingredientes para '{nombre_plato}'")
        return None

    # Filtro de coherencia semántica (antes de resolver en BD)
    ingredientes_raw = _filtrar_coherencia_semantica(nombre_plato, ingredientes_raw)

    # Resolver o crear cada ingrediente en BD
    resueltos: list[tuple] = []  # [(Alimento, gramos)]
    for item in ingredientes_raw:
        nombre_ing_es = item["nombre_es"]
        gramos = item["gramos"]
        nombre_ing_norm = _norm(nombre_ing_es)
        alim = await _buscar_o_crear_alimento_async(db, nombre_ing_norm, nombre_ing_es)
        if alim:
            resueltos.append((alim, gramos))
        else:
            print(f"[PlatoConstructor] Ingrediente no resuelto: '{nombre_ing_es}' — omitido")

    if len(resueltos) < 2:
        print(f"[PlatoConstructor] Insuficientes ingredientes resueltos ({len(resueltos)}) para '{nombre_plato}'")
        return None

    # Autocorrección de gramajes si las kcal están fuera del rango para tipo_plato
    resueltos = _autocorregir_gramajes(resueltos, tipo_plato)

    # Crear el registro Plato
    nombre_display = _initcap(nombre_plato)
    try:
        plato = Plato(
            nombre=nombre_display[:255],
            nombre_normalizado=nombre_norm[:255],
            tipo_plato=tipo_plato,
            origen="llm",
        )
        db.add(plato)
        db.flush()  # obtener plato.id sin commit

        # Crear PlatoIngrediente por cada ingrediente resuelto
        for orden, (alim, gramos) in enumerate(resueltos):
            pi = PlatoIngrediente(
                plato_id=plato.id,
                alimento_id=alim.id,
                gramos=float(gramos),
                orden=orden,
            )
            db.add(pi)

        db.flush()

        # Generar preparación con LLM
        nombres_ings = [alim.nombre for alim, _ in resueltos]
        preparacion = await _generar_preparacion_llm(nombre_plato, nombres_ings)
        if preparacion:
            plato.preparacion = preparacion

        db.commit()
        db.refresh(plato)
        print(f"[PlatoConstructor] Plato '{nombre_display}' creado (id={plato.id}, {len(resueltos)} ingredientes)")
        _verificar_trinidad_nutricional(plato)
        return plato

    except Exception as e:
        db.rollback()
        print(f"[PlatoConstructor] Error creando plato '{nombre_plato}': {e}")
        return None
