"""
nlp_food_extractor.py
─────────────────────
Extractor de alimentos desde texto libre usando Llama-3 como parser JSON
+ cálculo determinista desde BD (alimento_unidades).

Flujo:
  1. Llama-3 extrae JSON estructurado (NUNCA calcula calorías)
  2. Python resuelve cada ítem contra la BD
  3. Si un alimento no está en BD → fallback USDA → guarda en BD
  4. Python calcula los macros finales (siempre igual para la misma entrada)

Uso:
  from app.services.nlp_food_extractor import NLPFoodExtractor
  extractor = NLPFoodExtractor(ia_service, db)
  resultado = await extractor.extraer(mensaje)
"""

from __future__ import annotations

import json
import re
import unicodedata
import urllib.parse
import urllib.request

# Importación lazy para evitar import circular — se resuelve en tiempo de llamada
def _normalizar_voz_comida(texto: str) -> str:
    """Wrapper lazy de _normalizar_voz del módulo de ejercicios."""
    try:
        from app.services.asistente.asistente_registro_ejercicio import _normalizar_voz
        return _normalizar_voz(texto)
    except Exception:
        return texto
from dataclasses import dataclass


def _sufijos_con_compat(a: str, b: str) -> bool:
    """Guard 'con X': evita que 'tortilla con pan' matchee 'tortilla con atún'."""
    def _suf(s: str) -> list[str]:
        idx = s.rfind(" con ")
        return s[idx + 5:].split() if idx >= 0 else []
    s1, s2 = _suf(a), _suf(b)
    if not s1 or not s2:
        return True
    return bool(set(s1) & set(s2))
from typing import List, Optional

from sqlalchemy.orm import Session

from app.models.alimento import Alimento
from app.models.alimento_alias import AlimentoAlias
from app.models.alimento_unidad import AlimentoUnidad
from app.services.asistente.asistente_nutricion import coherencia_proteina_platos
from app.services.nutricional_result import validar_macros_atwater
from app.core.logging_config import get_logger

logger = get_logger("nlp_food_extractor")

# Si un ítem sale con kcal extremadamente alto, solo advertimos (no bloquea registro).
# Esto evita registros absurdos por cantidades/unidades mal interpretadas.
KCAL_ITEM_WARN = 1500.0

# ─── PROMPT que va a Llama-3 ──────────────────────────────────────────────────
EXTRACTION_PROMPT = """Eres un extractor de datos nutricionales. Tu ÚNICA tarea es analizar el texto del usuario y devolver un JSON array.

REGLAS ESTRICTAS:
- Si el texto NO menciona ningún alimento real (comida o bebida), devuelve EXACTAMENTE: []
  Ejemplos de cosas que NO son alimentos: juegos, ropa, libros, música, objetos, emociones,
  materiales (hierro, madera, plástico), actividades (deporte, trabajo, estudio).
  Si el usuario dice "comi juegos frito" → [] porque "juegos" no es comida.
- REGLA ANTI-FICCIÓN — PALABRAS INVENTADAS: Si el nombre del alimento contiene una palabra
  que claramente no existe como alimento, ingrediente ni marca en ningún idioma conocido
  (español, inglés, quechua, italiano, francés, japonés…), devuelve EXACTAMENTE: []
  NUNCA sustituyas una palabra inventada por un alimento real similar.
  Ejemplos de palabras inventadas: "florbonix", "zarblak", "frublatex", "glurpix", "snorflax".
  Si el usuario dice "comí florbonix tostado" → [] porque "florbonix" no es un alimento real.
  Si el usuario dice "tomé bebida de zorblax" → [] porque "zorblax" no existe.
  REGLA CLAVE: si tienes dudas sobre si una palabra es un alimento real o una marca, pero suena
  completamente inventada (sin raíz en ningún idioma), devuelve []. Es mejor rechazar que inventar.
- REGLA BLOQUEADOR — ANIMALES DOMÉSTICOS/NO COMESTIBLES: Si el alimento mencionado es carne de
  animal doméstico o no apto para consumo humano en contexto culinario normal, devuelve EXACTAMENTE: []
  NUNCA sustituyas por otro alimento similar. Lista de animales bloqueados:
  perro, gato, caballo, rata, ratón, culebra, serpiente, lobo, zorro, mono, loro, paloma doméstica.
  Ejemplos: "carne de perro" → [] | "comi gato frito" → [] | "bistec de caballo" → []
  IMPORTANTE: cuy, alpaca, llama y pato SÍ son comestibles en Perú → extraer normalmente.
- NO calcules calorías. NO sumes nada. SOLO extrae lo que el usuario mencionó.
- Extrae EXACTAMENTE lo que dijo el usuario, sin agregar ingredientes ni complementos que NO mencionó.
  Si el usuario dijo "pollo al horno", extrae "pollo al horno" — no añadas arroz, verduras ni nada más.
- NO separes platos tradicionales conocidos ni combinaciones de arroz/cereal con acompañamiento. Son UN solo ítem:
  "lomo saltado", "aji de gallina", "arroz con leche", "arroz con pollo", "arroz con verduras", "arroz con atun",
  "choclo con queso", "pan con huevo", "yogur con granola", "yogur con fruta", "yogur con avena",
  "platano con yogur", "avena con leche", "avena con granola", "cafe con leche", "te con leche",
  "pollo con arroz", "tallarines con carne", "sopa de verduras", "sopa de lentejas", "crema de verduras",
  "ensalada de pollo", "ensalada de frutas", "tortilla de huevo".
- REGLA CRÍTICA: "ensalada de X", "ensalada de X con Y", "sopa de X", "crema de X" siempre son UN solo ítem.
  Ejemplos: "ensalada de plátano con pollo" → UN ítem; "ensalada de frutas con yogur" → UN ítem.
  NUNCA extraigas los ingredientes internos de una ensalada/sopa/crema como ítems separados.
- REGLA MENÚ PERUANO: cuando el usuario describe un almuerzo con "X con sopa de Y" o "X y sopa de Y",
  la sopa es SIEMPRE un ítem SEPARADO (en Perú el menú incluye sopa + segundo + bebida por separado).
  Ejemplos:
    "arroz a la naranja con sopa de fideos" → [{alimento:"arroz a la naranja",...}, {alimento:"sopa de fideos",...}]
    "lomo saltado con sopa de verduras"     → [{alimento:"lomo saltado",...}, {alimento:"sopa de verduras",...}]
    "arroz con pollo y sopa de letras"      → [{alimento:"arroz con pollo",...}, {alimento:"sopa de letras",...}]
- REGLA CRÍTICA: "plátano" siempre es "plátano maduro" (banana dulce). NUNCA uses "plátano verde cocido".
  "plátano verde" solo si el usuario dice explícitamente "plátano verde" o "patacón" o "tostón".
- REGLA ESPECIAL: cantidades numéricas antes de un ítem discreto dentro de una combinación.
  Si el usuario dice "avena con 2 panes" o "arroz con 3 huevos", el ítem con número ES SEPARADO:
  "avena con 2 panes con mermelada" → [{alimento:"avena",cantidad:1,unidad:"porcion"}, {alimento:"pan con mermelada",cantidad:2,unidad:"unidad"}]
  "avena con 2 panes" → [{alimento:"avena",cantidad:1,unidad:"porcion"}, {alimento:"pan",cantidad:2,unidad:"unidad"}]
  EXCEPCIÓN: si el número describe la receta (ej. "sopa de 3 verduras", "arroz con 2 tipos de menestra") → mantener como 1 solo ítem.
  La señal de separación es: número + ítem discreto contable (pan, huevo, fruta, galleta, naranja, manzana, etc.).
- REGLA ESPECIAL: fracciones de porción — extrae el ítem completo con la cantidad decimal correspondiente.
  "medio X" / "media X"          → cantidad=0.5
  "un cuarto de X" / "cuarto X"  → cantidad=0.25
  "un octavo de X" / "octavo X"  → cantidad=0.125
  Ejemplos:
    "media ensalada de plátano con pollo" → [{alimento:"ensalada de platano con pollo", cantidad:0.5, unidad:"porcion"}]
    "medio arroz con verduras"            → [{alimento:"arroz con verduras", cantidad:0.5, unidad:"porcion"}]
    "un cuarto de torta de chocolate"     → [{alimento:"torta de chocolate", cantidad:0.25, unidad:"porcion"}]
    "un octavo de pizza"                  → [{alimento:"pizza", cantidad:0.125, unidad:"porcion"}]
  NUNCA separes los ingredientes internos cuando se diga una fracción.
- REGLA ESPECIAL: si el usuario dice "X con Y" donde Y es una verdura/guarnición, trátalo como UN plato.
  Ejemplo: "arroz con verduras", "pollo con papas", "pescado con ensalada" → UN ítem cada uno.
- Convierte cantidades verbales exactamente así:
    "medio" o "media"             → 0.5
    "un cuarto" o "cuarto"        → 0.25
    "un octavo" o "octavo"        → 0.125
    "un" / "una" / "uno"          → 1.0
    "dos"                         → 2.0
    "tres"                        → 3.0
    "cuatro"                      → 4.0
    "un par"                      → 2.0
    "un poco"                     → 0.5
    Si no hay cantidad             → 1.0
- REGLA CRÍTICA para "cantidad + alimento en plural":
    "dos peras" → {alimento:"pera", cantidad:2.0, unidad:"unidad"}  ← NO "dos peras" como alimento
    "tres manzanas" → {alimento:"manzana", cantidad:3.0, unidad:"unidad"}
    "dos huevos" → {alimento:"huevo", cantidad:2.0, unidad:"unidad"}
    "dos tazas de leche" → {alimento:"leche", cantidad:2.0, unidad:"taza"}
    El número SIEMPRE va en "cantidad", NUNCA en el nombre del alimento.
    PROHIBIDO: {alimento:"dos peras"}, {alimento:"tres manzanas"}, {alimento:"dos huevos"}
- Elige la unidad más lógica:
    pan/galleta/huevo/fruta/unidad discreta → "unidad"
    leche/jugo/gaseosa/agua/bebida → "vaso"
    platos de comida completos/arroz/fideos/menestras → "porcion"
    aceite/salsa → "cucharada"
    Si el usuario especifica unidad (taza, vaso, plato), úsala.
    Si el usuario especifica gramos explícitamente (ej: "300g de pollo", "200 gramos de arroz") →
      usa unidad: "g" y cantidad: <número de gramos>. Ejemplos:
      "comí 250g de pollo al horno con plátano" → [{alimento:"pollo al horno con platano", cantidad:250, unidad:"g"}]
      "comi 100 gramos de pollo saltado" → [{alimento:"pollo saltado", cantidad:100, unidad:"g"}]
      "tome 300 gramos de arroz con leche" → [{alimento:"arroz con leche", cantidad:300, unidad:"g"}]
      REGLA CRÍTICA: "X gramos de [alimento]" es SIEMPRE UN SOLO objeto con unidad:"g". NUNCA ignores los gramos
      aunque el alimento sea normalmente discreto (fruta, etc.). NUNCA separes el número en un ítem distinto.
      MENSAJES MIXTOS — cuando algunos ítems tienen gramos y otros tienen conteo:
      "comi 50g de palta y 2 pan integral" → [{alimento:"palta",cantidad:50,unidad:"g"},{alimento:"pan integral",cantidad:2,unidad:"unidad"}]
      "tome 100ml de leche y 3 galletas" → [{alimento:"leche",cantidad:100,unidad:"ml"},{alimento:"galleta",cantidad:3,unidad:"unidad"}]
      "comi 200g de pollo y una manzana" → [{alimento:"pollo",cantidad:200,unidad:"g"},{alimento:"manzana",cantidad:1,unidad:"unidad"}]
      En mensajes mixtos: CADA ítem mantiene SU unidad. El ítem con gramos usa unidad:"g", el ítem con conteo usa unidad:"unidad".
    Si el usuario especifica ml o litros explícitamente para bebidas →
      usa unidad: "ml" y cantidad: <número de ml>. Ejemplos:
      "tome 500ml de gaseosa" → [{alimento:"gaseosa", cantidad:500, unidad:"ml"}]
      "tome una jugo de naranja de 200ml" → [{alimento:"jugo de naranja", cantidad:200, unidad:"ml"}]
      "bebi 250ml de jugo de manzana" → [{alimento:"jugo de manzana", cantidad:250, unidad:"ml"}]
      NUNCA conviertas ml explícitos a "vaso".
      REGLA ANTI-DUPLICADO: "jugo de naranja", "jugo de mango", "jugo de piña" son UN SOLO alimento.
      NUNCA extraigas la fruta por separado si ya está incluida en el nombre del jugo.
      INCORRECTO: [{alimento:"jugo de naranja"}, {alimento:"naranja"}] ← PROHIBIDO
      CORRECTO:   [{alimento:"jugo de naranja", cantidad:200, unidad:"ml"}]
- Usa nombres EXACTOS y genéricos en español. Para bebidas: "agua", "gaseosa", "jugo de naranja".
- BEBIDAS DE CEREALES/GRANOS: cuando el usuario pide un "vaso de cebada", "taza de cebada" o solo "cebada"
  en contexto de bebida caliente, usa SIEMPRE el nombre "agua de cebada" (no "cebada" a secas ni "cebada con cáscara").
  "emoliente" o "vaso de emoliente" → "emoliente de cebada".
  "avena bebida" o "vaso de avena" (la bebida andina líquida, no el cereal) → "avena bebida".
- MARCAS COMERCIALES: Si el usuario menciona una marca de bebida (Inca Kola, Coca Cola, Pepsi, Sprite, Fanta, etc.), usa SOLO el nombre genérico. Ejemplo: "gaseosa inca kola" → {alimento:"gaseosa"}. Nunca registres la marca como ítem separado.
- BEBIDAS INDEPENDIENTES — REGLA ABSOLUTA: "gaseosa", "jugo", "chicha", "limonada", "refresco",
  "cerveza", "agua con sabor" y cualquier bebida explícita son SIEMPRE un objeto SEPARADO en el
  array raíz. NUNCA los incluyas en "con_extra" de otro alimento.
  El campo "con_extra" SOLO acepta strings simples de ingredientes sólidos (ej: "queso", "arroz").
  Ejemplo CORRECTO para "lomo saltado con su gaseosa":
  [{"alimento":"lomo saltado","cantidad":1.0,"unidad":"porcion","sin":[],"con_extra":[]},
   {"alimento":"gaseosa","cantidad":1.0,"unidad":"vaso","sin":[],"con_extra":[]}]
  Ejemplo INCORRECTO (NUNCA hagas esto):
  [{"alimento":"lomo saltado","con_extra":["gaseosa"]}]  ← PROHIBIDO para bebidas
- MODIFICADORES: "sin X" → agregar en campo "sin" (strings simples).
  "con extra X" / "con más X" → campo "con_extra" (strings simples, SOLO sólidos: "queso", "huevo extra", "más arroz").

IMPORTANTE — El texto puede venir de transcripción de voz:
- Sin puntuación: "comí arroz con pollo doscientos gramos y una gaseosa"
- Números escritos: "setenta gramos", "ciento veinte mililitros", "dos y medio"
- Muletillas ya eliminadas, pero puede haber variaciones fonéticas

Convierte números escritos a dígitos: "setenta"→70, "ciento veinte"→120, "dos y medio"→2.5
Ignora muletillas residuales que no sean alimentos.

Responde SOLO con JSON array válido, sin texto adicional, sin markdown.

[
  {
    "alimento": "<nombre_generico_español>",
    "cantidad": <float>,
    "unidad": "<unidad>",
    "sin": [],
    "con_extra": []
  }
]

Texto del usuario: "{texto}"
"""

# ─── Bebidas con 0 calorías (no afectan macros) ───────────────────────────────
BEBIDAS_CERO_KCAL = {
    "agua", "agua mineral", "agua con gas", "agua sola",
    "te", "te sin azucar", "infusion", "cafe solo", "cafe negro",
}

# ─── Post-procesador: "avena con 2 panes" → ["avena", {pan, qty:2}] ──────────
# Separa "X con N <ítem_discreto>" en dos ítems cuando N≥2 y el ítem es contable.
_DISCRETOS_CONTABLES = frozenset({
    "pan", "panes", "huevo", "huevos", "manzana", "manzanas", "naranja", "naranjas",
    "platano", "platanos", "mandarina", "mandarinas", "galleta", "galletas",
    "biscocho", "biscochito", "tostada", "tostadas", "fruta", "frutas",
    "biscot", "biscocho", "empanada", "empanadas",
})
_RE_CON_N_ITEM = re.compile(
    r"^(.+?)\s+con\s+(\d+)\s+(.+)$", re.IGNORECASE
)

def _separar_con_n_items(items_raw: list[dict]) -> list[dict]:
    """Divide 'avena con 2 panes con mermelada' → ['avena', {pan con mermelada, qty:2}]."""
    resultado = []
    for item in items_raw:
        nombre = item.get("alimento", "")
        m = _RE_CON_N_ITEM.match(nombre)
        if m:
            base   = m.group(1).strip()
            qty    = int(m.group(2))
            resto  = m.group(3).strip()
            primer_token = _norm(resto).split()[0] if resto else ""
            if primer_token in _DISCRETOS_CONTABLES and qty >= 2:
                resultado.append({**item, "alimento": base})
                resultado.append({**item, "alimento": resto, "cantidad": float(qty), "unidad": "unidad"})
                continue
        resultado.append(item)
    return resultado


# ─── Normalizaciones de texto antes de enviar al LLM ─────────────────────────
PRE_NORM_PATRONES = [
    # ── Muletillas de VOZ (input hablado transcripto) ─────────────────────────
    # El asistente recibe mensajes de voz → el texto puede contener muletillas,
    # dudas y conectores vacíos que confunden al LLM extractor.
    (r"(?i)^(este|pues|bueno|oye\s+pues)\s*[,.]?\s*", ""),  # muletilla al inicio
    (r"(?i)\b(mm+h?|eeh?|aah?|uh+|uhm+|hmm+)\b[,.]?\s*", " "),
    (r"(?i)\bo\s+sea\s+(que\s+)?", " "),
    (r"(?i)\bcomo\s+que\s+", " "),
    (r"(?i)\bbueno\s+(?:pues\s+)?(?=\w)", " "),
    (r"(?i)\bpues\s+(?=\w)", " "),
    (r"(?i)\bla\s+verdad\s+(?:es\s+)?(?:que\s+)?", " "),
    (r"(?i)\besto\s+es\b[,.]?\s*", " "),
    # "y este y" / "este que" → eliminar (relleno de voz)
    (r"(?i)\by\s+este\s+", " "),
    (r"(?i)\beste\s+que\s+", " "),
    # ── Conectores temporales de consumo ──────────────────────────────────────
    (r"(?i)\b(y\s+)?despu[eé]s\s+(com[ií]|tom[eé]|beb[ií])\b", " y "),
    (r"(?i)\b(y\s+)?luego\s+(com[ií]|tom[eé]|beb[ií])\b",     " y "),
    (r"(?i)\btambi[eé]n\s+(com[ií]|tom[eé]|beb[ií])\b",        " y "),
    # "y de tomar/comer/beber X" → " y X"
    (r"(?i)\by\s+de\s+(?:tom[aáe]r|com[eé]r|beb[eé]r)\s+", " y "),
    # Hint de volumen al final "como 300ml" → eliminar (causa duplicados)
    (r"(?i),?\s+(?:como|unos?|unas?)\s+\d+\s*(?:ml|cl|cc|litros?)\s*$", ""),
    # Separador implícito por unidad: "arroz con huevo una taza de avena" → "... y una taza"
    (r"(?i)(?<![cC][oO][nN])\s+(?=un[ao]?\s+(?:taza|vaso|copa|plato|porci[oó]n)\s+de\s+)", " y "),
    # Redundancias comunes
    (r"(?i)\bun\s+poco\s+de\b",   "medio "),
    (r"\s{2,}",                    " "),
]

# ─── Marcas comerciales de bebidas → nombre genérico ─────────────────────────
# "gaseosa inkacola" → "gaseosa", "inca kola" solo → "gaseosa"
# Se aplica ANTES de enviar al LLM para evitar que el LLM las registre como
# alimentos separados (LLM estimaría macros inventados para "inkacola").
_MARCAS_GASEOSAS_RE = re.compile(
    r"(?i)\b(?:inca[\s\-]?kola|inka[\s\-]?kola|inkakola|inkacola|"
    r"coca[\s\-]?cola|cocacola|"
    r"pepsi(?:[\s\-]?cola)?|"
    r"sprite|fanta|"
    r"seven[\s\-]?up|7[\s\-]?up|"
    r"kola[\s\-]?real|"
    r"guarana)\b"
)

# ─── Palabras que indican negación → no registrar nada ─────────────────────
NEGACION_PATRONES = [
    r"(?i)^\s*(hoy\s+)?no\s+com[ií]",          # "no comi", "hoy no comi"
    r"(?i)^\s*(hoy\s+)?no\s+beb[ií]",          # "no bebi"
    r"(?i)^\s*(hoy\s+)?no\s+tom[eé]",          # "no tome"
    r"(?i)\bno\s+com[ií]\s+(nada|ning)",       # "no comi nada"
    r"(?i)\bsin\s+comer\b",                     # "me fui sin comer"
    r"(?i)\bno\s+prob[eé]\b",                  # "no probe nada"
]

# Palabras que claramente NO son alimentos → nunca consultar USDA con ellas
# Incluye ficción, objetos, materiales, actividades, conceptos abstractos
# y animales domésticos/no aptos para consumo humano.
NO_ALIMENTOS: frozenset[str] = frozenset({
    # Ficción / palabras inventadas (incluir variantes comunes de prueba)
    "unicornio", "dragon", "flobonix", "florbonix", "zombie", "alien", "cripton",
    "zarblak", "frublatex", "glurpix", "snorflax", "zorblax",
    "monstruo", "magico", "invisible", "virtual", "digital", "fake",
    # ── ANIMALES DOMÉSTICOS / NO COMESTIBLES (Bug 5 fix) ─────────────────
    # CRÍTICO: bloquear sustitución silenciosa (ej: perro → cuy)
    "perro", "gato", "caballo", "rata", "raton", "culebra",
    "serpiente", "lobo", "zorro", "mono", "loro", "hamster",
    # ─────────────────────────────────────────────────────────────────────
    # Materiales y objetos inorgánicos
    "hierro", "acero", "madera", "plastico", "vidrio", "metal", "piedra",
    "cemento", "carbon", "petroleo", "gasolina", "tierra", "arena", "barro",
    "clavo", "tornillo", "alambre", "cable", "tubo", "pintura",
    # Ropa y accesorios
    "ropa", "camisa", "pantalon", "zapato", "calcetines", "vestido",
    "chaqueta", "abrigo", "sombrero", "corbata", "guantes", "bolso",
    # Muebles y utensilios del hogar
    "mesa", "silla", "cama", "sofa", "lampara", "television", "espejo",
    "ventana", "puerta", "piso", "techo", "pared",
    # Tecnología
    "computadora", "telefono", "celular", "tableta", "computador",
    "pantalla", "teclado", "mouse", "auricular",
    # Útiles / papelería
    "libro", "cuaderno", "lapiz", "boligrafo", "papel", "tijeras",
    "regla", "borrador", "mochila", "cartera",
    # Actividades y conceptos abstractos
    "juego", "juegos", "deporte", "deportes", "ejercicio", "musica",
    "trabajo", "tarea", "reunion", "clase", "estudio", "examen",
    "dinero", "plata", "billete", "moneda", "tarjeta", "cheque",
    "amor", "odio", "tristeza", "alegria", "miedo", "felicidad",
    "idea", "pensamiento", "sueno", "silencio", "ruido",
    # Partes del cuerpo (no comestibles en contexto culinario)
    "pelo", "cabello", "unas", "piel", "sudor",
    # Palabras ofensivas / absurdas en contexto de comida
    "caca", "orina", "excremento", "heces", "vomito", "basura", "veneno",
    "veneno", "toxico", "explosivo", "bomba",
})

# Set específico de animales domésticos no comestibles para bloqueo rápido por token
# (complementa NO_ALIMENTOS con detección por "carne de X")
_ANIMALES_NO_COMESTIBLES: frozenset[str] = frozenset({
    "perro", "gato", "caballo", "rata", "raton", "culebra",
    "serpiente", "lobo", "zorro", "mono", "loro", "hamster",
})

# Modificadores ficticios: si acompañan a cualquier sustantivo, el ítem es ficticio.
# Ej: "carne de unicornio", "huevo de dragón", "leche de fénix" → BLOQUEADO.
_MODIFICADORES_FICTICIOS: frozenset[str] = frozenset({
    "unicornio", "dragon", "fenix", "centauro", "hada", "mitico",
    "olimpico", "fantasia", "magico", "mitologico", "quimera",
    "grifo", "hidra", "ciclope", "sirena", "pixie", "goblin",
})


def contiene_modificador_ficticio(mensaje: str) -> bool:
    """Devuelve True si el mensaje contiene algún modificador de ingrediente ficticio."""
    tokens = set(_norm(mensaje).split())
    return bool(tokens & _MODIFICADORES_FICTICIOS)


def _nombre_es_no_alimento(nombre: str) -> bool:
    """
    True si el nombre extraído por el LLM es claramente no-comestible.
    Verifica si TODAS las palabras significativas (>3 letras) están en NO_ALIMENTOS.
    Excepción: palabras de método de cocción ('frito', 'cocido', 'asado', etc.)
    no cuentan como 'palabras base' del alimento.
    """
    _METODOS_COCCION = frozenset({
        "frito", "fritos", "frita", "fritas", "cocido", "cocida", "asado", "asada",
        "hervido", "hervida", "horneado", "horneada", "a", "al", "la", "el", "de",
        "con", "sin", "en", "y", "o",
    })
    palabras_base = [
        w for w in _norm(nombre).split()
        if len(w) > 3 and w not in _METODOS_COCCION
    ]
    if not palabras_base:
        return False
    return all(w in NO_ALIMENTOS for w in palabras_base)


# Números en texto → float (para detectar cantidades extremas)
NUMEROS_TEXTO: dict[str, float] = {
    "un": 1, "una": 1, "uno": 1,
    "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5,
    "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10,
    "once": 11, "doce": 12, "media": 0.5, "medio": 0.5,
    "cuarto": 0.25, "octavo": 0.125,
}

# ─── Mapa de unidades globales (fallback cuando el alimento no tiene unidad en BD) ──
UNIDADES_GLOBALES: dict[str, float] = {
    "unidad": 100.0,     # se sobreescribe con el peso real del alimento si existe
    "porcion": 250.0,    # porción estándar de plato/comida (~1 taza grande servida)
    "vaso": 240.0,
    "taza": 240.0,
    "cucharada": 14.0,
    "cucharadita": 5.0,
    "plato": 300.0,
    "rebanada": 30.0,
    "tajada": 30.0,
    "filete": 120.0,
    "loncha": 30.0,
    "lata": 150.0,
    "botella": 500.0,
    "pote": 200.0,
    "sobre": 20.0,
    # Unidades coloquiales peruanas
    "tarro": 400.0,
    "bolsita": 50.0,
    "paquete": 100.0,
    # Unidades de masa/volumen explícitas — 1 unidad = N gramos/ml
    # CRÍTICO: sin estas entradas el fallback 100.0 causa multiplicación ×100
    "g": 1.0,
    "gr": 1.0,
    "gramo": 1.0,
    "gramos": 1.0,
    "kg": 1000.0,
    "kilo": 1000.0,
    "kilogramo": 1000.0,
    "ml": 1.0,
    "cc": 1.0,
    "l": 1000.0,
    "litro": 1000.0,
}

# ─── Pesos específicos por alimento cuando la unidad es "unidad" ──────────────
PESOS_UNIDAD: dict[str, float] = {
    "pan frances":          50.0,
    "pan integral":         28.0,
    "pan de molde":         25.0,
    "huevo entero cocido":  50.0,
    "manzana con cascara": 182.0,
    "platano maduro":      118.0,
    "naranja":             140.0,
    "papa cocida":         150.0,
    "camote cocido":       130.0,
    "fresa":                12.0,
    "galleta":              15.0,   # 1 galleta de soda/vainilla (~15g)
}

USDA_API_KEY = "uFeX5hag2c1mmeR7ueaJj0K86VmsgnQsoxhsyyBt"
NUTRIENT_IDS = {
    1008: "calorias_100g",
    1003: "proteina_100g",
    1005: "carbohidratos_100g",
    1004: "grasas_100g",
    1079: "fibra_100g",
    2000: "azucar_100g",
}


@dataclass
class ItemExtraido:
    alimento: str        # nombre normalizado en BD
    cantidad: float      # número de unidades
    unidad: str          # "vaso", "porcion", "unidad", etc.
    gramos_totales: float
    calorias: float
    proteinas_g: float
    carbohidratos_g: float
    grasas_g: float
    origen: str          # "bd", "usda", "estimado"
    confianza_baja: bool = False  # True cuando el alimento fue estimado por Groq (no encontrado en BD/USDA)
    es_liquido: bool = False       # True cuando la unidad es ml/vaso/taza (Bug 4 fix)


@dataclass
class ResultadoExtraccion:
    items: List[ItemExtraido]
    calorias_total: float
    proteinas_total: float
    carbohidratos_total: float
    grasas_total: float
    nombres: List[str]
    advertencia: Optional[str]


def _promover_bebidas_extras(
    items_raw: list[dict],
    bebidas_keywords: "frozenset[str]",
) -> list[dict]:
    """
    Garantía Python-level: si el LLM puso una bebida en ``con_extra`` de otro
    alimento (como string O como dict), la extrae y la convierte en ítem independiente.
    Evita duplicados si el LLM TAMBIÉN la incluyó como ítem propio en el array raíz.
    """
    def _extra_nombre(extra) -> str:
        """Extrae el nombre legible de un extra (string o dict)."""
        if isinstance(extra, dict):
            return str(extra.get("alimento") or extra.get("nombre") or "").strip()
        return str(extra).strip()

    def _extra_unidad(extra) -> str:
        if isinstance(extra, dict):
            return str(extra.get("unidad") or "vaso").strip()
        return "vaso"

    def _extra_cantidad(extra) -> float:
        if isinstance(extra, dict):
            try:
                return float(extra.get("cantidad") or 1.0)
            except (TypeError, ValueError):
                return 1.0
        return 1.0

    # Nombres de bebidas ya presentes como ítems raíz (para evitar duplicados)
    nombres_bebidas_ya = {
        _norm(it.get("alimento", ""))
        for it in items_raw
        if any(bk in _norm(str(it.get("alimento", "")))
               for bk in bebidas_keywords)
    }

    resultado: list[dict] = []
    bebidas_nuevas: list[dict] = []

    for item in items_raw:
        con_extra = item.get("con_extra") or []
        extras_filtrados = []
        for extra in con_extra:
            nombre_extra = _extra_nombre(extra)
            extra_norm = _norm(nombre_extra)
            es_bebida = any(bk in extra_norm for bk in bebidas_keywords)
            if es_bebida:
                # SIEMPRE quitar bebida de con_extra — evita doble conteo cuando el LLM
                # pone la bebida en con_extra Y también como ítem propio.
                if extra_norm not in nombres_bebidas_ya:
                    # No está como ítem propio aún → promover a ítem nuevo
                    bebidas_nuevas.append({
                        "alimento": nombre_extra,
                        "cantidad": _extra_cantidad(extra),
                        "unidad": _extra_unidad(extra),
                        "sin": [],
                        "con_extra": [],
                    })
                    nombres_bebidas_ya.add(extra_norm)
                    logger.info(
                        "[NLPExtractor] Bebida '%s' promovida de con_extra → ítem propio",
                        nombre_extra,
                    )
                else:
                    # Ya existe como ítem propio → solo eliminar de con_extra (no duplicar)
                    logger.info(
                        "[NLPExtractor] Bebida '%s' eliminada de con_extra (ya es ítem propio)",
                        nombre_extra,
                    )
            elif extra_norm:
                # Mantener extras no-bebida como strings simples
                extras_filtrados.append(nombre_extra if isinstance(extra, dict) else extra)
        item_copia = dict(item)
        item_copia["con_extra"] = extras_filtrados
        resultado.append(item_copia)

    return resultado + bebidas_nuevas


def _norm(texto: str) -> str:
    """Normaliza texto: minúsculas, sin tildes, sin caracteres raros."""
    s = (texto or "").strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    return re.sub(r"\s{2,}", " ", s).strip()


class NLPFoodExtractor:
    """
    Extrae alimentos de texto libre con Llama-3 como parser
    y cálculo determinista desde la BD.
    """

    def __init__(self, ia_service, db: Session):
        self.ia = ia_service
        self.db = db

    # ─── PRE-NORMALIZACIÓN del texto antes del LLM ───────────────────────────
    def _pre_normalizar(self, texto: str) -> str:
        """Normaliza conectores temporales, redundancias y marcas comerciales
        para reducir varianza del LLM antes de enviar al extractor JSON.
        También aplica normalización de voz: muletillas + números hablados."""
        t = _normalizar_voz_comida(texto)  # muletillas + "setenta"→70
        for patron, reemplazo in PRE_NORM_PATRONES:
            t = re.sub(patron, reemplazo, t)
        # Marcas comerciales de gaseosas → nombre genérico "gaseosa"
        # Ej: "tome gaseosa inkacola" → "tome gaseosa gaseosa" → "tome gaseosa"
        # Ej: "tome inca kola"        → "tome gaseosa"
        t = _MARCAS_GASEOSAS_RE.sub("gaseosa", t)
        # Deduplicar "gaseosa gaseosa" → "gaseosa" (ocurre cuando el texto tenía
        # "gaseosa MARCA" y la marca fue reemplazada por "gaseosa")
        t = re.sub(r"(?i)\bgaseosa\s+gaseosa\b", "gaseosa", t)
        return t.strip()

    def _es_negacion(self, texto: str) -> bool:
        """Detecta si el mensaje indica que el usuario NO comio algo."""
        for patron in NEGACION_PATRONES:
            if re.search(patron, texto):
                return True
        return False

    # ─── PASO 1: Llama-3 extrae el JSON ───────────────────────────────────────
    async def _llm_extraer_json(self, mensaje: str) -> list[dict]:
        """Llama a Llama-3 con prompt estricto y parsea el JSON resultante."""
        texto_norm = self._pre_normalizar(mensaje)
        prompt = EXTRACTION_PROMPT.replace("{texto}", texto_norm)
        try:
            respuesta = await self.ia._llamar_groq(prompt=prompt, max_tokens=500, temp=0.05)
            # Limpiar posible markdown del LLM
            respuesta = re.sub(r"```(?:json)?", "", respuesta).strip().strip("`")
            # Extraer solo el array JSON
            m = re.search(r"\[.*\]", respuesta, re.DOTALL)
            if not m:
                return []
            items = json.loads(m.group(0))
            items = self._normalizar_cantidad_en_nombre(items)
            return self._fusionar_item_duplicado(items)
        except Exception as e:
            print(f"[NLPExtractor] Error parsing LLM JSON: {e}")
            return []

    @staticmethod
    def _fusionar_item_duplicado(items: list[dict]) -> list[dict]:
        """
        1) Fusiona cuando el LLM divide "X gramos de [plato]" en dos objetos:
           [{alimento:"pollo saltado", unidad:"porcion"}, {alimento:"pollo", unidad:"g"}]
           → [{alimento:"pollo saltado", unidad:"g"}]
           Condición: ítem B es substring del nombre de ítem A Y tiene unidad de peso.

        2) Deduplica cuando el LLM extrae el mismo alimento dos veces con distintas
           unidades (p.ej. "smoothie X" como vaso Y como 300ml).
           Se conserva el ítem con unidad más precisa (g/ml > vaso/porcion).
        """
        if len(items) < 2:
            return items
        _PESOS = {"g", "gr", "gramo", "gramos", "kg", "kilo", "kilogramo", "ml", "cc", "l", "litro"}
        resultado = list(items)
        fusionados: set[int] = set()

        # ── Paso 1: fusión substring (lógica original) ────────────────────────
        for i, a in enumerate(items):
            if i in fusionados:
                continue
            nombre_a = _norm(str(a.get("alimento", "")))
            for j, b in enumerate(items):
                if j <= i or j in fusionados:
                    continue
                nombre_b = _norm(str(b.get("alimento", "")))
                unidad_b = str(b.get("unidad", "")).lower()
                if unidad_b in _PESOS and nombre_b and nombre_b in nombre_a and nombre_b != nombre_a:
                    resultado[i] = {**a, "cantidad": b.get("cantidad", a.get("cantidad")), "unidad": unidad_b}
                    fusionados.add(j)
                    break

        items_paso1 = [item for k, item in enumerate(resultado) if k not in fusionados]

        # ── Paso 2: dedup mismo nombre (nuevo) ───────────────────────────────
        # Cuando el LLM genera duplicados del mismo alimento con distintas unidades,
        # conservamos el que tiene unidad más precisa (g/ml antes que vaso/porcion).
        _PESOS_PRECISION = frozenset({"g", "gr", "gramo", "gramos", "ml", "cc"})
        seen: dict[str, int] = {}  # nombre_norm → índice en `final`
        final: list[dict] = []
        for item in items_paso1:
            k = _norm(str(item.get("alimento", "")))
            unit = str(item.get("unidad", "")).lower()
            if k not in seen:
                seen[k] = len(final)
                final.append(item)
            else:
                # Reemplazar si el nuevo ítem tiene unidad más precisa
                prev = final[seen[k]]
                prev_unit = str(prev.get("unidad", "")).lower()
                if unit in _PESOS_PRECISION and prev_unit not in _PESOS_PRECISION:
                    final[seen[k]] = item
        return final

    # Mapeo de palabras numéricas a float (para normalizar "dos peras" → pera×2)
    _NUMEROS_ES: dict[str, float] = {
        "un": 1, "una": 1, "uno": 1,
        "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5,
        "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10,
    }

    @staticmethod
    def _normalizar_cantidad_en_nombre(items: list[dict]) -> list[dict]:
        """
        Detecta cuando el LLM incluyó el número en el nombre del alimento
        (ej. {alimento:"dos peras", cantidad:1}) y lo separa correctamente
        → {alimento:"pera", cantidad:2, unidad:"unidad"}.
        """
        _NUM = NLPFoodExtractor._NUMEROS_ES
        resultado = []
        for item in items:
            nombre = str(item.get("alimento", "")).strip().lower()
            partes = nombre.split(maxsplit=1)
            if len(partes) == 2 and partes[0] in _NUM:
                num_val = _NUM[partes[0]]
                # Singularizar el alimento (quitar -s o -as al final si corresponde)
                food = partes[1]
                if food.endswith("as") and len(food) > 4:
                    food = food[:-1]   # "peras" → "pera", "manzanas" → "manzana"
                elif food.endswith("es") and len(food) > 4:
                    food = food[:-2]   # "melones" → "melon"
                elif food.endswith("s") and not food.endswith("ss") and len(food) > 3:
                    food = food[:-1]   # "huevos" → "huevo"
                resultado.append({
                    **item,
                    "alimento": food,
                    "cantidad": float(item.get("cantidad", 1)) * num_val,
                    "unidad": item.get("unidad") if item.get("unidad") != "porcion" else "unidad",
                })
            else:
                resultado.append(item)
        return resultado

    # ─── PASO 2: Buscar alimento en BD por nombre o alias ─────────────────────
    def _buscar_alimento_bd(self, nombre: str) -> Optional[Alimento]:
        n = _norm(nombre)
        if not n:
            return None
        # 1. Búsqueda exacta por nombre normalizado
        a = self.db.query(Alimento).filter(Alimento.nombre_normalizado == n).first()
        if a:
            return a
        # 2. Búsqueda por alias exacto
        alias = self.db.query(AlimentoAlias).filter(AlimentoAlias.alias_normalizado == n).first()
        if alias:
            return self.db.query(Alimento).filter(Alimento.id == alias.alimento_id).first()
        # 3. Búsqueda por palabra inicial exacta: "papa" NO debe coincidir con "papaya"
        #    Usamos LIKE "papa %" para evitar falsos positivos.
        #    REGLA 6: ORDER BY LENGTH ASC → preferir el nombre más corto/específico.
        #    Evita que "fruta" matchee "Ensalada de Frutas" (381 kcal) en lugar de
        #    "Fruta Seca" o el alimento individual correcto.
        from sqlalchemy import func as _sqlfunc
        a3 = (
            self.db.query(Alimento)
            .filter(Alimento.nombre_normalizado.like(f"{n} %"))
            .order_by(_sqlfunc.length(Alimento.nombre_normalizado).asc())
            .first()
        )
        if a3:
            return a3
        # 4. Bug 3 fix: cuando el nombre tiene ≥2 palabras con "de" (ej: "galleta de avena"),
        #    priorizar el match que contenga TODAS las palabras clave, no solo la primera.
        #    Esto evita que "galleta de avena" matchee "Galleta De Soda" por LIKE "%galleta%".
        palabras_n = [w for w in n.split() if len(w) >= 4 and w not in {"de", "con", "del", "las", "los"}]
        if len(palabras_n) >= 2:
            # Requiere que TODAS las palabras clave estén en el nombre normalizado de BD
            query_multi = self.db.query(Alimento)
            for pw in palabras_n:
                query_multi = query_multi.filter(Alimento.nombre_normalizado.like(f"%{pw}%"))
            a_multi = (
                query_multi
                .order_by(_sqlfunc.length(Alimento.nombre_normalizado).asc())
                .first()
            )
            if a_multi:
                return a_multi
        # 5. Fallback amplio: el nombre buscado aparece en cualquier posición.
        #    REGLA 6: ORDER BY LENGTH ASC garantiza que "Fruta" (genérico corto) tenga
        #    prioridad sobre "Ensalada de Frutas" o "Mezcla de Frutas" (compuestos largos).
        a5 = (
            self.db.query(Alimento)
            .filter(Alimento.nombre_normalizado.like(f"%{n}%"))
            .order_by(
                Alimento.nombre_normalizado.like(f"{n}%").desc(),
                _sqlfunc.length(Alimento.nombre_normalizado).asc(),
            )
            .first()
        )
        return a5

    # ─── PASO 3b: Aplicar modificadores sin/con_extra ─────────────────────────
    def _gramos_tipicos_ingrediente(self, nombre_ingrediente: str, nombre_plato: str) -> float:
        """
        Estima los gramos típicos de un ingrediente dentro de un plato.
        Usa la tabla alimento_unidades si existe, sino usa valores estándar.
        """
        GRAMOS_TIPICOS = {
            "arroz":     186.0,   # 1 taza cocida
            "papa":      150.0,   # 1 papa mediana
            "pan":        50.0,   # 1 pan frances
            "camote":    130.0,
            "yuca":      100.0,
            "fideos":    140.0,
            "platano":   118.0,
            "galleta":    15.0,   # 1 galleta de soda/vainilla (~15g)
            "galletas":   15.0,
        }
        n = _norm(nombre_ingrediente)
        tokens = set(n.split())
        for kw, gramos in GRAMOS_TIPICOS.items():
            if kw in tokens:
                return gramos
        return 100.0  # default: 1 porción

    def _aplicar_modificadores(
        self,
        calorias: float, proteinas: float, carbos: float, grasas: float,
        sin_lista: list[str], con_extra_lista: list[str],
        nombre_plato: str,
    ) -> tuple[float, float, float, float, list[str]]:
        """
        Ajusta los macros de un plato según modificadores sin/con_extra.
        Retorna (kcal, prot, carbs, grasas, notas).
        """
        notas = []

        # ── Restar ingredientes excluidos ("sin arroz") ──────────────────────
        for ingrediente in (sin_lista or []):
            alim = self._buscar_alimento_bd(ingrediente)
            if alim:
                gramos = self._gramos_tipicos_ingrediente(ingrediente, nombre_plato)
                factor = gramos / 100.0
                kcal_resta  = round(float(alim.calorias_100g)      * factor, 1)
                prot_resta  = round(float(alim.proteina_100g)      * factor, 1)
                carbs_resta = round(float(alim.carbohidratos_100g) * factor, 1)
                grasas_resta= round(float(alim.grasas_100g)        * factor, 1)
                calorias  = max(0, calorias  - kcal_resta)
                proteinas = max(0, proteinas - prot_resta)
                carbos    = max(0, carbos    - carbs_resta)
                grasas    = max(0, grasas    - grasas_resta)
                notas.append(f"sin {alim.nombre} (-{kcal_resta:.0f} kcal)")
            else:
                notas.append(f"no encontre '{ingrediente}' para restar")

        # ── Sumar ingredientes extra ("con extra queso") ─────────────────────
        for ingrediente in (con_extra_lista or []):
            alim = self._buscar_alimento_bd(ingrediente)
            if alim:
                gramos = self._gramos_tipicos_ingrediente(ingrediente, nombre_plato)
                factor = gramos / 100.0
                calorias  += round(float(alim.calorias_100g)      * factor, 1)
                proteinas += round(float(alim.proteina_100g)      * factor, 1)
                carbos    += round(float(alim.carbohidratos_100g) * factor, 1)
                grasas    += round(float(alim.grasas_100g)        * factor, 1)
                notas.append(f"con extra {alim.nombre} (+{round(float(alim.calorias_100g)*factor, 0):.0f} kcal)")

        return round(calorias, 1), round(proteinas, 1), round(carbos, 1), round(grasas, 1), notas

    # ─── PASO 3: Obtener gramos para la unidad indicada ───────────────────────
    def _resolver_gramos(self, alimento: Alimento, unidad: str, cantidad: float) -> float:
        u = _norm(unidad)

        # Para unidades de peso/volumen universales (g, kg, ml…) ir directo a la
        # tabla UNIDADES_GLOBALES — nunca buscar en alimento_unidades porque
        # ilike('%g%') matchea strings como "unidad grande" causando ×60 erróneo.
        _PESOS_UNIVERSALES = {
            "g", "gr", "gramo", "gramos",
            "kg", "kilo", "kilogramo", "kilogramos",
            "ml", "cc", "l", "litro", "litros",
        }
        if u in _PESOS_UNIVERSALES:
            return UNIDADES_GLOBALES.get(u, 1.0) * cantidad

        # 1. Buscar en alimento_unidades de la BD (porciones específicas: rebanada, taza…)
        row = (
            self.db.query(AlimentoUnidad)
            .filter(
                AlimentoUnidad.alimento_id == alimento.id,
                AlimentoUnidad.nombre.ilike(f"%{u}%"),
            )
            .first()
        )
        if row and row.gramos and row.gramos > 0:
            return float(row.gramos) * cantidad

        # 2. Peso específico por alimento para "unidad"
        if u == "unidad":
            nombre_lower = (alimento.nombre or "").lower()
            for clave, peso in PESOS_UNIDAD.items():
                if clave in nombre_lower:
                    return peso * cantidad

        # 3. Fallback: tabla global de unidades
        g_unit = UNIDADES_GLOBALES.get(u, 100.0)
        return g_unit * cantidad

    # ─── PASO 4: Fallback USDA cuando el alimento no está en BD ──────────────
    def _buscar_usda(self, nombre_en: str) -> Optional[dict]:
        """Busca en USDA y retorna macros por 100g.
        Incluye validación de calidad del match para evitar que términos
        inventados encuentren resultados no relacionados.
        """
        # No consultar USDA si el término es claramente no-comestible
        n_lower = _norm(nombre_en)
        for palabra in NO_ALIMENTOS:
            if palabra in n_lower.split():
                print(f"[USDA] Término bloqueado por lista negra: '{nombre_en}'")
                return None

        # No consultar si la query tiene más de 5 palabras (probablemente es una frase completa)
        if len(nombre_en.split()) > 5:
            print(f"[USDA] Query demasiado larga, omitiendo: '{nombre_en[:40]}'")
            return None

        params = urllib.parse.urlencode({
            "query": nombre_en,
            "api_key": USDA_API_KEY,
            "pageSize": 1,
            "dataType": "Foundation,SR Legacy",
        })
        url = f"https://api.nal.usda.gov/fdc/v1/foods/search?{params}"
        try:
            with urllib.request.urlopen(url, timeout=8) as resp:
                data = json.loads(resp.read().decode())
            foods = data.get("foods", [])
            if not foods:
                return None

            food = foods[0]
            # Validar calidad del match: al menos 1 palabra de la query
            # debe aparecer en la descripción del alimento USDA
            desc_usda = _norm(food.get("description", ""))
            palabras_query = set(n_lower.split()) - {"de", "con", "y", "el", "la", "un", "una"}
            match_quality = any(p in desc_usda for p in palabras_query if len(p) >= 4)
            if not match_quality:
                print(f"[USDA] Match de baja calidad para '{nombre_en}' → '{food.get('description', '')[:40]}', omitiendo")
                return None

            macros = {}
            for n in food.get("foodNutrients", []):
                nid = n.get("nutrientId")
                if nid in NUTRIENT_IDS:
                    macros[NUTRIENT_IDS[nid]] = round(n.get("value", 0), 2)
            return macros if "calorias_100g" in macros else None
        except Exception:
            return None

    # Verbos de acción que indican que el nombre es el mensaje del usuario, no un alimento.
    # Si el nombre a guardar contiene uno de estos tokens, se rechaza para evitar crear
    # alimentos con nombre "tomé un vaso de X" o "comí una porción de Y".
    _RE_NOMBRE_MENSAJE = re.compile(
        r"\b(tom[eé]|com[ií]|beb[ií]|almorcé|almorce|desayun[eé]|cen[eé]|"
        r"come\s|comes\s|estoy\s+comiendo|voy\s+a\s+comer|"
        r"registra|anota|guard[ao]|agreg[ao]|prob[eé]|me\s+com[ií])\b",
        re.IGNORECASE,
    )

    def _guardar_en_bd(
        self,
        nombre_es: str,
        macros: dict,
        fuente: str = "USDA (auto-aprendido)",
    ) -> Optional[Alimento]:
        """Guarda un alimento nuevo en BD. REGLA 4: siempre propaga fuente y flags de confianza."""
        try:
            # Guard: rechazar nombres que son el mensaje del usuario (contienen verbos de acción).
            # Previene alimentos como 'tomé un vaso de leche' creados cuando el texto
            # completo del mensaje llega como nombre al pipeline de estimación.
            if self._RE_NOMBRE_MENSAJE.search(nombre_es) or len(nombre_es.split()) > 6:
                logger.warning(
                    "Alimento '%s' rechazado — nombre parece un mensaje de usuario, no un alimento",
                    nombre_es[:80],
                )
                return None
            _es_estimado = "groq" in fuente.lower() or "estimado" in fuente.lower()
            a = Alimento(
                nombre=nombre_es,
                nombre_normalizado=_norm(nombre_es),
                calorias_100g=macros.get("calorias_100g", 0),
                proteina_100g=macros.get("proteina_100g", 0),
                carbohidratos_100g=macros.get("carbohidratos_100g", 0),
                grasas_100g=macros.get("grasas_100g", 0),
                fibra_100g=macros.get("fibra_100g", 0),
                azucar_100g=macros.get("azucar_100g", 0),
                categoria="Otros",
                fuente=fuente,
                es_confiable=not _es_estimado,
                pendiente_validacion=_es_estimado,
            )
            _ok, _motivo = validar_macros_atwater(
                a.calorias_100g, a.proteina_100g, a.carbohidratos_100g, a.grasas_100g
            )
            if not _ok:
                logger.warning("Alimento '%s' descartado — %s", nombre_es, _motivo)
                return None
            self.db.add(a)
            self.db.flush()   # Fix 2: flush mantiene la transacción del caller intacta
            logger.info("Nuevo alimento '%s' (fuente=%s)", nombre_es, fuente)
            return a
        except Exception as e:
            self.db.rollback()
            logger.error("Error guardando '%s': %s", nombre_es, e)
            return None

    # ─── PASO 4b: Fallback Groq cuando BD y USDA fallan ─────────────────────
    async def _groq_estimar_macros(self, nombre: str) -> Optional[dict]:
        """
        Última instancia: le pide a Groq macros estimados por 100g para el alimento.
        Solo se usa cuando ni la BD ni USDA tienen datos.
        Devuelve dict con claves calorias_100g, proteina_100g, carbohidratos_100g, grasas_100g
        o None si el modelo no puede estimarlo.
        """
        prompt = (
            f"Eres un nutricionista experto. Necesito los macronutrientes por 100g del siguiente alimento peruano: "
            f"'{nombre}'.\n"
            f"Responde SOLO con un JSON válido, sin texto adicional:\n"
            f'{{"calorias_100g": <número>, "proteina_100g": <número>, "carbohidratos_100g": <número>, "grasas_100g": <número>}}\n'
            f"Usa valores reales y realistas. Si es un plato completo, estima por 100g de porción servida."
        )
        try:
            respuesta = await self.ia._llamar_groq(prompt=prompt, max_tokens=120, temp=0.1)
            respuesta = re.sub(r"```(?:json)?", "", respuesta).strip().strip("`")
            m = re.search(r"\{.*\}", respuesta, re.DOTALL)
            if not m:
                return None
            data = json.loads(m.group(0))
            if "calorias_100g" not in data or float(data.get("calorias_100g", 0)) <= 0:
                return None
            return {
                "calorias_100g":      round(float(data.get("calorias_100g", 0)), 1),
                "proteina_100g":      round(float(data.get("proteina_100g", 0)), 1),
                "carbohidratos_100g": round(float(data.get("carbohidratos_100g", 0)), 1),
                "grasas_100g":        round(float(data.get("grasas_100g", 0)), 1),
                "fibra_100g": 0,
                "azucar_100g": 0,
            }
        except Exception as e:
            print(f"[NLPExtractor] Groq fallback falló para '{nombre}': {e}")
            return None

    # Porción estándar en gramos por tipo de ingrediente en contexto de plato
    _PORCIONES_COMPONENTE: dict[str, float] = {
        # Proteínas
        "pollo":    200.0, "pechuga":  180.0, "muslo":    180.0,
        # Nota: "res" se comprueba por token exacto (evita match en "fresco")
        "pescado":  180.0, "salmon":   180.0, "atun":     100.0,
        "carne":    180.0, "res":       180.0, "cerdo":    180.0,
        "huevo":     50.0, "huevos":   100.0, "jamon":     60.0,
        "queso":     40.0, "tofu":     150.0,
        # Carbohidratos
        "arroz":    180.0, "papa":     150.0, "camote":   130.0,
        "fideos":   180.0, "tagliatelle": 180.0, "pasta": 180.0,
        "pan":       60.0, "quinua":   120.0, "yuca":     150.0,
        "tostada":   30.0, "tostadas":  30.0,   # 1 rebanada tostada
        "choclo":   120.0, "avena":     80.0, "granola":   40.0,
        # Frutas
        "platano":  120.0, "manzana":  150.0, "naranja":  140.0,
        "fresa":     80.0, "mango":    120.0, "pera":     150.0,
        "uva":       80.0, "piña":     100.0, "papaya":   120.0,
        # Verduras/Ensalada
        "lechuga":   80.0, "tomate":    80.0, "pepino":    80.0,
        "zanahoria": 80.0, "brocoli":  100.0, "espinaca":  80.0,
        "verduras": 100.0, "ensalada": 100.0, "cebolla":   50.0,
        # Lácteos
        "leche":    240.0, "yogur":    150.0, "yogurt":   150.0,
        # Grasas/Condimentos
        "aceite":     8.0, "mantequilla": 10.0, "palta":   60.0,
        "aguacate":  60.0, "nueces":    20.0, "almendras": 20.0,
    }

    def _porcion_componente(self, nombre_componente: str) -> float:
        """Devuelve gramos de porción estándar para un componente dado su nombre normalizado."""
        n = _norm(nombre_componente)
        tokens = set(n.split())  # word-level: evita "res" en "fresco", "pan" en "plátano", etc.
        for clave, gramos in self._PORCIONES_COMPONENTE.items():
            if clave in tokens:
                return gramos
        return 150.0  # default genérico

    async def _resolver_componente_async(self, parte: str) -> Optional[Alimento]:
        """
        Busca UN componente de plato con flujo en 3 pasos (sin fallback genérico):
        1. alimentos BD — nombre COMPLETO (ej: "pollo al horno", resuelto por alias)
        2. USDA API  — con el nombre completo y específico del componente
        3. Groq      — estima macros por 100g para ese componente específico
                       y lo guarda en BD para no volver a consultar

        NO usa fallback a base genérica (ej: "pollo" si no encuentra "pollo al horno")
        porque los macros del genérico pueden diferir significativamente del plato real.
        NO busca en platos para componentes individuales porque causaría
        que "pollo" matchee "pollo con arroz" (plato completo, macros incorrectos).
        """
        q = _norm(parte)
        if not q:
            return None

        # ── Paso 1: BD local con nombre completo (alias incluidos) ───────────
        alim = self._buscar_alimento_bd(parte)
        if alim:
            return alim

        # ── Paso 2: USDA con el nombre COMPLETO y específico ─────────────────
        macros_usda = self._buscar_usda(parte)
        if macros_usda:
            guardado = self._guardar_en_bd(parte, macros_usda, "USDA (auto-aprendido)")
            if guardado:
                return guardado

        # ── Paso 3: Groq estima con nombre específico y guarda en BD ─────────
        # REGLA 4: fuente "Groq (estimado)" → es_confiable=False, pendiente_validacion=True
        macros_groq = await self._groq_estimar_macros(parte)
        if macros_groq:
            guardado = self._guardar_en_bd(parte, macros_groq, "Groq (estimado)")
            if guardado:
                return guardado

        return None

    async def _calcular_macros_plato_combinado(self, nombre: str, cantidad: float,
                                                unidad: str = "porcion",
                                                gramos_usuario: Optional[float] = None):
        """
        Algoritmo general para platos compuestos "X con/y Y [con/y Z...]":

        1. Divide el nombre en componentes por "con" y "y"
        2. Por cada componente: BD completo → USDA (nombre específico) → Groq → guarda en BD
           (SIN fallback genérico: si "pollo al horno" no existe, busca exactamente eso,
            no se degrada a "pollo" porque los macros serían distintos)
        3. Asigna porción estándar en gramos por tipo de ingrediente
           (o distribuye gramos proporcionalmente si el usuario los especificó, ej: "300g")
        4. Suma todos los macros escalados por `cantidad` (ej: 0.5 = media porción)
           o por factor = gramos_usuario / gramos_estándar_totales

        Devuelve (calorias, proteinas, carbos, grasas, gramos_totales) o None.
        """
        # Solo aplica a nombres con conectores de composición
        if not re.search(r'\s+(?:con|y)\s+', nombre):
            return None

        # ── Dividir en componentes por "con" y "y" ───────────────────────────
        # "pollo al horno con platano y arroz" → ["pollo al horno", "platano", "arroz"]
        # "arroz con leche" → ["arroz", "leche"] (legítimo: 2 componentes)
        partes = [p.strip() for p in re.split(r'\s+(?:con|y)\s+', nombre) if p.strip()]
        if len(partes) < 2:
            return None

        # ── Resolver cada componente ──────────────────────────────────────────
        componentes: list[tuple[Alimento, float]] = []  # (alimento, gramos_porcion)
        for parte in partes:
            alim = await self._resolver_componente_async(parte)
            if not alim:
                continue
            gramos_std = self._porcion_componente(alim.nombre_normalizado or alim.nombre or "")
            componentes.append((alim, gramos_std))

        if not componentes:
            return None

        # ── Determinar escala según unidad/cantidad del usuario ───────────────
        gramos_std_total = sum(g for _, g in componentes)

        _UNIDADES_MASA_O_VOL = {"g", "gr", "gramo", "gramos", "ml", "cc"}
        if unidad in _UNIDADES_MASA_O_VOL and gramos_usuario and gramos_usuario > 0:
            # Usuario especificó gramos/ml → distribuir proporcionalmente entre componentes.
            # Ej: "300g pollo al horno con plátano" → escala = 300 / gramos_std_total
            # Ej: "300ml smoothie de kale con jengibre" → igual (ml ≈ g para bebidas)
            escala = gramos_usuario / gramos_std_total
            gramos_finales = gramos_usuario
        else:
            # Escala por cantidad (1.0 = porción completa, 0.5 = media)
            escala = cantidad
            gramos_finales = gramos_std_total * cantidad

        # ── Sumar macros ──────────────────────────────────────────────────────
        total_cal = total_prot = total_carb = total_gras = 0.0
        for alim, gramos_std in componentes:
            gramos = gramos_std * escala
            factor = gramos / 100.0
            kcal_c = round(float(alim.calorias_100g or 0) * gramos / 100, 1)
            total_cal  += float(alim.calorias_100g or 0) * factor
            total_prot += float(alim.proteina_100g or 0) * factor
            total_carb += float(alim.carbohidratos_100g or 0) * factor
            total_gras += float(alim.grasas_100g or 0) * factor
            print(f"  · '{alim.nombre}' {round(gramos,0)}g → {kcal_c} kcal")

        if total_cal <= 0:
            return None

        print(f"[NLPExtractor] '{nombre}' × escala={round(escala,2)}: "
              f"{round(total_cal,1)} kcal ({len(componentes)}/{len(partes)} comp.)")
        return (round(total_cal, 1), round(total_prot, 1),
                round(total_carb, 1), round(total_gras, 1), round(gramos_finales, 1))

    def _buscar_ingrediente_base(self, nombre: str) -> Optional[Alimento]:
        """
        Extrae el ingrediente principal del nombre de un plato compuesto y lo busca en BD.
        Ej: 'sopa de lentejas ligera' → busca 'lentejas'
            'tortilla de huevo con pan' → busca 'huevo'
        """
        palabras_clave = [
            w for w in nombre.split()
            if len(w) >= 5 and w not in {
                "sopa", "crema", "guiso", "estofado", "ensalada", "sandwich",
                "tostada", "tortilla", "ligera", "fresca", "cocida", "asado",
                "frito", "griego", "natural", "integral", "entero", "pequeño",
                "grande", "porcion", "plato", "con", "sin", "para", "del"
            }
        ]
        for palabra in palabras_clave:
            resultado = self._buscar_alimento_bd(palabra)
            if resultado:
                print(f"[NLPExtractor] Ingrediente base '{palabra}' encontrado para '{nombre}'")
                return resultado
        return None

    # ─── MÉTODO PRINCIPAL ─────────────────────────────────────────────────────
    async def extraer(self, mensaje: str) -> Optional[ResultadoExtraccion]:
        """
        Extrae alimentos del mensaje y calcula macros.
        Soporta modificadores: 'sin arroz', 'con extra queso', etc.
        Retorna None si no se detectó ningún alimento.
        """
        # GUARD: Detectar negaciones antes de procesar
        if self._es_negacion(mensaje):
            logger.info("[NLPExtractor] Negacion detectada, sin registro: '%s'", mensaje[:50])
            return None

        # PASO 0: Pre-check catálogo platos con mensaje completo (antes de Groq).
        # Evita que Groq renombre platos (ej: "tortilla de huevo" → "huevo entero cocido").
        if self.db:
            _verbos_log = (
                "comi ", "comí ", "almorcé ", "almorce ", "desayuné ", "desayune ",
                "cené ", "cene ", "tomé ", "tome ", "bebí ", "bebi ", "meriendé ",
                "come ", "comes ", "estoy comiendo ", "voy a comer ",
                "registra que comi ", "registra que comí ", "registra que almorce ",
                "registra que almorcé ", "registra que desayune ", "registra que desayuné ",
                "registra que tome ", "registra que tomé ", "registra que ",
            )
            _msg_clean = mensaje.lower().strip()
            for _v in _verbos_log:
                if _msg_clean.startswith(_v):
                    _msg_clean = _msg_clean[len(_v):].strip()
                    break
            # quitar sufijos de contexto
            for _sf in (" en el almuerzo", " en el desayuno", " en la cena",
                        " al almuerzo", " al desayuno", " a la cena"):
                if _msg_clean.endswith(_sf):
                    _msg_clean = _msg_clean[:-len(_sf)].strip()
            # ── Extraer cantidad numérica antes del fuzzy ("dos", "tres", etc.) ─────
            _qty_pre = 1.0
            # Fracciones: "un cuarto de X" → strip "un cuarto de " antes del fuzzy
            _m_fraccion_pre = re.match(
                r"(?i)^(?:un\s+)?(?:cuarto|octavo)\s+(?:de\s+)?(.+)$", _msg_clean
            )
            if _m_fraccion_pre:
                _qty_pre = 0.25 if re.search(r'\bcuarto\b', _msg_clean, re.IGNORECASE) else 0.125
                _msg_clean = _m_fraccion_pre.group(1).strip()
            else:
                _m_qty_pre = re.match(
                    r"(?i)^(dos|tres|cuatro|cinco|2|3|4|5|medio|media)\s+(.+)$",
                    _msg_clean,
                )
                if _m_qty_pre:
                    _qty_map_pre = {
                        "dos": 2.0, "tres": 3.0, "cuatro": 4.0, "cinco": 5.0,
                        "2": 2.0, "3": 3.0, "4": 4.0, "5": 5.0,
                        "medio": 0.5, "media": 0.5,
                    }
                    _qty_pre = _qty_map_pre.get(_m_qty_pre.group(1).lower(), 1.0)
                    _msg_clean = _m_qty_pre.group(2).strip()
            # ──────────────────────────────────────────────────────────────────────
            _q_pre = _norm(_msg_clean)
            if _q_pre and len(_q_pre) >= 5:
                try:
                    from sqlalchemy import text as _text0
                    from app.models.plato import Plato as _Plato0
                    import difflib as _diff0
                    # Exact match first
                    _pre_row = self.db.execute(_text0(
                        "SELECT p.id, p.nombre,"
                        " SUM(a.calorias_100g*pi2.gramos/100.0),"
                        " SUM(a.proteina_100g*pi2.gramos/100.0),"
                        " SUM(a.carbohidratos_100g*pi2.gramos/100.0),"
                        " SUM(a.grasas_100g*pi2.gramos/100.0),"
                        " SUM(pi2.gramos)"
                        " FROM platos p"
                        " JOIN plato_ingredientes pi2 ON pi2.plato_id=p.id"
                        " JOIN alimentos a ON a.id=pi2.alimento_id"
                        " WHERE p.nombre_normalizado=:q GROUP BY p.id,p.nombre LIMIT 1"
                    ), {"q": _q_pre}).fetchone()
                    # Similarity fallback (≥0.88) if no exact match
                    if not _pre_row:
                        _cands0 = (
                            self.db.query(_Plato0.id, _Plato0.nombre_normalizado)
                            .order_by(_Plato0.id.desc()).limit(300).all()
                        )
                        _bid0, _bsc0, _bpnn0 = None, 0.0, ""
                        for _pid0, _pnn0 in _cands0:
                            if not _pnn0:
                                continue
                            if not coherencia_proteina_platos(_q_pre, str(_pnn0)):
                                continue
                            _sc0 = _diff0.SequenceMatcher(a=_q_pre, b=str(_pnn0)).ratio()
                            if _sc0 > _bsc0:
                                _bsc0 = _sc0
                                _bid0 = _pid0
                                _bpnn0 = str(_pnn0)
                        if _bid0 and _bsc0 >= 0.88 and _sufijos_con_compat(_q_pre, _bpnn0):
                            _pre_row = self.db.execute(_text0(
                                "SELECT p.id, p.nombre,"
                                " SUM(a.calorias_100g*pi2.gramos/100.0),"
                                " SUM(a.proteina_100g*pi2.gramos/100.0),"
                                " SUM(a.carbohidratos_100g*pi2.gramos/100.0),"
                                " SUM(a.grasas_100g*pi2.gramos/100.0),"
                                " SUM(pi2.gramos)"
                                " FROM platos p"
                                " JOIN plato_ingredientes pi2 ON pi2.plato_id=p.id"
                                " JOIN alimentos a ON a.id=pi2.alimento_id"
                                " WHERE p.id=:pid GROUP BY p.id,p.nombre LIMIT 1"
                            ), {"pid": _bid0}).fetchone()
                            if _pre_row:
                                logger.info(
                                    "[NLPExtractor] Pre-check similarity %.2f: '%s' → '%s'",
                                    _bsc0, _q_pre, _pre_row[1],
                                )
                    if _pre_row:
                        _kcal0 = round(float(_pre_row[2] or 0), 1)
                        _prot0 = round(float(_pre_row[3] or 0), 1)
                        _carb0 = round(float(_pre_row[4] or 0), 1)
                        _gras0 = round(float(_pre_row[5] or 0), 1)
                        _g0    = float(_pre_row[6] or 300.0)
                        logger.info(
                            "[NLPExtractor] Pre-check: '%s' qty=%.0f %.1f kcal",
                            _pre_row[1], _qty_pre, _kcal0,
                        )
                        _item0 = ItemExtraido(
                            alimento=str(_pre_row[1]),
                            cantidad=_qty_pre,
                            unidad="porcion",
                            gramos_totales=round(_g0 * _qty_pre, 1),
                            calorias=round(_kcal0 * _qty_pre, 1),
                            proteinas_g=round(_prot0 * _qty_pre, 1),
                            carbohidratos_g=round(_carb0 * _qty_pre, 1),
                            grasas_g=round(_gras0 * _qty_pre, 1),
                            origen="bd",
                        )
                        return ResultadoExtraccion(
                            items=[_item0],
                            nombres=[str(_pre_row[1])],
                            calorias_total=round(_kcal0 * _qty_pre, 1),
                            proteinas_total=round(_prot0 * _qty_pre, 1),
                            carbohidratos_total=round(_carb0 * _qty_pre, 1),
                            grasas_total=round(_gras0 * _qty_pre, 1),
                            advertencia=None,
                        )
                except Exception as _ep:
                    logger.warning("[NLPExtractor] Pre-check error: %s", _ep)

        # PASO 1: Llama-3 extrae JSON
        items_raw = await self._llm_extraer_json(mensaje)
        logger.info("[NLPExtractor] LLM extrajo %d items: %s", len(items_raw or []), items_raw)
        if not items_raw:
            # FALLBACK: el LLM no reconoció ningún alimento (ej. "gomitas", "caramelos").
            # Intentar extraer el nombre directamente del mensaje con regex y enviarlo a
            # Groq para estimar macros — permite registrar alimentos reales pero poco comunes.
            _m_fb = (mensaje or "").lower().strip()
            # Quitar verbo de ingesta al inicio
            _m_fb = re.sub(
                r"^(com[ií]|tom[eé]|beb[ií]|almorcé|almorce|desayun[eé]|cen[eé]|"
                r"registra?\s+(?:que\s+)?(?:com[ií]\s+)?|probé|probe)\s+",
                "", _m_fb, flags=re.IGNORECASE,
            ).strip()
            # Quitar artículos iniciales
            _m_fb = re.sub(r"^(un[ao]?s?\s+|unas?\s+|algo\s+de\s+)", "", _m_fb).strip()
            if _m_fb and 2 <= len(_m_fb) <= 40 and not _nombre_es_no_alimento(_m_fb):
                # Separar "N X" donde N es un dígito: "2 caramelos" → alimento="caramelos", cantidad=2
                _qty_fb = 1.0
                _digit_m = re.match(r"^(\d+(?:[.,]\d+)?)\s+(.+)$", _m_fb)
                if _digit_m:
                    _qty_fb = float(_digit_m.group(1).replace(",", "."))
                    _m_fb   = _digit_m.group(2).strip()
                if not _m_fb or _nombre_es_no_alimento(_m_fb):
                    return None
                # Para piezas discretas (candy, galleta, etc.) usar gramos directamente
                # para evitar que "2 caramelos" = 2 × 100g = 200g (demasiado).
                # Estimación razonable: ~10g por pieza, max 100g total.
                _gramos_fb = min(_qty_fb * 10.0, 100.0)
                logger.info("[NLPExtractor] Fallback regex → '%s' x%.0f (~%.0fg) para Groq", _m_fb, _qty_fb, _gramos_fb)
                items_raw = [{"alimento": _m_fb, "cantidad": _gramos_fb, "unidad": "g",
                              "sin": [], "con_extra": []}]
            else:
                return None
        # Post-proc: "avena con 2 panes con mermelada" → ["avena", {pan con mermelada, qty:2}]
        items_raw = _separar_con_n_items(items_raw)

        # ── POST-PROCESO: promover bebidas que siguen en con_extra a items propios ─────
        # Llama-3 a veces ignora la regla BEBIDAS_INDEPENDIENTES y las pone en con_extra.
        # Este bloque Python garantiza la separación independientemente del LLM.
        _BEBIDAS_KEYWORDS = frozenset({
            "gaseosa", "jugo", "chicha", "limonada", "refresco",
            "cerveza", "agua con sabor", "bebida",
        })
        items_raw = _promover_bebidas_extras(items_raw, _BEBIDAS_KEYWORDS)

        # Granos/semillas que, en contexto de vaso/taza, son bebidas distintas.
        # Clave: nombre normalizado del grano; valor: nombre de la bebida equivalente.
        _GRANO_A_BEBIDA = {
            "cebada":                 "agua de cebada",
            "emoliente":              "emoliente de cebada",
            "avena bebida":           "avena bebida",
        }
        _UNIDADES_LIQUIDO = frozenset({"vaso", "taza", "copa", "ml", "cc", "l", "litro"})

        items_calculados: List[ItemExtraido] = []
        advertencias = []

        for item in items_raw:
            nombre         = str(item.get("alimento", "")).strip()
            nombre_input   = nombre  # nombre tal como lo extrajo el LLM (antes de resolución)
            cantidad = float(item.get("cantidad", 1.0) or 1.0)
            unidad   = str(item.get("unidad", "porcion")).strip().lower()
            # Remap: si el LLM extrajo un grano en contexto de bebida, usar la bebida
            _n_norm = _norm(nombre)
            if unidad in _UNIDADES_LIQUIDO and _n_norm in _GRANO_A_BEBIDA:
                nombre = _GRANO_A_BEBIDA[_n_norm].title()
                _n_norm = _norm(nombre)
            sin_lista = [str(x) for x in item.get("sin", []) if x]
            # con_extra puede tener strings o dicts (si el LLM usó el schema completo)
            _raw_extras = item.get("con_extra") or []
            con_extra_lista = []
            for _x in _raw_extras:
                if isinstance(_x, dict):
                    _nm = str(_x.get("alimento") or _x.get("nombre") or "").strip()
                    if _nm:
                        con_extra_lista.append(_nm)
                elif _x:
                    con_extra_lista.append(str(_x).strip())
            logger.info("[NLPExtractor] → item: '%s' %.1f×%s", nombre, cantidad, unidad)

            if not nombre or cantidad <= 0:
                continue

            # Guard: rechazar nombres que son claramente no-alimentos
            if _nombre_es_no_alimento(nombre):
                logger.warning("[NLPExtractor] '%s' bloqueado: no es un alimento", nombre)
                continue

            # Guard Bug 5: bloquear animales domésticos no comestibles a nivel Python
            # Cubre el caso "carne de perro" donde el token 'perro' está en el nombre
            _tokens_nombre = set(_norm(nombre).split())
            if _tokens_nombre & _ANIMALES_NO_COMESTIBLES:
                _animal_detectado = next(t for t in _tokens_nombre if t in _ANIMALES_NO_COMESTIBLES)
                logger.warning(
                    "[NLPExtractor] Animal no comestible detectado: '%s' en '%s' — bloqueado",
                    _animal_detectado, nombre,
                )
                advertencias.append(
                    f"No es posible registrar '{nombre}': animal no apto para consumo."
                )
                continue

            # PASO 2: Buscar en catálogo platos (macros desde ingredientes reales)
            alimento_bd = None
            origen = "bd"
            calorias = proteinas = carbos = grasas = 0.0
            nombre_final = nombre

            from sqlalchemy import text as _text
            q_norm = _norm(nombre)

            # ── Bebida sin calorías: registrar como 0 kcal sin buscar en platos ──
            if q_norm in BEBIDAS_CERO_KCAL:
                items_calculados.append(ItemExtraido(
                    alimento=nombre, cantidad=cantidad, unidad=unidad,
                    gramos_totales=240.0 * cantidad,
                    calorias=0.0, proteinas_g=0.0, carbohidratos_g=0.0, grasas_g=0.0,
                    origen="regla",
                ))
                continue

            # ── Buscar en platos (catálogo con ingredientes reales) ────────────
            # Cuando la unidad ya es un peso explícito (g/kg) el usuario especificó
            # gramos exactos → escalar desde alimento, no desde plato de porción.
            # Usar platos solo cuando la unidad es "porcion", "unidad", "vaso", etc.
            _unidad_es_peso = unidad in (
                "g", "gr", "gramo", "gramos", "kg", "kilo", "kilogramo",
                "ml", "cc", "l", "litro",
            )

            # Bebidas simples en recipiente (vaso/taza) → siempre usar alimentos,
            # nunca platos. El plato_constructor puede crear platos con nombres
            # como "jugo de naranja" con gramos incorrectos (ej. 450g por porción).
            _BEBIDAS_SIMPLES_KW = frozenset({
                "jugo", "gaseosa", "chicha", "limonada", "refresco", "cerveza",
                "agua", "leche", "bebida", "smoothie", "batido", "nectar", "infusion",
            })
            _UNIDADES_RECIPIENTE = frozenset({"vaso", "taza", "copa"})
            _es_bebida_en_recipiente = (
                unidad in _UNIDADES_RECIPIENTE
                and any(bk in q_norm for bk in _BEBIDAS_SIMPLES_KW)
            )

            # 1) Exact match. 2) Si falla, similarity ≥ 0.83 sobre top-300.
            _SQL_PLATO = (
                "SELECT p.id, p.nombre,"
                " SUM(a.calorias_100g      * pi2.gramos / 100.0),"
                " SUM(a.proteina_100g      * pi2.gramos / 100.0),"
                " SUM(a.carbohidratos_100g * pi2.gramos / 100.0),"
                " SUM(a.grasas_100g        * pi2.gramos / 100.0),"
                " SUM(pi2.gramos)"
                " FROM platos p"
                " JOIN plato_ingredientes pi2 ON pi2.plato_id = p.id"
                " JOIN alimentos a ON a.id = pi2.alimento_id"
                " WHERE p.nombre_normalizado = :q"
                " GROUP BY p.id, p.nombre"
                " LIMIT 1"
            )
            _skip_plato_lookup = _unidad_es_peso or _es_bebida_en_recipiente
            plato_row = None if _skip_plato_lookup else self.db.execute(_text(_SQL_PLATO), {"q": q_norm}).fetchone()

            # Fallback similaridad: si no hay exact match, buscar plato con mayor
            # similitud (≥0.83) para rescatar renombrados del LLM (ej: "huevo entero cocido"
            # cuando el usuario dijo "tortilla de huevo con pan tostado ligero").
            if not plato_row and not _skip_plato_lookup:
                import difflib as _diff
                from app.models.plato import Plato as _Plato
                _cands = (
                    self.db.query(_Plato.id, _Plato.nombre_normalizado)
                    .order_by(_Plato.id.desc())
                    .limit(300)
                    .all()
                )
                _best_id, _best_score, _best_pnn = None, 0.0, ""
                for _pid, _pnn in _cands:
                    if not _pnn:
                        continue
                    if not coherencia_proteina_platos(q_norm, str(_pnn)):
                        continue
                    _sc = _diff.SequenceMatcher(a=q_norm, b=str(_pnn)).ratio()
                    if _sc > _best_score:
                        _best_score = _sc
                        _best_id = _pid
                        _best_pnn = str(_pnn)
                if _best_id and _best_score >= 0.83 and _sufijos_con_compat(q_norm, _best_pnn):
                    plato_row = self.db.execute(_text(
                        "SELECT p.id, p.nombre,"
                        " SUM(a.calorias_100g*pi2.gramos/100.0),"
                        " SUM(a.proteina_100g*pi2.gramos/100.0),"
                        " SUM(a.carbohidratos_100g*pi2.gramos/100.0),"
                        " SUM(a.grasas_100g*pi2.gramos/100.0),"
                        " SUM(pi2.gramos)"
                        " FROM platos p"
                        " JOIN plato_ingredientes pi2 ON pi2.plato_id=p.id"
                        " JOIN alimentos a ON a.id=pi2.alimento_id"
                        " WHERE p.id=:pid GROUP BY p.id, p.nombre LIMIT 1"
                    ), {"pid": _best_id}).fetchone()
                    if plato_row:
                        logger.info(
                            "[NLPExtractor] Similarity %.2f: '%s' → '%s'",
                            _best_score, q_norm, plato_row[1],
                        )

            if plato_row:
                # Columnas: 0=id, 1=nombre, 2=kcal, 3=prot, 4=carb, 5=gras, 6=gramos_total
                nombre_final    = plato_row[1]
                calorias        = float(plato_row[2] or 0)
                proteinas       = float(plato_row[3] or 0)
                carbos          = float(plato_row[4] or 0)
                grasas          = float(plato_row[5] or 0)
                _gramos_plato   = float(plato_row[6] or 300.0)
                logger.info(
                    "[NLPExtractor] '%s' desde platos+ingredientes: %.1f kcal",
                    nombre_final, calorias,
                )
                # Aplicar modificadores sobre el plato completo
                if sin_lista or con_extra_lista:
                    calorias, proteinas, carbos, grasas, notas = self._aplicar_modificadores(
                        calorias, proteinas, carbos, grasas,
                        sin_lista, con_extra_lista, nombre_final
                    )
                    if notas:
                        advertencias.append(f"{nombre_final}: {', '.join(notas)}")
                # Escalar por cantidad
                calorias  = round(calorias  * cantidad, 1)
                proteinas = round(proteinas * cantidad, 1)
                carbos    = round(carbos    * cantidad, 1)
                grasas    = round(grasas    * cantidad, 1)
                gramos    = _gramos_plato * cantidad
            else:
                # PASO 3: Buscar en alimentos base
                alimento_bd = self._buscar_alimento_bd(nombre)

                # PASO 3b: Plato compuesto — descompone por "con"/"y" y busca cada ingrediente
                # Flujo por componente: BD (exacto) → USDA → Groq → guarda en BD
                if not alimento_bd:
                    # Para ml/cc tratar como g (densidad ≈ 1g/ml para bebidas).
                    # Sin esto, _calcular_macros_plato_combinado usaría escala=300
                    # para "300ml de smoothie" (300 porciones, resultado absurdo).
                    _UNIDADES_MASA_O_VOL = {"g", "gr", "gramo", "gramos", "ml", "cc"}
                    gramos_usuario = cantidad if unidad in _UNIDADES_MASA_O_VOL else None
                    resultado_combinado = await self._calcular_macros_plato_combinado(
                        q_norm, cantidad, unidad, gramos_usuario
                    )
                    if resultado_combinado:
                        calorias, proteinas, carbos, grasas, gramos = resultado_combinado
                        nombre_final = nombre.title()
                        logger.info("[NLPExtractor] Plato combinado '%s': %.1f kcal", nombre, calorias)
                        # Aplicar modificadores
                        if sin_lista or con_extra_lista:
                            calorias, proteinas, carbos, grasas, notas = self._aplicar_modificadores(
                                calorias, proteinas, carbos, grasas, sin_lista, con_extra_lista, nombre_final
                            )
                        items_calculados.append(ItemExtraido(
                            alimento=nombre_final, cantidad=cantidad, unidad=unidad,
                            gramos_totales=gramos, calorias=calorias,
                            proteinas_g=proteinas, carbohidratos_g=carbos, grasas_g=grasas,
                            origen="bd_combinado",
                        ))
                        continue

                # PASO 3d: Construcción dinámica con plato_constructor.
                # Activa para nombres ≥2 palabras no resueltos por pasos anteriores.
                # Cubre platos con "de"/"a la"/adjetivos sin "con"/"y" explícito
                # (p.ej. "arroz a la naranja", "sopa de mariscos", "ají de gallina").
                # El plato se persiste en BD para futuras consultas (cache).
                # Guardia: "dos peras", "tres manzanas" → no son platos, son cantidad+alimento.
                # El normalizador debería haberlo separado, pero como defensa extra se bloquea aquí.
                _primera_p = nombre.split()[0].lower() if nombre.split() else ""
                _es_cantidad_alim = _primera_p in NLPFoodExtractor._NUMEROS_ES

                if not alimento_bd and len(nombre.split()) >= 2 and not _es_cantidad_alim:
                    try:
                        from app.services.plato_constructor import crear_plato_dinamico as _cpd
                        _plato_din = await _cpd(self.db, nombre)
                        if _plato_din and _plato_din.id:
                            _din_row = self.db.execute(_text(
                                "SELECT p.id, p.nombre,"
                                " SUM(a.calorias_100g*pi2.gramos/100.0),"
                                " SUM(a.proteina_100g*pi2.gramos/100.0),"
                                " SUM(a.carbohidratos_100g*pi2.gramos/100.0),"
                                " SUM(a.grasas_100g*pi2.gramos/100.0),"
                                " SUM(pi2.gramos)"
                                " FROM platos p"
                                " JOIN plato_ingredientes pi2 ON pi2.plato_id=p.id"
                                " JOIN alimentos a ON a.id=pi2.alimento_id"
                                " WHERE p.id=:pid GROUP BY p.id,p.nombre LIMIT 1"
                            ), {"pid": _plato_din.id}).fetchone()
                            if _din_row:
                                nombre_final = str(_din_row[1]).title()
                                _kcal_d = round(float(_din_row[2] or 0), 1)
                                _prot_d = round(float(_din_row[3] or 0), 1)
                                _carb_d = round(float(_din_row[4] or 0), 1)
                                _gras_d = round(float(_din_row[5] or 0), 1)
                                _grms_d = float(_din_row[6] or 300.0)
                                # Si el usuario especificó gramos explícitos (unidad="g"),
                                # escalar el plato proporcionalmente en vez de multiplicar
                                # por cantidad (que en ese caso sería los gramos, no porciones).
                                if _unidad_es_peso and _grms_d > 0:
                                    _factor = cantidad / _grms_d
                                    calorias  = round(_kcal_d * _factor, 1)
                                    proteinas = round(_prot_d * _factor, 1)
                                    carbos    = round(_carb_d * _factor, 1)
                                    grasas    = round(_gras_d * _factor, 1)
                                    gramos    = cantidad  # exactamente los gramos del usuario
                                else:
                                    calorias  = round(_kcal_d * cantidad, 1)
                                    proteinas = round(_prot_d * cantidad, 1)
                                    carbos    = round(_carb_d * cantidad, 1)
                                    grasas    = round(_gras_d * cantidad, 1)
                                    gramos    = _grms_d * cantidad
                                logger.info(
                                    "[NLPExtractor] Plato dinámico '%s': %.1f kcal",
                                    nombre_final, calorias,
                                )
                                if sin_lista or con_extra_lista:
                                    calorias, proteinas, carbos, grasas, _ = \
                                        self._aplicar_modificadores(
                                            calorias, proteinas, carbos, grasas,
                                            sin_lista, con_extra_lista, nombre_final,
                                        )
                                import difflib as _dl_3d
                                _sim_3d = _dl_3d.SequenceMatcher(
                                    None, _norm(nombre_input), _norm(nombre_final)
                                ).ratio()
                                _confianza_baja_3d = _sim_3d < 0.85
                                items_calculados.append(ItemExtraido(
                                    alimento=nombre_final, cantidad=cantidad, unidad=unidad,
                                    gramos_totales=gramos, calorias=calorias,
                                    proteinas_g=proteinas, carbohidratos_g=carbos,
                                    grasas_g=grasas, origen="plato_dinamico",
                                    confianza_baja=_confianza_baja_3d,
                                ))
                                continue
                    except Exception as _ep:
                        logger.warning(
                            "[NLPExtractor] Plato dinámico falló para '%s': %s", nombre, _ep
                        )

                # PASO 3c: Si no encontró por nombre completo, buscar ingrediente base
                # Ej: "sopa de lentejas ligera" → busca "lentejas" en alimentos
                if not alimento_bd:
                    alimento_bd = self._buscar_ingrediente_base(q_norm)
                    if alimento_bd:
                        origen = "bd_ingrediente"

                # PASO 4: Fallback USDA (solo para nombres cortos/genéricos, no platos compuestos)
                if not alimento_bd:
                    macros_usda = self._buscar_usda(nombre)
                    if macros_usda:
                        # REGLA 4: fuente correcta para trazabilidad
                        alimento_bd = self._guardar_en_bd(nombre, macros_usda, "USDA (auto-aprendido)")
                        origen = "usda"

                # PASO 4b: Última instancia → Groq estima macros por 100g
                # REGLA 4: fuente "Groq (estimado)" → es_confiable=False, pendiente_validacion=True
                if not alimento_bd:
                    macros_groq = await self._groq_estimar_macros(nombre)
                    if macros_groq:
                        alimento_bd = self._guardar_en_bd(nombre, macros_groq, "Groq (estimado)")
                        origen = "groq"

                if not alimento_bd:
                    logger.warning(
                        "[NLPExtractor] '%s' no encontrado en BD/USDA/Groq — descartado",
                        nombre,
                    )
                    advertencias.append(f"No encontré datos para '{nombre}'")
                    continue

                gramos = self._resolver_gramos(alimento_bd, unidad, cantidad)
                factor = gramos / 100.0
                calorias  = round(float(alimento_bd.calorias_100g)      * factor, 1)
                proteinas = round(float(alimento_bd.proteina_100g)      * factor, 1)
                carbos    = round(float(alimento_bd.carbohidratos_100g) * factor, 1)
                grasas    = round(float(alimento_bd.grasas_100g)        * factor, 1)
                nombre_final = alimento_bd.nombre
                logger.info(
                    "[NLPExtractor] '%s' desde alimentos: %.1f kcal (%.0fg, %s)",
                    nombre_final, calorias, gramos, origen,
                )

                # Aplicar modificadores
                if sin_lista or con_extra_lista:
                    calorias, proteinas, carbos, grasas, notas = self._aplicar_modificadores(
                        calorias, proteinas, carbos, grasas,
                        sin_lista, con_extra_lista, nombre_final
                    )
                    if notas:
                        advertencias.append(f"{nombre_final}: {', '.join(notas)}")

            # Validación de seguridad: cantidad sospechosamente alta
            if calorias > KCAL_ITEM_WARN:
                advertencias.append(
                    f"'{nombre_final}' registra {calorias:.0f} kcal. "
                    f"¿Es correcto? ({cantidad:.1f} × {unidad})"
                )
            # Validar que el nombre sea un alimento real (no inventado)
            # Si llegó aquí sin plato_row y con alimento_bd=None es señal de USDA
            # Si origen='usda' y calorias=0, es alimento inválido
            if origen == "usda" and calorias == 0:
                advertencias.append(f"No encontré datos nutricionales para '{nombre_final}'.")
                continue

            # Capitalización uniforme: "pollo saltado" → "Pollo Saltado"
            nombre_final = nombre_final.title() if nombre_final == nombre_final.lower() else nombre_final

            # Detectar baja confianza:
            # 1. Groq estimó macros directamente (alimento no encontrado en BD ni USDA)
            # 2. El nombre resuelto difiere del nombre que el usuario escribió:
            #    - Para platos dinámicos y alimentos: umbral más estricto (0.85)
            #      porque el constructor puede "adivinar" platos incorrectos
            #    - Para BD directa: solo se marca baja si la diferencia es muy grande (<0.60)
            import difflib as _dl_conf
            _sim_nombres = _dl_conf.SequenceMatcher(
                None, _norm(nombre_input), _norm(nombre_final)
            ).ratio()
            _umbral = 0.85 if origen in ("plato_dinamico", "groq") else 0.60
            _confianza_baja = (
                origen == "groq"
                or (_sim_nombres < _umbral and origen not in ("bd_combinado", "bd_ingrediente", "regla", "bd"))
            )

            # Detectar si el ítem es líquido según su unidad (Bug 4 fix)
            _es_liquido_item = unidad in {"ml", "cc", "l", "litro", "vaso", "taza", "copa", "botella", "jarra"}

            items_calculados.append(ItemExtraido(
                alimento=nombre_final,
                cantidad=cantidad,
                unidad=unidad,
                gramos_totales=gramos if plato_row else gramos,
                calorias=calorias,
                proteinas_g=proteinas,
                carbohidratos_g=carbos,
                grasas_g=grasas,
                origen=origen,
                confianza_baja=_confianza_baja,
                es_liquido=_es_liquido_item,
            ))

        if not items_calculados:
            return None

        return ResultadoExtraccion(
            items=items_calculados,
            calorias_total=round(sum(i.calorias for i in items_calculados), 1),
            proteinas_total=round(sum(i.proteinas_g for i in items_calculados), 1),
            carbohidratos_total=round(sum(i.carbohidratos_g for i in items_calculados), 1),
            grasas_total=round(sum(i.grasas_g for i in items_calculados), 1),
            nombres=[i.alimento for i in items_calculados],
            advertencia="; ".join(advertencias) if advertencias else None,
        )
