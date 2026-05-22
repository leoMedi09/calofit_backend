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
from app.services.asistente_nutricion import coherencia_proteina_platos
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
- NO calcules calorías. NO sumes nada. SOLO extrae lo que el usuario mencionó.
- Extrae EXACTAMENTE lo que dijo el usuario, sin agregar ingredientes ni complementos que NO mencionó.
  Si el usuario dijo "pollo al horno", extrae "pollo al horno" — no añadas arroz, verduras ni nada más.
- NO separes platos tradicionales conocidos ni combinaciones de arroz/cereal con acompañamiento. Son UN solo ítem:
  "lomo saltado", "aji de gallina", "arroz con leche", "arroz con pollo", "arroz con verduras", "arroz con atun",
  "choclo con queso", "pan con huevo", "yogur con granola", "platano con yogur", "avena con leche",
  "pollo con arroz", "tallarines con carne", "sopa de verduras", "sopa de lentejas", "crema de verduras",
  "ensalada de pollo", "ensalada de frutas", "tortilla de huevo".
- REGLA CRÍTICA: "ensalada de X", "ensalada de X con Y", "sopa de X", "crema de X" siempre son UN solo ítem.
  Ejemplos: "ensalada de plátano con pollo" → UN ítem; "ensalada de frutas con yogur" → UN ítem.
  NUNCA extraigas los ingredientes internos de una ensalada/sopa/crema como ítems separados.
- REGLA CRÍTICA: "plátano" siempre es "plátano maduro" (banana dulce). NUNCA uses "plátano verde cocido".
  "plátano verde" solo si el usuario dice explícitamente "plátano verde" o "patacón" o "tostón".
- REGLA ESPECIAL: si el usuario dice "medio X" o "media X", extrae el ítem completo (X) con cantidad=0.5.
  Ejemplo: "media ensalada de plátano con pollo" → [{alimento:"ensalada de platano con pollo", cantidad:0.5, unidad:"porcion"}]
  Ejemplo: "medio arroz con verduras" → [{alimento:"arroz con verduras", cantidad:0.5, unidad:"porcion"}]
  NUNCA separes los ingredientes internos cuando se diga "medio" o "media".
- REGLA ESPECIAL: si el usuario dice "X con Y" donde Y es una verdura/guarnición, trátalo como UN plato.
  Ejemplo: "arroz con verduras", "pollo con papas", "pescado con ensalada" → UN ítem cada uno.
- Convierte cantidades verbales exactamente así:
    "medio" o "media" → 0.5
    "un" / "una" / "uno" → 1.0
    "dos" → 2.0
    "tres" → 3.0
    "cuatro" → 4.0
    "un par" → 2.0
    "un poco" → 0.5
    Si no hay cantidad → 1.0
- Elige la unidad más lógica:
    pan/galleta/huevo/fruta/unidad discreta → "unidad"
    leche/jugo/gaseosa/agua/bebida → "vaso"
    platos de comida completos/arroz/fideos/menestras → "porcion"
    aceite/salsa → "cucharada"
    Si el usuario especifica unidad (taza, vaso, plato), úsala.
    Si el usuario especifica gramos explícitamente (ej: "300g de pollo", "200 gramos de arroz") →
      usa unidad: "g" y cantidad: <número de gramos>. Ejemplo: "comí 250g de pollo al horno con plátano"
      → [{alimento:"pollo al horno con platano", cantidad:250, unidad:"g"}]
- Usa nombres EXACTOS y genéricos en español. Para bebidas: "agua", "gaseosa", "jugo de naranja".
- MODIFICADORES: "sin X" → agregar en campo "sin". "con extra X" / "con más X" → campo "con_extra".

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

# ─── Normalizaciones de texto antes de enviar al LLM ─────────────────────────
PRE_NORM_PATRONES = [
    # Conectores temporales que confunden al LLM → reemplazar por " y "
    (r"(?i)\b(y\s+)?despu[eé]s\s+(com[ií]|tom[eé]|beb[ií])\b", " y "),
    (r"(?i)\b(y\s+)?luego\s+(com[ií]|tom[eé]|beb[ií])\b",     " y "),
    (r"(?i)\btambi[eé]n\s+(com[ií]|tom[eé]|beb[ií])\b",        " y "),
    # Separador implícito por unidad: "arroz con huevo una taza de avena"
    # → "arroz con huevo y una taza de avena"
    # Lookbehind fijo (?<!con) evita romper "arroz con una taza de leche".
    (r"(?i)(?<![cC][oO][nN])\s+(?=un[ao]?\s+(?:taza|vaso|copa|plato|porci[oó]n)\s+de\s+)", " y "),
    # Redundancias comunes
    (r"(?i)\bun\s+poco\s+de\b",   "medio "),
    (r"\s{2,}",                    " "),
]

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
# Incluye ficción, objetos, materiales, actividades y conceptos abstractos.
NO_ALIMENTOS: frozenset[str] = frozenset({
    # Ficción / mitología
    "unicornio", "dragon", "flobonix", "zombie", "alien", "cripton",
    "monstruo", "magico", "invisible", "virtual", "digital", "fake",
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


@dataclass
class ResultadoExtraccion:
    items: List[ItemExtraido]
    calorias_total: float
    proteinas_total: float
    carbohidratos_total: float
    grasas_total: float
    nombres: List[str]
    advertencia: Optional[str]


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
        """Normaliza conectores temporales y redundancias para reducir varianza del LLM."""
        t = texto
        for patron, reemplazo in PRE_NORM_PATRONES:
            t = re.sub(patron, reemplazo, t)
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
            return json.loads(m.group(0))
        except Exception as e:
            print(f"[NLPExtractor] Error parsing LLM JSON: {e}")
            return []

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
        # 4. Fallback amplio: el nombre buscado aparece en cualquier posición.
        #    REGLA 6: ORDER BY LENGTH ASC garantiza que "Fruta" (genérico corto) tenga
        #    prioridad sobre "Ensalada de Frutas" o "Mezcla de Frutas" (compuestos largos).
        a4 = (
            self.db.query(Alimento)
            .filter(Alimento.nombre_normalizado.like(f"%{n}%"))
            .order_by(
                Alimento.nombre_normalizado.like(f"{n}%").desc(),
                _sqlfunc.length(Alimento.nombre_normalizado).asc(),
            )
            .first()
        )
        return a4

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
        }
        n = _norm(nombre_ingrediente)
        for kw, gramos in GRAMOS_TIPICOS.items():
            if kw in n:
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
            if self._RE_NOMBRE_MENSAJE.search(nombre_es):
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
        "pescado":  180.0, "salmon":   180.0, "atun":     100.0,
        "carne":    180.0, "res":       180.0, "cerdo":    180.0,
        "huevo":     50.0, "huevos":   100.0, "jamon":     60.0,
        "queso":     40.0, "tofu":     150.0,
        # Carbohidratos
        "arroz":    180.0, "papa":     150.0, "camote":   130.0,
        "fideos":   180.0, "tagliatelle": 180.0, "pasta": 180.0,
        "pan":       60.0, "quinua":   120.0, "yuca":     150.0,
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
        for clave, gramos in self._PORCIONES_COMPONENTE.items():
            if clave in n:
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

        if unidad == "g" and gramos_usuario and gramos_usuario > 0:
            # Usuario especificó gramos totales → distribuir proporcionalmente
            # Ej: "300g pollo al horno con plátano" → escala = 300 / gramos_std_total
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
            _m_qty_pre = re.match(r"(?i)^(dos|tres|cuatro|cinco|2|3|4|5)\s+(.+)$", _msg_clean)
            if _m_qty_pre:
                _qty_map_pre = {
                    "dos": 2.0, "tres": 3.0, "cuatro": 4.0, "cinco": 5.0,
                    "2": 2.0, "3": 3.0, "4": 4.0, "5": 5.0,
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
                        " SUM(a.grasas_100g*pi2.gramos/100.0)"
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
                                " SUM(a.grasas_100g*pi2.gramos/100.0)"
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
                        logger.info(
                            "[NLPExtractor] Pre-check: '%s' qty=%.0f %.1f kcal",
                            _pre_row[1], _qty_pre, _kcal0,
                        )
                        _item0 = ItemExtraido(
                            alimento=str(_pre_row[1]),
                            cantidad=_qty_pre,
                            unidad="porcion",
                            gramos_totales=100.0,
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
        if not items_raw:
            return None

        items_calculados: List[ItemExtraido] = []
        advertencias = []

        for item in items_raw:
            nombre   = str(item.get("alimento", "")).strip()
            cantidad = float(item.get("cantidad", 1.0) or 1.0)
            unidad   = str(item.get("unidad", "porcion")).strip().lower()
            sin_lista       = [str(x) for x in item.get("sin", []) if x]
            con_extra_lista = [str(x) for x in item.get("con_extra", []) if x]

            if not nombre or cantidad <= 0:
                continue

            # Guard: rechazar nombres que son claramente no-alimentos
            if _nombre_es_no_alimento(nombre):
                logger.warning("[NLPExtractor] '%s' bloqueado: no es un alimento", nombre)
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

            # 1) Exact match. 2) Si falla, similarity ≥ 0.83 sobre top-300.
            _SQL_PLATO = (
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
            plato_row = None if _unidad_es_peso else self.db.execute(_text(_SQL_PLATO), {"q": q_norm}).fetchone()

            # Fallback similaridad: si no hay exact match, buscar plato con mayor
            # similitud (≥0.83) para rescatar renombrados del LLM (ej: "huevo entero cocido"
            # cuando el usuario dijo "tortilla de huevo con pan tostado ligero").
            if not plato_row and not _unidad_es_peso:
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
                        " SUM(a.grasas_100g*pi2.gramos/100.0)"
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
                # Columnas: 0=id, 1=nombre, 2=kcal, 3=prot, 4=carb, 5=gras
                nombre_final = plato_row[1]
                calorias     = float(plato_row[2] or 0)
                proteinas    = float(plato_row[3] or 0)
                carbos       = float(plato_row[4] or 0)
                grasas       = float(plato_row[5] or 0)
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
                gramos    = 300.0 * cantidad  # porción estándar de plato
            else:
                # PASO 3: Buscar en alimentos base
                alimento_bd = self._buscar_alimento_bd(nombre)

                # PASO 3b: Plato compuesto — descompone por "con"/"y" y busca cada ingrediente
                # Flujo por componente: BD (exacto) → USDA → Groq → guarda en BD
                if not alimento_bd:
                    gramos_usuario = cantidad if unidad == "g" else None
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
                    advertencias.append(f"No encontré datos para '{nombre}'")
                    continue

                gramos = self._resolver_gramos(alimento_bd, unidad, cantidad)
                factor = gramos / 100.0
                calorias  = round(float(alimento_bd.calorias_100g)      * factor, 1)
                proteinas = round(float(alimento_bd.proteina_100g)      * factor, 1)
                carbos    = round(float(alimento_bd.carbohidratos_100g) * factor, 1)
                grasas    = round(float(alimento_bd.grasas_100g)        * factor, 1)
                nombre_final = alimento_bd.nombre

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
