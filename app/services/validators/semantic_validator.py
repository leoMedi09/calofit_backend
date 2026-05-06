"""
Validador semántico: coherencia culinaria de ingredientes en un plato.
"""
from typing import Dict, Any, List, Optional, Tuple
from sqlalchemy.orm import Session
from app.services.validators.base_validator import BaseValidator, ValidationResult


# Reglas: por tipo de plato → ingredientes requeridos y prohibidos
_REGLAS: Dict[str, Dict] = {
    "ceviche": {
        "requiere":  ["pescado", "limón", "lima"],
        "prohíbe":   ["leche", "queso", "mantequilla", "yogur", "mayonesa"],
        "descripcion": "Ceviche debe llevar pescado fresco + limón/lima, sin lácteos.",
    },
    "tiradito": {
        "requiere":  ["pescado"],
        "prohíbe":   ["leche", "queso", "mantequilla"],
        "descripcion": "Tiradito lleva pescado fresco marinado, sin lácteos.",
    },
    "ensalada": {
        "requiere":  [],
        "prohíbe":   [],
        "descripcion": "Ensalada: predominio de vegetales frescos.",
    },
    "sopa": {
        "requiere":  [],
        "prohíbe":   [],
        "descripcion": "Sopa: base líquida.",
    },
    "tostada": {
        "requiere":  [],
        "prohíbe":   [],
        "descripcion": "Tostada: pan tostado con ingredientes coherentes.",
    },
}

# Mapa de categorías para coherencia general
_CATEGORIAS: Dict[str, List[str]] = {
    "pescados":       ["pescado", "corvina", "lenguado", "caballa", "lisa", "mero", "tollo",
                       "atún", "bonito", "sardina", "anchoveta", "salpreso"],
    "mariscos":       ["camarón", "langostino", "cangrejo", "pulpo", "calamar", "mejillón",
                       "concha", "choro"],
    "lácteos":        ["leche", "queso", "yogur", "mantequilla", "crema", "nata", "lácteo"],
    "verduras":       ["lechuga", "tomate", "pepino", "zanahoria", "brócoli", "espinaca",
                       "cebolla", "ajo", "apio", "zapallo", "coliflor"],
    "carbohidratos":  ["arroz", "papa", "camote", "pasta", "pan", "quinua", "maíz",
                       "yuca", "trigo"],
    "proteínas":      ["pollo", "pato", "pavo", "res", "cerdo", "cabrito", "cordero",
                       "huevo", "lentejas", "frijol", "pallares"],
    "gluten":         ["trigo", "harina", "pasta", "pan", "cebada", "centeno", "avena",
                       "sémola", "gluten"],
    "frutas":         ["manzana", "naranja", "plátano", "limón", "lima", "mango", "uva",
                       "piña", "pera", "durazno", "fresa"],
    "condimentos":    ["sal", "pimienta", "comino", "orégano", "ají", "aceite", "vinagre",
                       "limón", "ajo", "cilantro", "perejil"],
}

# Combinaciones incoherentes culinariamente
# Cada entrada: (patron_nombre_plato, ingrediente_incompatible, mensaje)
_COMBOS_INCOHERENTES: List[tuple] = [
    # Plato base + ingrediente que no tiene sentido
    ("tostada",   "plátano",       "Tostada de plátano no es una preparación estándar — considera 'plátano asado' o 'tostones'"),
    ("tostada",   "platano",       "Tostada de plátano no es una preparación estándar"),
    ("tostada",   "leche en polvo","Leche en polvo no es un ingrediente típico en tostadas"),
    ("arroz",     "leche en polvo","Arroz con leche en polvo es incoherente — usa leche fresca para arroz con leche"),
    ("ceviche",   "leche en polvo","Leche en polvo no pertenece a un ceviche"),
    ("sopa",      "leche en polvo","Leche en polvo en sopa salada es incoherente — usa leche fresca o caldo"),
    ("pollo",     "leche en polvo","Leche en polvo no se usa en platos de pollo salado"),
    ("ensalada",  "leche en polvo","Leche en polvo no pertenece a una ensalada"),
]

# Ingredientes que sugieren procesamiento industrial fuera de contexto
_INGREDIENTES_PROCESADOS_FUERA_CONTEXTO = [
    "leche en polvo",
    "margarina",
    "sustituto",
    "colorante",
    "saborizante artificial",
]

# Conjunto de tokens reconocidos como no-alimento / ficticios
_TOKENS_FICTICIOS: frozenset = frozenset({
    "unicornio", "dragon", "dragón", "fenix", "fénix", "centauro", "hada", "grifo",
    "hidra", "quimera", "sirena", "goblin", "pixie", "mágico", "magico",
    "mitológico", "mitologico", "ficticio",
})


def _detectar_tipo_plato(nombre: str) -> str:
    n = nombre.lower()
    for tipo in _REGLAS:
        if tipo in n:
            return tipo
    return ""


def _categorias_de(ingredientes: List[str]) -> set:
    cats: set = set()
    for ing in ingredientes:
        for cat, items in _CATEGORIAS.items():
            if any(item in ing for item in items):
                cats.add(cat)
    return cats


class SemanticValidator(BaseValidator):
    """
    Valida coherencia semántica culinaria de un plato.

    Entrada (datos):
        nombre_plato  : str
        ingredientes  : [{"nombre": str, "gramos": float}, ...]
        client_id     : int  (opcional – para restricciones personales)
    """

    def __init__(self, db: Optional[Session] = None) -> None:
        super().__init__("SemanticValidator")
        self.db = db

    def validar(self, datos: Dict[str, Any]) -> ValidationResult:
        nombre_plato = (datos.get("nombre_plato") or "").strip()
        ingredientes  = datos.get("ingredientes") or []
        client_id     = datos.get("client_id")

        if not nombre_plato:
            return self._crear_resultado(False, 0, errores=["Nombre de plato vacío"])
        if not ingredientes:
            return self._crear_resultado(False, 0, errores=["Plato sin ingredientes"])

        nombres = [ing.get("nombre", "").lower().strip() for ing in ingredientes]
        errores: List[str] = []
        advertencias: List[str] = []
        sugerencias: List[str] = []
        confianza = 100

        # 1. Restricciones del cliente
        if client_id and self.db:
            r = self._restricciones_cliente(client_id, nombres)
            errores.extend(r["errores"])
            advertencias.extend(r["advertencias"])
            if r["errores"]:
                confianza -= 30

        # 2. Reglas culinarias del tipo de plato
        tipo = _detectar_tipo_plato(nombre_plato)
        if tipo and tipo in _REGLAS:
            regla = _REGLAS[tipo]
            for req in regla["requiere"]:
                if not any(req in ing for ing in nombres):
                    advertencias.append(
                        f"{tipo.capitalize()} habitualmente lleva '{req}'"
                    )
                    confianza -= 5
            for proh in regla["prohíbe"]:
                if any(proh in ing for ing in nombres):
                    errores.append(
                        f"{tipo.capitalize()} no debe llevar '{proh}': {regla['descripcion']}"
                    )
                    confianza -= 20

        # 3. Combos culinariamente incoherentes
        nombre_plato_lower = nombre_plato.lower()
        nombres_ings_lower = " ".join(nombres)
        for patron_plato, ingrediente_incompat, mensaje in _COMBOS_INCOHERENTES:
            plato_match = patron_plato in nombre_plato_lower
            ing_match = ingrediente_incompat in nombres_ings_lower or \
                        ingrediente_incompat in nombre_plato_lower
            if plato_match and ing_match:
                advertencias.append(f"Combinación incoherente: {mensaje}")
                confianza -= 25

        # 4. Ingredientes ficticios o no reconocidos
        tokens = set(t for n in nombres for t in n.split())
        ficticios = tokens & _TOKENS_FICTICIOS
        if ficticios:
            advertencias.append(
                f"Ingredientes no reconocidos como alimentos reales: {', '.join(sorted(ficticios))}"
            )
            confianza -= 60

        cats = _categorias_de(nombres)
        if not cats and not ficticios:
            advertencias.append("Ningún ingrediente fue reconocido en categorías alimentarias conocidas")
            confianza -= 30

        # 4. Coherencia general
        if len(ingredientes) > 10:
            advertencias.append(
                f"Plato con {len(ingredientes)} ingredientes — verifica que sea coherente"
            )
            confianza -= 5

        if len(cats) == 1:
            advertencias.append(
                f"Todos los ingredientes caen en una sola categoría: {list(cats)[0]}"
            )

        return self._crear_resultado(
            es_valido=len(errores) == 0,
            confianza=max(0, confianza),
            errores=errores,
            advertencias=advertencias,
            sugerencias=sugerencias,
            metadata={
                "tipo_plato_detectado": tipo or "genérico",
                "total_ingredientes": len(ingredientes),
                "categorias": list(cats),
            },
        )

    def _restricciones_cliente(
        self, client_id: int, ingredientes: List[str]
    ) -> Dict[str, List[str]]:
        try:
            from app.models import Client
            cliente = self.db.query(Client).filter(Client.id == client_id).first()
            if not cliente:
                return {"errores": [], "advertencias": []}

            raw_forbidden = [f.lower().strip() for f in (cliente.forbidden_foods or [])]
            errores: List[str] = []

            # Expandir categorías: si forbidden contiene "lácteos", bloquear todos los lácteos
            forbidden_tokens: List[str] = []
            for f in raw_forbidden:
                if f in _CATEGORIAS:
                    forbidden_tokens.extend(_CATEGORIAS[f])
                else:
                    forbidden_tokens.append(f)

            for ing in ingredientes:
                # Coincidencia directa (forbidden token dentro del nombre del ing)
                # O el nombre del ing dentro del token prohibido
                if any(tok in ing or ing in tok for tok in forbidden_tokens):
                    errores.append(f"'{ing}' está en la lista prohibida del cliente")

            return {"errores": errores, "advertencias": []}
        except Exception as exc:
            return {"errores": [], "advertencias": [f"No se pudo verificar restricciones: {exc}"]}
