"""Filtros de seguridad y calidad para las recomendaciones del KNN (nivel ingrediente/producto).

El KNN solo compara perfiles de macros por 100 g; por sí solo puede devolver fórmula infantil a un
intolerante a la lactosa, pescado salado a un hipertenso o tres frejoles casi iguales. Este módulo
descarta esos candidatos ANTES de elegir el top-N. No reemplaza el criterio del nutricionista.

Se usa solo en el KNN: `recomendador_platos._CONDICION_TOKENS` filtra PLATOS ('seco' allí sería 'Seco de res').
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Optional

# alias (normalizado) -> condición canónica
_ALIAS_CONDICION = {
    "vegano": "vegano", "vegetariano": "vegetariano",
    "intolerancia a la lactosa": "lactosa", "lactosa": "lactosa",
    "hipertension": "hipertension", "hipertension arterial": "hipertension",
    "diabetes": "diabetes", "diabetes tipo 1": "diabetes", "diabetes tipo 2": "diabetes",
    "celiaco": "celiaco", "celiaquia": "celiaco", "enfermedad celiaca": "celiaco",
}

_LACTEOS = {"leche", "queso", "yogur", "yogurt", "mantequilla", "crema", "lactosa", "quesillo", "requeson", "helado", "suero",
            "caseina", "manjar", "maternizada", "nan", "cerelac", "nestum", "milo", "evaporada", "condensada", "kefir", "natilla", "chocolate"}
_SODIO = {"salado", "salada", "charque", "conserva", "enlatado", "enlatada", "ahumado", "ahumada", "embutido", "jamon", "tocino", "salchicha",
          "chorizo", "mortadela", "frito", "frita", "chicharron", "deshidratado", "deshidratada", "queso", "snack", "chifle", "aceituna", "caldo", "sardina"}
_AZUCAR = {"azucar", "dulce", "miel", "mermelada", "gaseosa", "refresco", "chocolate", "galleta", "caramelo", "jarabe", "helado", "torta",
           "milo", "nestum", "cerelac", "manjar", "alfajor", "bizcocho", "keke"}
_GLUTEN = {"trigo", "cebada", "centeno", "avena", "pan", "pasta", "fideo", "fideos", "tallarin", "galleta", "cuscus", "tostada", "milo",
           "cerelac", "nestum", "semola", "malta", "bizcocho"}
TOKENS_CONDICION = {"lactosa": _LACTEOS, "hipertension": _SODIO, "diabetes": _AZUCAR, "celiaco": _GLUTEN}

# Siempre fuera del KNN: fórmula/cereal infantil, condimentos y suplementos, repostería y dulces.
_NO_RECOMENDABLE = {
    # fórmula/cereal infantil
    "maternizada", "cerelac", "nestum", "papilla",
    # condimentos, especias, aditivos y suplementos (singular y plural)
    "levadura", "palillo", "paprika", "oregano", "comino", "cominos", "achiote", "pimienta", "canela", "clavo", "laurel", "estragon", "anis", "tomillo",
    "sazonador", "condimento", "condimentos", "vinagre", "esencia", "vainilla", "colorante", "bicarbonato", "extracto", "algarrobina", "cocoa",
    # grasas y aceites puros, harinas sueltas, algas y polvos (no son un alimento que se recomiende comer solo)
    "aceite", "manteca", "margarina", "harina", "alga", "algas", "polvo", "almidon", "maicena",
    # repostería y dulces
    "bollito", "alfajor", "caramelo", "portenitas", "keke"}
_NO_RECOMENDABLE_FRASE = ("polvo de hornear", "concentrado proteico", "pan dulce", "galleta dulce", "pan de yema", "pan torta", "seco, molido",
                          "molido (paprika)", "proteinas, (")

_MOMENTO_NO = {
    "desayuno": {"pescado", "caballa", "res", "cerdo", "pollo", "carne", "charque", "higado", "pulpo", "camaron", "frejol", "frijol", "pallar", "lenteja",
                 "garbanzo", "haba", "arveja", "soya"},
    "merienda": {"pescado", "caballa", "res", "cerdo", "pollo", "carne", "charque", "higado", "pulpo", "camaron", "frejol", "frijol", "pallar",
                 "lenteja", "garbanzo", "haba", "arveja", "arroz"},
}
_MOMENTO_NO["snack"] = _MOMENTO_NO["merienda"]

_FAMILIAS = (
    ("legumbre", {"frejol", "frijol", "pallar", "haba", "habas", "lenteja", "lentejas", "garbanzo", "garbanzos", "arveja", "poroto", "soya", "tarwi"}),
    ("grano", {"quinua", "canihua", "kiwicha", "achita", "trigo", "cebada", "maiz", "avena", "arroz", "semola", "mote"}),
    ("pan", {"pan", "tostada", "galleta", "galletas", "bizcocho"}),
    ("pescado", {"pescado", "caballa", "sardina", "atun", "lorna", "merluza", "boquichico", "camaroncito", "camaron"}),
    ("carne", {"res", "cerdo", "pollo", "pavo", "carne", "churrasco", "charque", "higado"}),
    ("semilla", {"chia", "almendra", "mani", "nuez", "ajonjoli", "pecana"}),
)

# frase de la lista "evitar" del usuario (normalizada) -> palabras que la violan
_ROJAS = {"res", "cerdo", "chancho", "churrasco", "cordero", "ternera", "charque", "higado", "bistec", "hamburguesa"}
_PROCESADO = {"galleta", "snack", "embutido", "salchicha", "jamon", "tocino", "hamburguesa", "nugget", "instantaneo", "chifle", "bollito"}


def norm(s: Optional[str]) -> str:
    s = unicodedata.normalize("NFD", (s or "").lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def _palabras(nombre: str) -> set[str]:
    return set(re.findall(r"[a-z0-9ñ]+", norm(nombre)))


def _condiciones(condiciones: Optional[Iterable[str]]) -> set[str]:
    return {_ALIAS_CONDICION[norm(c).strip()] for c in (condiciones or []) if norm(c).strip() in _ALIAS_CONDICION}


def _tokens_de_prohibidos(prohibidos: Optional[Iterable[str]]) -> set[str]:
    """Convierte frases como 'carnes rojas con grasa' o 'alimentos procesados' en palabras que las violan."""
    out: set[str] = set()
    for fr in prohibidos or []:
        f = norm(fr)
        if "carne" in f and "roja" in f:
            out |= _ROJAS
        if "procesad" in f or "snack" in f or "golosina" in f or "comida rapida" in f or "embutid" in f:
            out |= _PROCESADO
        if "azucar" in f or "bebida" in f and "azuc" in f:
            out |= _AZUCAR
        if "sodio" in f or "sal " in f + " ":
            out |= _SODIO
        if "grasa" in f:
            out |= {"frito", "frita", "chicharron", "tocino", "manteca", "mantequilla", "piel"}
        if "lacte" in f or "leche" in f or "queso" in f or "yogur" in f:
            out |= _LACTEOS
    return out


def excluir(nombre: str, condiciones: Optional[Iterable[str]] = None, prohibidos: Optional[Iterable[str]] = None,
            momento: Optional[str] = None) -> Optional[str]:
    """Devuelve el motivo si el alimento NO debe recomendarse, o None si es aceptable."""
    n, pal = norm(nombre), _palabras(nombre)
    if not es_peruano_conocido(nombre):
        return "no es un alimento peruano de consumo común (variedad rara, vísceras, extranjero o procesado)"
    if (_NO_RECOMENDABLE & pal) or any(f in n for f in _NO_RECOMENDABLE_FRASE):
        return "no es un alimento recomendable (infantil, condimento, suplemento o repostería)"
    conds = _condiciones(condiciones)
    for c in conds & set(TOKENS_CONDICION):
        if TOKENS_CONDICION[c] & pal:
            return f"incompatible con {c}"
        if c == "hipertension" and ("seco" in pal or "seca" in pal) and pal & {"pescado", "camaron", "camaroncito", "res", "carne", "pota", "pulpo", "calamar"}:
            return "incompatible con hipertension (producto seco/salado)"
    if conds & {"vegano", "vegetariano"}:
        from app.services.recomendador_platos import _CONDICION_TOKENS

        for c in conds & {"vegano", "vegetariano"}:
            if any(t in pal for t in {norm(x) for x in _CONDICION_TOKENS.get(c.capitalize(), set())}):
                return f"incompatible con dieta {c}"
    if _tokens_de_prohibidos(prohibidos) & pal:
        return "está en la lista de alimentos a evitar del usuario"
    if prohibidos and any("sodio" in norm(f) for f in prohibidos) and ("seco" in pal or "seca" in pal) and pal & {"pescado", "camaron", "camaroncito", "res", "carne", "pota", "pulpo", "calamar", "carnero"}:
        return "producto seco/salado (usuario evita sodio alto)"
    if prohibidos and any("procesad" in norm(f) or "snack" in norm(f) or "comida rapida" in norm(f) for f in prohibidos) and (
        " - " in nombre or "molde" in pal or "mollete" in pal or "tallarines" in pal or "pasta" in pal
    ):
        return "producto industrial o ultraprocesado (usuario evita procesados)"
    if momento and _MOMENTO_NO.get(norm(momento)) and _MOMENTO_NO[norm(momento)] & pal:
        return f"no corresponde al momento ({momento})"
    return None


# ---------------------------------------------------------------------------------------------------------------
# Alimentos peruanos CONOCIDOS: la tabla INS/CENAN trae variedades de laboratorio, vísceras, especies amazónicas
# casi desconocidas y productos extranjeros con nombre en inglés. Para el KNN solo se dejan los de consumo común.
# ---------------------------------------------------------------------------------------------------------------
_INGLES = {"bean", "beans", "bush", "pole", "kidney", "bountiful", "white", "black", "original", "classic", "light", "brown", "chips", "olive", "california"}
_RAROS_GENERAL = ("con piel", "c/piel", "sin coronta", "maca, pasta", "variedad", "nativa", "silvestre", "deshidrat", "chuno", "chaquepa", "fortificad", "frito", "frita", "huevera", "con espinas", "afrecho", "afrechillo")
_INICIO_RARO = {"hiwicto", "kara", "aska", "sachapapa", "macambo", "pijuayo", "tumbo", "chonta", "llipta", "jetka", "janchoy", "manchay", "muna", "pajuro", "pirca",
                "siuca", "chijchipa", "chullcos", "llampun", "airampo", "tallapaqui", "achira", "achita", "shiwawaco", "cushusho", "guaba", "gigante", "carnero",
                "alpaca", "llama", "cuy", "inca", "siete", "cashu", "lapas", "concha", "ñuña", "nuna", "cominos", "abridores", "rosquita",
                "ashipa", "koshnipa", "koshñipa", "nuzive", "purunkari", "quillu", "ungurahui", "ungurauy", "humari", "humarí", "guindon", "guindón", "caza", "poroto",
                "chancay", "camaroncito", "cordero", "seitan", "seitán", "tarwi", "yacon", "mashua", "pituca", "llacon", "sancayo", "yak'u", "warr'a", "huito", "kumuvi", "kuros",
                "sorgo", "pecana", "pecanas", "codorniz", "arvejon", "7", "dátiles", "datiles", "tamarindo"}
_VISCERAS = {"higado", "bazo", "rinon", "sesos", "panza", "mondongo", "lengua", "pulmon", "bofe", "corazon", "ubre", "sangre", "criadillas", "cabeza", "pata",
             "patas", "molleja", "mollejas", "visceras", "menudencia", "cuero", "alas"}
_NO_ALIMENTO = {  # bebidas, alcohol, azúcares, embutidos, condimentos/aromáticas y fauna silvestre: no sirven para cubrir un déficit
    "cerveza", "vino", "coca", "cola", "frugos", "jugo", "nectar", "refresco", "bebida", "te", "cafe", "chicha", "agua", "azucar", "miel", "mermelada", "chancaca",
    "gelatina", "salchicha", "hot", "mortadela", "jamonada", "jamon", "chorizo", "tocino", "pate", "relleno", "sangrecita", "ballena", "majaz", "sajino", "motelo",
    "iguana", "tortuga", "rana", "caracol", "cecina", "venado", "salsa", "sillao", "aji", "ajo", "albahaca", "huacatay", "culantro", "perejil", "rocoto", "kion",
    "romero", "hierbabuena", "ruda", "paico", "mostaza", "condensada", "aceituna", "aceitunas", "alcaparras", "pique", "nuez moscada", "sal", "caldo", "leche materna",
    "chicharron", "barquillos", "tostada", "galleta", "galletas", "barra", "chivo", "conejo", "relleno", "erizo", "machas", "loche", "chanque", "hemico",
}
_FREJOL_OK = {"canario", "castilla", "caballero", "bayo", "panamito", "palo", "negro", "chiclayo", "soya"}
_PAN_OK = {"frances", "integral", "molde", "mollete", "marraqueta", "huevo", "campesino", "granos"}
_PESCADO_OK = {"anchoveta", "atun", "bonito", "caballa", "cabinza", "cachema", "ayanque", "corvina", "jurel", "lenguado", "lisa", "lorna", "merluza", "mero", "pejerrey",
               "perico", "robalo", "sardina", "tilapia", "toyo", "trucha", "cojinova", "pampanito", "bacalao", "congrio", "chita", "pintadilla", "machete", "cabrilla"}
_PESCADO_NO = ("salado", "seco", "seca", "conserva", "con tomate", "grated", "enlatado en aceite")
_MAIZ_OK = {"amarillo", "blanco", "choclo", "mote", "cancha", "morado", "polenta"}
_QUINUA_OK = {"quinua", "quinua cocida", "quinua, hojuelas de", "quinua, semola de"}
_PAPA_OK = {"blanca", "amarilla", "canchan", "huayro"}
_CAMOTE_OK = {"amarillo", "blanco", "morado"}


def es_peruano_conocido(nombre: str) -> bool:
    n = norm(nombre)
    pal = _palabras(nombre)
    ini = re.split(r"[ ,(]", n)[0]
    if (_NO_ALIMENTO & pal) or "materna" in n or "queso de chancho" in n or (_INGLES & pal) or " - " in nombre or "�" in nombre or any(r in n for r in _RAROS_GENERAL) or ini in _INICIO_RARO:
        return False
    if ini in ("frejol", "frijol"):
        return bool(_FREJOL_OK & pal) and not any(x in n for x in ("tarhui", "chocho", "bebida", "cocido con"))
    if ini == "pan":
        return bool(_PAN_OK & pal)
    if ini == "pescado":
        return bool(_PESCADO_OK & pal) and not any(x in n for x in _PESCADO_NO)
    if ini == "maiz":
        return bool(_MAIZ_OK & pal)
    if ini == "quinua":
        return n.rstrip("* ") in _QUINUA_OK
    if ini == "trigo":
        return n.strip() == "trigo"
    if ini == "papa":
        return bool(_PAPA_OK & pal) and not any(x in n for x in ("seca", "helada"))
    if ini == "camote":
        return bool(_CAMOTE_OK & pal)
    if ini == "huevo":
        return "gallina" in pal and "crudo" not in pal and "yema" not in pal and "clara" not in pal
    if ini == "arroz":
        return "cascara" not in n or "integral" in n
    if ini in ("res", "pollo", "cerdo", "pavo", "carne"):
        return not (_VISCERAS & pal) and not ("crudo" in pal or "cruda" in pal) and "charque" not in pal
    if ini == "pallar":
        return not ({"manu", "morado"} & pal)
    if ini in ("canihua", "kiwicha", "cebada"):
        return n.strip() in ("canihua", "kiwicha", "cebada perlada o resbalada cocida", "cebada perlada o resbalada cruda") or n.startswith("kiwicha pop")
    return True


# ---------------------------------------------------------------------------------------------------------------
# Nombre legible: la tabla INS/CENAN escribe «Alimento, parte de» y deja marcas de nota al pie (*, **, ***).
# ---------------------------------------------------------------------------------------------------------------
_CALIFICADORES = {"cruda", "crudo", "cocida", "cocido", "fresco", "fresca", "seco", "seca", "descremada", "entera", "asada", "sancochada", "sancochado", "tostado", "tostada"}
_LEGIBLE = {
    "res, carne pulpa de": "Carne de res (pulpa)", "pollo, carne pulpa": "Carne de pollo (pulpa)", "res, cuadril, churrasco": "Churrasco de res (cuadril)",
    "cerdo, churrasco c/piel": "Churrasco de cerdo con piel", "pasta con huevo sin fortificar, cruda": "Pasta con huevo cruda", "pasta sin fortificar, cruda": "Pasta cruda",
    "pasta con huevo fortificada, cruda": "Pasta con huevo cruda", "leche fresca de vaca, descremada": "Leche fresca de vaca descremada", "col, hojas de": "Hojas de col",
}


def nombre_legible(nombre: str) -> str:
    n = re.sub(r"\s+", " ", re.sub(r"\*+", "", str(nombre))).strip(" ,;")
    if norm(n) in _LEGIBLE:
        return _LEGIBLE[norm(n)]
    m = re.match(r"^(.+?), (.+?) de$", n, re.I)
    if m:
        n = f"{m.group(2)} de {m.group(1).lower()}"
    else:
        m = re.match(r"^(.+?), ([^,]+)$", n)
        if m and norm(m.group(2)).split()[0] in _CALIFICADORES:
            n = f"{m.group(1)} {m.group(2)}"
    return n[:1].upper() + n[1:]


def familia(nombre: str) -> str:
    pal = _palabras(nombre)
    for fam, palabras in _FAMILIAS:
        if palabras & pal:
            return fam
    return norm(nombre).split(" ")[0] or "otro"
