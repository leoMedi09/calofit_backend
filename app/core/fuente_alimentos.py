"""Contraste de lo que estima el modelo con las tablas oficiales de composicion de alimentos.

Los macros de un registro los estima el modelo de lenguaje; aqui solo se comprueba si el alimento existe en
las tablas locales (INS/CENAN 2017, USDA, IMARPE) y si su densidad calorica (kcal por 100 g) cae en el rango de
esa tabla. Solo en ese caso se cita la tabla como fuente de contraste. Los platos preparados (lomo saltado, ají
de gallina...) no estan en esas tablas y quedan como "valor referencial": no se afirma una fuente que no hay.
"""

import json
import os
import re
import unicodedata
from typing import Dict, List, Optional

_RUTA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "alimentos_peru_ins.json")
_TOLERANCIA = 0.35  # ±35% sobre las kcal/100 g de la tabla (cubre crudo/cocido, marcas y porciones)
_ETIQUETAS = (("ins", "INS/CENAN"), ("cenan", "INS/CENAN"), ("usda", "USDA"), ("imarpe", "IMARPE"))

_indice: Optional[Dict[str, List[tuple]]] = None
_entradas: List[tuple] = []  # (nombre normalizado, palabras del nombre, kcal/100 g, etiqueta)
_VACIAS = {"de", "del", "la", "el", "con", "y", "en", "al", "a"}


def _norm(texto: str) -> str:
    t = unicodedata.normalize("NFKD", texto or "").encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", t)).strip()


def _etiqueta(fuente: Optional[str]) -> Optional[str]:
    f = (fuente or "").lower()
    return next((e for k, e in _ETIQUETAS if k in f), None)


def _cargar() -> Dict[str, List[tuple]]:
    global _indice
    if _indice is not None:
        return _indice
    _indice = {}
    try:
        with open(_RUTA, "r", encoding="utf-8") as f:
            datos = json.load(f)
    except Exception:
        return _indice
    for it in datos:
        etiqueta = _etiqueta(it.get("fuente"))
        try:
            kcal = float(it.get("calorias_100g"))
        except (TypeError, ValueError):
            continue
        if not etiqueta or kcal <= 0:
            continue
        for nombre in [it.get("alimento")] + list(it.get("alias") or []):
            clave = _norm(nombre or "")
            if clave:
                _indice.setdefault(clave, []).append((kcal, etiqueta))
                _entradas.append((clave, frozenset(clave.split()) - _VACIAS, kcal, etiqueta))
    return _indice


def contrastar(nombre: str, kcal_por_100g: float) -> Optional[str]:
    """Devuelve la tabla ("INS/CENAN", "USDA"...) si el alimento existe con ese nombre exacto (o alias) y sus kcal/100 g
    estan dentro de la tolerancia; si no, None."""
    if not nombre or kcal_por_100g <= 0:
        return None
    clave = _norm(nombre)
    indice = _cargar()
    candidatos = list(indice.get(clave) or indice.get(clave.rstrip("s")) or [])
    # Una palabra ("pollo", "leche"): cualquier entrada de la tabla que la contenga. Varias palabras ("arroz blanco"):
    # solo entradas que empiecen igual ("arroz blanco corriente"); "pan con huevo" NO coincide con "pan de huevo".
    palabras = {w.rstrip("s") if len(w) > 3 else w for w in clave.split()} - _VACIAS
    if len(clave.split()) == 1 and palabras:
        candidatos += [
            (k, e)
            for _, nombres, k, e in _entradas
            if palabras <= {w.rstrip("s") if len(w) > 3 else w for w in nombres}
        ]
    elif clave:
        candidatos += [(k, e) for nombre, _, k, e in _entradas if nombre.startswith(clave + " ")]
    for kcal, etiqueta in candidatos:
        if abs(kcal_por_100g - kcal) <= kcal * _TOLERANCIA:
            return etiqueta
    return None


def texto_fuente(contrastes: List[tuple]) -> str:
    """contrastes = [(nombre, etiqueta|None)] → linea para mostrar bajo el registro."""
    ok = [(n, e) for n, e in contrastes if e]
    resto = [n for n, e in contrastes if not e]
    partes = []
    if ok:
        tablas = " y ".join(dict.fromkeys(e for _, e in ok))
        partes.append(f"Contrastado con {tablas}: " + ", ".join(dict.fromkeys(n for n, _ in ok)))
    if resto:
        partes.append("Valor referencial: " + ", ".join(dict.fromkeys(resto)))
    return " · ".join(partes)
