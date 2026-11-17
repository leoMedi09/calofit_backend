"""Registro de comidas/ejercicio de días pasados.

Dos formas de indicar el día:
- El calendario del Balance (la app envía ``fecha`` en cada mensaje del chat): ``validar_fecha``.
- Las palabras fijas «ayer», «anteayer», «anoche» dentro del mensaje: ``extraer_fecha_retroactiva`` (sin costo, sin modelo).

El servidor SIEMPRE valida la fecha: no futura y máximo ``MAX_DIAS_RETRO`` días atrás.
"""
import re
import unicodedata
from datetime import date, datetime, timedelta
from typing import Optional, Tuple

from app.core.utils import get_peru_date

MAX_DIAS_RETRO = 7

_RX_PASADO = re.compile(r"\b(antes\s+de\s+ayer|anteayer|antier|ayer|anoche)\b")
_RX_HOY = re.compile(r"\bhoy\b")
_DIAS_ATRAS = {"ayer": 1, "anoche": 1, "anteayer": 2, "antier": 2}


def _sin_acentos(texto: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", texto) if unicodedata.category(c) != "Mn")


def extraer_fecha_retroactiva(mensaje: str, hoy: Optional[date] = None) -> Tuple[Optional[date], str]:
    """(fecha, mensaje_sin_la_palabra_de_fecha). (None, mensaje) si no hay fecha pasada o es ambigua."""
    mensaje = mensaje or ""
    norm = _sin_acentos(mensaje.lower())
    m = _RX_PASADO.search(norm)
    if not m or _RX_HOY.search(norm):
        return None, mensaje
    clave = re.sub(r"\s+", " ", m.group(1))
    dias = 2 if clave == "antes de ayer" else _DIAS_ATRAS[clave]
    # quitar acentos no cambia la longitud (texto precompuesto): se recorta la misma palabra del mensaje original
    limpio = (mensaje[: m.start()] + mensaje[m.end():]).strip(" ,.;:")
    limpio = re.sub(r"\s{2,}", " ", limpio).strip(" ,.;:")
    return (hoy or get_peru_date()) - timedelta(days=dias), limpio or mensaje


def etiqueta_fecha(fecha: date) -> str:
    return fecha.strftime("%d/%m")


def validar_fecha(valor, hoy: Optional[date] = None, max_dias: int = MAX_DIAS_RETRO) -> Tuple[Optional[date], Optional[str]]:
    """(fecha, error). fecha None sin error = hoy / sin fecha; con error = no se debe registrar nada."""
    hoy = hoy or get_peru_date()
    if valor in (None, "", "null", "None"):
        return None, None
    try:
        f = datetime.strptime(str(valor).strip()[:10], "%Y-%m-%d").date()
    except ValueError:
        return None, None  # no es una fecha: se ignora (queda como hoy)
    if f == hoy:
        return None, None
    if f > hoy:
        try:  # «31/12» escrito el 2 de enero: se corrige al año anterior si cae dentro del límite
            previo = f.replace(year=f.year - 1)
        except ValueError:  # 29/02
            previo = None
        if previo and 0 < (hoy - previo).days <= max_dias:
            return previo, None
        return None, f"Esa fecha ({etiqueta_fecha(f)}) todavía no llega. ¿Quisiste decir otro día? No registré nada."
    if (hoy - f).days > max_dias:
        return None, (
            f"Solo puedo registrar hasta {max_dias} días atrás (desde el {etiqueta_fecha(hoy - timedelta(days=max_dias))}). "
            f"Tu fecha fue el {etiqueta_fecha(f)}. No registré nada."
        )
    return f, None
