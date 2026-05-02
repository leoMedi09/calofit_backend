from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from sqlalchemy.orm import Session

from app.models.alimento import Alimento
from app.models.alimento_alias import AlimentoAlias
from app.models.alimento_unidad import AlimentoUnidad
from app.utils.alimento_nombre import norm_alimento_key


def _norm(s: str) -> str:
    return norm_alimento_key(s)


def _parse_cantidad_token(token: str) -> float:
    """Convierte '1,5', 'media', 'una' a float (fracciones para lata/tarro)."""
    p = (token or "").strip().lower()
    if not p:
        return 1.0
    m = re.match(r"^\s*(\d+(?:[.,]\d+)?)\b", p)
    if m:
        return float(m.group(1).replace(",", "."))
    mapa = {
        "un": 1.0,
        "uno": 1.0,
        "una": 1.0,
        "dos": 2.0,
        "tres": 3.0,
        "cuatro": 4.0,
        "cinco": 5.0,
        "media": 0.5,
        "medio": 0.5,
    }
    for k, v in mapa.items():
        if p.startswith(k):
            return float(v)
    return 1.0


_RX_G = re.compile(r"(?i)^\s*(?P<num>[\d.,]+)\s*(?:g|gr|gramos?)\s+(?P<name>.+?)\s*$")
_RX_UNIDAD = re.compile(r"(?i)^\s*(?P<num>[\d.,]+)\s+(?P<unit>[a-záéíóúñ\s\.]+?)\s+de\s+(?P<name>.+?)\s*$")
# "una lata de atun", "media lata de atun" (sin dígito al inicio)
# Cantidad en palabras + unidad + "de" + alimento (sin dígito obligatorio)
_RX_UNIDAD_ES = re.compile(
    r"(?i)^\s*(?P<num>[\d.,]+|media|medio|un|una|uno|dos|tres|cuatro|cinco)\s+"
    r"(?P<unit>"
    r"latas?|tarros?|botes?|tazas?|vasos?|botellas?|cda(?:s)?|cucharadas?|cdta(?:s)?|cucharaditas?"
    r")\s+de\s+(?P<name>.+?)\s*$"
)
_RX_CLEAN_PREFIX = re.compile(
    r"(?i)^\s*(?:com[ií]|desayun[eé]|almorz[eé]|cen[eé]|meri[eé]nd[ae]|tom[eé]|beb[ií]"
    r"|he\s+comido|me\s+com[ií]|hoy\s+com[ií])\s*[:\-]?\s*"
)

# Drenado típico lata atún 170g/140g: ~155g escurrido rinde ~180 kcal a 116 kcal/100g
_DEFAULT_LATA_G_ATUN = 155.0


@dataclass(frozen=True)
class MacroPorcion:
    kcal: float
    p_g: float
    c_g: float
    g_g: float
    gramos: float
    nombre_alimento: str


class AlimentosDBService:
    """
    Resolver ingredientes contra Postgres:
      alimentos.nombre_normalizado, alimento_alias.alias_normalizado y alimento_unidades.
    """

    def __init__(self, db: Session):
        self.db = db

    def resolver_alimento_id(self, nombre: str) -> Optional[int]:
        n = _norm(nombre)
        if not n:
            return None
        a = (
            self.db.query(Alimento)
            .filter(Alimento.nombre_normalizado == n)
            .first()
        )
        if a:
            return int(a.id)
        al = (
            self.db.query(AlimentoAlias)
            .filter(AlimentoAlias.alias_normalizado == n)
            .first()
        )
        if al:
            return int(al.alimento_id)
        # Fallback simple por LIKE (prefijo) para UX; mantener determinista.
        like = f"{n}%"
        a2 = (
            self.db.query(Alimento)
            .filter(Alimento.nombre_normalizado.like(like))
            .order_by(Alimento.id.asc())
            .first()
        )
        return int(a2.id) if a2 else None

    def _resolver_alimento_id_lata(self, nombre: str) -> Optional[int]:
        """
        Para "lata de atún" priorizar la fila INS de atún en agua (evita mezclar con aceite/conserva
        según el orden del LIKE).
        """
        n = _norm(nombre)
        if not n:
            return None
        if "aceite" in n or "aceit" in n:
            row = (
                self.db.query(Alimento)
                .filter(Alimento.nombre_normalizado == _norm("pescado atún, enlatado en aceite"))
                .first()
            )
            return int(row.id) if row else self.resolver_alimento_id(nombre)
        if "agua" in n and "aceite" not in n:
            row = (
                self.db.query(Alimento)
                .filter(Alimento.nombre_normalizado == _norm("pescado atún, enlatado en agua"))
                .first()
            )
            return int(row.id) if row else self.resolver_alimento_id(nombre)
        if n in ("atun", "tuna") or re.search(r"\batun\b|\btuna\b", n):
            row = (
                self.db.query(Alimento)
                .filter(Alimento.nombre_normalizado == _norm("pescado atún, enlatado en agua"))
                .first()
            )
            if row:
                return int(row.id)
        return self.resolver_alimento_id(nombre)

    def _gramos_unidad_o_default_lata(self, alimento_id: int, unit: str) -> Optional[float]:
        g = self.gramos_por_unidad(alimento_id, unit)
        if g and g > 0:
            return float(g)
        u = _norm(unit)
        if u not in ("lata", "latas"):
            return None
        a = self.db.query(Alimento).filter(Alimento.id == alimento_id).first()
        if not a:
            return None
        an = (a.nombre or "").lower()
        if "atún" in an or "atun" in an or "tuna" in an:
            return float(_DEFAULT_LATA_G_ATUN)
        return 160.0

    def gramos_por_unidad(self, alimento_id: int, unidad: str) -> Optional[float]:
        u = _norm(unidad)
        if not u:
            return None

        # Normalizar plurales comunes / abreviaturas para que "tazas" matchee "taza", etc.
        alias_unit = {
            "tazas": "taza",
            "vasos": "vaso",
            "latas": "lata",
            "cucharadas": "cucharada",
            "cucharaditas": "cucharadita",
            "cdas": "cda",
            "cdtas": "cdta",
        }
        u2 = alias_unit.get(u, u)

        def _buscar(nombre_unit: str):
            return (
                self.db.query(AlimentoUnidad)
                .filter(AlimentoUnidad.alimento_id == alimento_id)
                .filter(AlimentoUnidad.nombre.ilike(nombre_unit))
                .first()
            )

        row = _buscar(u2)
        if not row and u2.endswith("s") and len(u2) > 3:
            # Fallback singular genérico: "tazas" -> "taza"
            row = _buscar(u2[:-1])
        if not row:
            # Fallback flexible: algunas BD guardan unidades como "taza cocida", "taza (arroz)".
            row = (
                self.db.query(AlimentoUnidad)
                .filter(AlimentoUnidad.alimento_id == alimento_id)
                .filter(AlimentoUnidad.nombre.ilike(f"%{u2}%"))
                .order_by(AlimentoUnidad.id.asc())
                .first()
            )
        if row and row.gramos and row.gramos > 0:
            return float(row.gramos)
        return None

    def macros_por_gramos(self, alimento_id: int, gramos: float) -> Optional[MacroPorcion]:
        if gramos <= 0:
            return None
        a = self.db.query(Alimento).filter(Alimento.id == alimento_id).first()
        if not a:
            return None
        factor = float(gramos) / 100.0
        return MacroPorcion(
            kcal=float(a.calorias_100g) * factor,
            p_g=float(a.proteina_100g) * factor,
            c_g=float(a.carbohidratos_100g) * factor,
            g_g=float(a.grasas_100g) * factor,
            gramos=float(gramos),
            nombre_alimento=str(a.nombre),
        )

    def parsear_ingrediente_a_porcion(self, linea: str) -> Optional[MacroPorcion]:
        """
        Soporta:
          - "100g arroz"
          - "1 taza de arroz" (si hay unidad en alimento_unidades)
        """
        t = (linea or "").strip()
        if not t:
            return None

        m = _RX_G.match(t)
        if m:
            gramos = float(m.group("num").replace(",", "."))
            name = m.group("name").strip()
            aid = self.resolver_alimento_id(name)
            return self.macros_por_gramos(aid, gramos) if aid else None

        m2 = _RX_UNIDAD.match(t)
        if m2:
            n = float(m2.group("num").replace(",", "."))
            unit = m2.group("unit").strip()
            name = m2.group("name").strip()
            is_lata = bool(re.search(r"(?i)lat\w", unit or ""))
            aid = self._resolver_alimento_id_lata(name) if is_lata else self.resolver_alimento_id(name)
            if not aid:
                return None
            g_un = (
                self._gramos_unidad_o_default_lata(aid, unit)
                if is_lata
                else self.gramos_por_unidad(aid, unit)
            )
            if not g_un:
                return None
            return self.macros_por_gramos(aid, n * g_un)

        m3 = _RX_UNIDAD_ES.match(t)
        if m3:
            qty = _parse_cantidad_token(m3.group("num"))
            unit = m3.group("unit").strip()
            name = m3.group("name").strip()
            is_lata = bool(re.search(r"(?i)lat\w", unit or ""))
            aid = self._resolver_alimento_id_lata(name) if is_lata else self.resolver_alimento_id(name)
            if not aid:
                return None
            g_un = (
                self._gramos_unidad_o_default_lata(aid, unit)
                if is_lata
                else self.gramos_por_unidad(aid, unit)
            )
            if not g_un:
                return None
            return self.macros_por_gramos(aid, qty * g_un)

        return None

    def extraer_porciones_desde_texto(self, texto: str) -> List[MacroPorcion]:
        """
        Extrae porciones desde texto libre, intentando mantener determinismo.
        Ejemplos:
          - "Comí 200g pollo y 1 taza de arroz"
          - "Cené 1 taza de arroz, 100g pollo"
        """
        t = (texto or "").strip()
        if not t:
            return []
        t = _RX_CLEAN_PREFIX.sub("", t).strip()
        # Separar por conectores simples.
        parts = re.split(r"(?i)\s+y\s+|,", t)
        out: List[MacroPorcion] = []
        for p in parts:
            p = p.strip()
            if not p:
                continue
            por = self.parsear_ingrediente_a_porcion(p)
            if por:
                out.append(por)
        return out



