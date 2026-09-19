"""
Persistencia de ejercicios — MET + workout_logs (código usado en producción).

Este módulo conserva EXCLUSIVAMENTE lo que el flujo real invoca:
  - log directo del Coach / endpoint                 (registrar_workout_log)
  - historial de logs del cliente                    (get_workout_logs)
  - inserción con ML sync                            (_registrar_workout_log_completo)
  - suma de kcal quemadas al progreso diario         (_sumar_calorias_progreso)
  - normalización de voz para NLP de alimentos       (_normalizar_voz)

NOTA (2026-09-05): se ELIMINÓ el flujo conversacional de follow-up
("¿cuántas series/reps hiciste?"). Se probó manualmente y NO funcionaba bien:
el turno 2 registraba "Ejercicio 3×10 @50.0kg (33 kcal)" — perdía el nombre del
ejercicio ("Press Banca") y calculaba kcal absurdas. Además nunca se invocaba
desde el chat real (que usa llm_registro.registrar_ejercicio_llm). El chat ahora
siempre estima con MET determinista del catálogo METS_GYM (ver llm_registro.py).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from sqlalchemy import text as _sql
from sqlalchemy.orm import Session

from app.core.utils import get_peru_date
from app.models.historial import ProgresoCalorias
from app.services.ejercicios_service import ejercicios_service

_RE_MULETILLAS = re.compile(
    r"(?i)\b(mm+h?|eeh?|aah?|uh+|uhm+|hmm+|o\s+sea(\s+que)?|como\s+que"
    r"|bueno\s+pues|pues\s+(?=\w)|la\s+verdad\s+(es\s+)?(que\s+)?"
    r"|y\s+este\s+|este\s+que\s+|este\s+(?=mm|este|o\s+sea|como\s+que))\b[,.]?\s*"
)
_RE_INICIO_MULETILLA = re.compile(r"(?i)^(este|pues|bueno|oye\s+pues)\s*[,.]?\s*")

_CENTENAS_MAP: dict[str, int] = {
    "cien": 100, "ciento": 100,
    "doscientos": 200, "doscientas": 200,
    "trescientos": 300, "trescientas": 300,
    "cuatrocientos": 400, "cuatrocientas": 400,
    "quinientos": 500, "quinientas": 500,
    "seiscientos": 600, "seiscientas": 600,
    "setecientos": 700, "setecientas": 700,
    "ochocientos": 800, "ochocientas": 800,
    "novecientos": 900, "novecientas": 900,
}
_DECENAS_MAP: dict[str, int] = {
    "veinte": 20, "veintiun": 21, "veintidos": 22, "veintitres": 23,
    "veinticuatro": 24, "veinticinco": 25, "veintiseis": 26,
    "veintisiete": 27, "veintiocho": 28, "veintinueve": 29,
    "treinta": 30, "cuarenta": 40, "cincuenta": 50,
    "sesenta": 60, "setenta": 70, "ochenta": 80, "noventa": 90,
}
_UNIDADES_MAP: dict[str, int] = {
    "cero": 0, "un": 1, "uno": 1, "una": 1,
    "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5,
    "seis": 6, "siete": 7, "ocho": 8, "nueve": 9,
    "diez": 10, "once": 11, "doce": 12, "trece": 13,
    "catorce": 14, "quince": 15, "dieciseis": 16,
    "diecisiete": 17, "dieciocho": 18, "diecinueve": 19,
}
_FRACCION_MAP: dict[str, str] = {
    "y medio": ".5", "y media": ".5",
    "y cuarto": ".25", "y tres cuartos": ".75",
}

_UNIDADES_STANDALONE = {k: v for k, v in _UNIDADES_MAP.items()
                         if k not in ("un", "uno", "una")}

_RE_NUMERO_COMPUESTO = re.compile(
    r"(?i)\b("
    + "|".join(sorted(_CENTENAS_MAP, key=len, reverse=True))
    + r")(?:\s+(" + "|".join(sorted(_DECENAS_MAP, key=len, reverse=True))
    + r"))?(?:\s+y\s+(" + "|".join(sorted(_UNIDADES_MAP, key=len, reverse=True))
    + r"))?(?:\s+(y\s+medi[ao]|y\s+cuarto|y\s+tres\s+cuartos))?\b"
    + r"|\b(" + "|".join(sorted(_DECENAS_MAP, key=len, reverse=True))
    + r")(?:\s+y\s+(" + "|".join(sorted(_UNIDADES_MAP, key=len, reverse=True))
    + r"))?(?:\s+(y\s+medi[ao]|y\s+cuarto|y\s+tres\s+cuartos))?\b"
    + r"|\b(" + "|".join(sorted(_UNIDADES_STANDALONE, key=len, reverse=True))
    + r")(?:\s+(y\s+medi[ao]|y\s+cuarto|y\s+tres\s+cuartos))?\b"
)


def _resolver_numero_compuesto(m: re.Match) -> str:
    """Convierte grupos capturados de número compuesto a dígito."""
    g = [x.lower().strip() if x else None for x in m.groups()]
    if g[0] and g[0] in {k.lower() for k in _CENTENAS_MAP}:
        val = _CENTENAS_MAP.get(g[0], 0)
        if g[1]:
            val += _DECENAS_MAP.get(g[1], 0)
        if g[2]:
            val += _UNIDADES_MAP.get(g[2], 0)
        frac = _FRACCION_MAP.get(g[3] or "", "")
        return str(val) + frac
    if g[4] and g[4] in {k.lower() for k in _DECENAS_MAP}:
        val = _DECENAS_MAP.get(g[4], 0)
        if g[5]:
            val += _UNIDADES_MAP.get(g[5], 0)
        frac = _FRACCION_MAP.get(g[6] or "", "")
        return str(val) + frac
    if g[7] and g[7] in {k.lower() for k in _UNIDADES_MAP}:
        val = _UNIDADES_MAP.get(g[7], 0)
        frac = _FRACCION_MAP.get(g[8] or "", "")
        return str(val) + frac
    return m.group(0)


def _normalizar_voz(texto: str) -> str:
    """
    Normaliza texto proveniente de voz o texto coloquial:
      1. Elimina muletillas iniciales (este, pues, bueno…)
      2. Elimina muletillas en el cuerpo (mmm, o sea, como que…)
      3. Convierte números hablados compuestos → dígitos
         'setenta y cinco kilos' → '75 kilos'
         'ciento veinte gramos'  → '120 gramos'
         'dos y medio'           → '2.5'
    """
    t = _RE_INICIO_MULETILLA.sub("", texto)
    t = _RE_MULETILLAS.sub(" ", t)
    t = _RE_NUMERO_COMPUESTO.sub(_resolver_numero_compuesto, t)
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t


def _met_a_intensity(met: float) -> str:
    if met >= 8.0:
        return "Alta"
    if met >= 5.0:
        return "Media"
    return "Baja"


class RegistroEjercicioHandler:
    """Persistencia de logs de entrenamiento (sin follow-up conversacional)."""


    def _registrar_workout_log_completo(
        self,
        client_id: int,
        ejercicio: str,
        series: int,
        reps: int,
        peso_kg: Optional[float],
        calorias_quemadas: float,
        session_duration_min: float,
        met: float,
        db: Session,
    ) -> None:
        """
        Inserta en workout_logs con los campos ML:
          calorias_quemadas = MET × peso_kg × 3.5 / 200 × minutos
          session_duration_min = tiempo de la sesión
          intensity = Baja/Media/Alta según MET
        """
        intensity = _met_a_intensity(met)
        db.execute(_sql(
            "INSERT INTO workout_logs "
            "(client_id, ejercicio, series, reps, peso_kg, created_at, "
            " calorias_quemadas, session_duration_min, intensity) "
            "VALUES (:cid, :ej, :se, :re, :pk, NOW(), :cal, :dur, :int)"
        ), {
            "cid": client_id,
            "ej":  ejercicio[:200],
            "se":  series,
            "re":  reps,
            "pk":  peso_kg,
            "cal": round(calorias_quemadas, 1),
            "dur": round(session_duration_min, 1),
            "int": intensity,
        })
        db.commit()

    def registrar_workout_log(
        self,
        client_id: int,
        ejercicio: str,
        series: int,
        reps: int,
        peso_kg: Optional[float],
        db: Session,
        met: float = 5.0,
        duracion_min: float = 45.0,
        peso_corporal_kg: float = 70.0,
    ) -> Dict[str, Any]:
        """
        API pública para el Coach o endpoint directo.
        Calcula automáticamente calorias_quemadas via MET.
        """
        cal = round(
            ejercicios_service.calcular_calorias(met, peso_corporal_kg, duracion_min), 1
        )
        self._registrar_workout_log_completo(
            client_id=client_id,
            ejercicio=ejercicio,
            series=series, reps=reps, peso_kg=peso_kg,
            calorias_quemadas=cal,
            session_duration_min=duracion_min,
            met=met,
            db=db,
        )
        self._sumar_calorias_progreso(client_id, cal, db)
        return {
            "success": True,
            "mensaje": f"Log guardado: {ejercicio} — {series}×{reps}"
                       + (f" @ {peso_kg} kg" if peso_kg else "")
                       + f" | {cal:.0f} kcal quemadas",
            "calorias_quemadas": cal,
            "intensity": _met_a_intensity(met),
        }

    def get_workout_logs(
        self, client_id: int, db: Session, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Devuelve los últimos logs de entrenamiento del cliente."""
        rows = db.execute(_sql(
            "SELECT ejercicio, series, reps, peso_kg, created_at, "
            "       calorias_quemadas, session_duration_min, intensity "
            "FROM workout_logs WHERE client_id = :cid "
            "ORDER BY created_at DESC LIMIT :lim"
        ), {"cid": client_id, "lim": limit}).fetchall()
        return [
            {
                "ejercicio": r[0], "series": r[1], "reps": r[2], "peso_kg": r[3],
                "created_at": str(r[4]), "calorias_quemadas": r[5],
                "session_duration_min": r[6], "intensity": r[7],
            }
            for r in rows
        ]

    def _sumar_calorias_progreso(self, client_id: int, cal: float, db: Session) -> None:
        """Suma calorías quemadas al registro de progreso del día."""
        hoy = get_peru_date()
        progreso = db.query(ProgresoCalorias).filter(
            ProgresoCalorias.client_id == client_id,
            ProgresoCalorias.fecha == hoy,
        ).first()
        if not progreso:
            progreso = ProgresoCalorias(client_id=client_id, fecha=hoy)
            db.add(progreso)
        progreso.calorias_quemadas = (progreso.calorias_quemadas or 0.0) + cal
        db.commit()


registro_ejercicio_handler = RegistroEjercicioHandler()