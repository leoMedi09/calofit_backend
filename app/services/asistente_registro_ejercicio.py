"""
Registro de ejercicios por NLP — Integración MET + workout_logs.

Capacidades:
  - Registro directo: "Hice 3 series de 10 reps de Press de Banca con 50kg"
  - Por referencia de rutina: "Hice la rutina Piernas de Acero"
  - Preguntas de seguimiento cuando faltan datos (peso/tiempo)
  - ML sync: workout_logs ← calorias_quemadas, session_duration_min, intensity
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from sqlalchemy import text as _sql
from sqlalchemy.orm import Session

from app.core.utils import get_peru_date
from app.models.historial import ProgresoCalorias
from app.services.asistente_ejercicio import (
    aplicar_extraccion_nlp_ejercicio_a_progreso,
    extraccion_ejercicio_fallback_fuerza,
    frase_registro_actividad_fisica,
    frase_vocabulario_gimnasio,
    parse_duracion_minutos,
    registrar_preferencias_ejercicios,
    resolver_met_mets_gym,
    rotulo_actividad_desde_mensaje,
)
from app.services.ejercicios_service import ejercicios_service

# ── Regexes de extracción ─────────────────────────────────────────────────────

# "3 series de 10 reps de Press de Banca con 50 kg"
_RE_SERIES_REPS = re.compile(
    r"(?i)(\d+)\s*(?:series?|sets?)\s*(?:de\s*)?(\d+)\s*(?:reps?|repeticiones?)",
)
# "con 50 kg", "a 50kg", "@ 50 kg"
_RE_PESO = re.compile(r"(?i)(?:con|a|@)\s*(\d+(?:[.,]\d+)?)\s*kg")

# "la rutina Piernas de Acero" o "Piernas de Acero" (nombre conocido)
_RE_RUTINA_REF = re.compile(
    r"(?i)(?:la\s+rutina\s+|rutina\s+|entrené\s+la\s+|hice\s+(?:la\s+)?rutina\s+)?"
    r"((?:piernas?\s+de\s+acero|pecho\s+explosivo|espalda\s+invencible|hombros?\s+de\s+tit[aá]n|"
    r"brazos?\s+de\s+tit[aá]n|core\s+blindado|n[uú]cleo\s+de\s+acero|gl[uú]teos?\s+de\s+fuego|"
    r"cardio\s+infernal|bestia\s+total|piernas?\s+de\s+acero|rutina\s+\w[\w\s]*?))"
)

# "hice la rutina …"
_RE_MENCIONA_RUTINA = re.compile(r"(?i)\b(?:hice|realicé|realice|terminé|termine)\b.{0,40}\brutina\b")

# Intensidad por MET
def _met_a_intensity(met: float) -> str:
    if met >= 8.0:
        return "Alta"
    if met >= 5.0:
        return "Media"
    return "Baja"


class RegistroEjercicioHandler:
    """Orquesta el registro de ejercicios por NLP y el log de entrenamiento."""

    # ── API pública ──────────────────────────────────────────────────────────

    async def registrar(
        self,
        mensaje: str,
        perfil,
        db: Session,
        ia_engine,
    ) -> Dict[str, Any]:
        """
        Detecta ejercicio en el mensaje y registra en progreso_calorias.
        Fórmula: MET × peso_kg × 3.5 / 200 × minutos
        """
        msg_lower = (mensaje or "").lower().strip()
        peso_kg   = float(getattr(perfil, "weight", None) or 70.0)

        # ── 1. Referencia a rutina nombrada ───────────────────────────────────
        if _RE_MENCIONA_RUTINA.search(msg_lower):
            return self._procesar_referencia_rutina(mensaje, msg_lower, perfil, db)

        # ── 2. Series/reps/peso explícitos ────────────────────────────────────
        series_match = _RE_SERIES_REPS.search(msg_lower)
        if series_match:
            return self._procesar_series_directo(mensaje, msg_lower, series_match, perfil, db)

        # ── 3. Flujo NLP estándar ─────────────────────────────────────────────
        if not (frase_registro_actividad_fisica(mensaje) or frase_vocabulario_gimnasio(mensaje)):
            return {
                "success": False, "tipo_detectado": "ninguno",
                "ejercicios": [], "datos": {},
                "mensaje": "No identifiqué ningún ejercicio en tu mensaje.",
            }

        extraccion = self._extraer_ejercicio_nlp(mensaje, msg_lower, peso_kg, ia_engine)
        if not extraccion or not extraccion.get("es_ejercicio"):
            return {
                "success": False, "tipo_detectado": "ninguno",
                "ejercicios": [], "datos": {},
                "mensaje": "No identifiqué ningún ejercicio en tu mensaje.",
            }

        hoy      = get_peru_date()
        progreso = db.query(ProgresoCalorias).filter(
            ProgresoCalorias.client_id == perfil.id, ProgresoCalorias.fecha == hoy
        ).first()
        if not progreso:
            progreso = ProgresoCalorias(client_id=perfil.id, fecha=hoy)
            db.add(progreso)

        aplicar_extraccion_nlp_ejercicio_a_progreso(extraccion, progreso)
        registrar_preferencias_ejercicios(extraccion, perfil, db)
        db.commit()

        # Sincronizar workout_log con ML fields
        dur_min = extraccion.get("duracion_min", 45.0)
        met     = extraccion.get("met", 5.0)
        cal     = extraccion.get("calorias", 0.0)
        nombres = extraccion.get("ejercicios_detectados", [])
        nombre_str = ", ".join(nombres) if nombres else "tu entrenamiento"

        self._registrar_workout_log_completo(
            client_id=perfil.id,
            ejercicio=nombre_str,
            series=1, reps=1, peso_kg=None,
            calorias_quemadas=cal,
            session_duration_min=dur_min,
            met=met,
            db=db,
        )

        return {
            "success": True, "tipo_detectado": "ejercicio",
            "ejercicios": nombres,
            "balance_actualizado": {
                "consumido": progreso.calorias_consumidas,
                "quemado":   progreso.calorias_quemadas,
            },
            "datos": {
                "calorias":     cal,
                "duracion_min": dur_min,
                "met":          met,
                "calidad":      extraccion.get("calidad_nutricional", "Alta"),
            },
            "mensaje": f"✅ Registré: {nombre_str} — {cal:.0f} kcal quemadas.",
        }

    # ── Procesadores especializados ──────────────────────────────────────────

    def _procesar_series_directo(
        self,
        _mensaje: str,
        msg_lower: str,
        series_match: re.Match,
        perfil,
        db: Session,
    ) -> Dict[str, Any]:
        """
        Maneja: "Hice 3 series de 10 reps de Press de Banca con 50 kg".
        Si falta el peso para ejercicios de fuerza, devuelve pregunta de seguimiento.
        """
        series = int(series_match.group(1))
        reps   = int(series_match.group(2))

        peso_match = _RE_PESO.search(msg_lower)
        peso_kg    = float(peso_match.group(1).replace(",", ".")) if peso_match else None

        # Extraer nombre del ejercicio
        ejercicio_nombre = self._extraer_nombre_ejercicio(msg_lower, series_match)
        clave, met = resolver_met_mets_gym(msg_lower)
        if not met:
            met = 5.0
        if not clave:
            clave = ejercicio_nombre or "ejercicio de fuerza"

        # Preguntar peso si se trata de ejercicio con peso y falta el dato
        es_fuerza = any(t in msg_lower for t in [
            "press", "curl", "remo", "sentad", "peso muerto", "jalón", "extensi"
        ])
        if es_fuerza and peso_kg is None:
            return self._pregunta_faltante(
                campo="peso",
                ejercicio=ejercicio_nombre or clave,
                series=series,
                reps=reps,
                contexto=msg_lower,
            )

        # Calcular calorías: usar tiempo estimado (series × reps × 3 seg + descanso)
        seg_totales  = series * (reps * 3 + 60)
        dur_min      = seg_totales / 60
        peso_corporal = float(getattr(perfil, "weight", None) or 70.0)
        cal = round(ejercicios_service.calcular_calorias(met, peso_corporal, dur_min), 1)

        # Persistir en workout_logs y progreso_calorias
        self._registrar_workout_log_completo(
            client_id=perfil.id,
            ejercicio=ejercicio_nombre or clave,
            series=series, reps=reps, peso_kg=peso_kg,
            calorias_quemadas=cal,
            session_duration_min=dur_min,
            met=met,
            db=db,
        )
        self._sumar_calorias_progreso(perfil.id, cal, db)

        detalle_peso = f" con {peso_kg} kg" if peso_kg else ""
        return {
            "success": True,
            "tipo_detectado": "ejercicio_series",
            "ejercicios": [ejercicio_nombre or clave],
            "datos": {
                "series": series, "reps": reps, "peso_kg": peso_kg,
                "calorias": cal, "duracion_min": round(dur_min, 1), "met": met,
            },
            "mensaje": (
                f"✅ Registré: {ejercicio_nombre or clave} — "
                f"{series}×{reps}{detalle_peso} → {cal:.0f} kcal quemadas."
            ),
        }

    def _procesar_referencia_rutina(
        self,
        _mensaje: str,
        msg_lower: str,
        perfil,
        db: Session,
    ) -> Dict[str, Any]:
        """
        Maneja: "Hice la rutina Piernas de Acero".
        Carga los ejercicios de la rutina del caché o BD, registra la sesión completa.
        """
        # Buscar si la rutina está en caché (generada previamente)
        rutina_nombre = self._extraer_nombre_rutina(msg_lower)

        # Registro de la sesión como actividad completa
        # Por ahora, usa MET de piernas/pecho según nombre
        met_rutina = self._met_desde_nombre_rutina(rutina_nombre)
        dur_min    = parse_duracion_minutos(msg_lower, default=60.0)
        peso_corporal = float(getattr(perfil, "weight", None) or 70.0)
        cal = round(ejercicios_service.calcular_calorias(met_rutina, peso_corporal, dur_min), 1)

        self._registrar_workout_log_completo(
            client_id=perfil.id,
            ejercicio=f"Rutina: {rutina_nombre}",
            series=1, reps=1, peso_kg=None,
            calorias_quemadas=cal,
            session_duration_min=dur_min,
            met=met_rutina,
            db=db,
        )
        self._sumar_calorias_progreso(perfil.id, cal, db)

        return {
            "success": True,
            "tipo_detectado": "rutina_referencia",
            "ejercicios": [f"Rutina: {rutina_nombre}"],
            "datos": {
                "calorias": cal,
                "duracion_min": dur_min,
                "met": met_rutina,
            },
            "mensaje": (
                f"🏋️ ¡Excelente! Registré la rutina «{rutina_nombre}» — "
                f"{cal:.0f} kcal quemadas en ~{dur_min:.0f} min. "
                f"¿Quieres ajustar algún ejercicio o añadir el peso utilizado?"
            ),
        }

    def _pregunta_faltante(
        self,
        campo: str,
        ejercicio: str,
        series: int,
        reps: int,
        contexto: str = "",  # noqa: ARG002
    ) -> Dict[str, Any]:
        """Genera pregunta de seguimiento cuando falta un dato obligatorio."""
        preguntas = {
            "peso": f"¡Excelente trabajo! {series}×{reps} de {ejercicio} está registrado. "
                    f"¿Pero qué peso utilizaste para completar esas series?",
            "tiempo": f"¡Muy bien con el {ejercicio}! ¿Cuántos minutos duró tu sesión?",
            "series": f"¿Cuántas series hiciste de {ejercicio}?",
        }
        return {
            "success": False,
            "tipo_detectado": "datos_incompletos",
            "campo_faltante": campo,
            "ejercicio_detectado": ejercicio,
            "series_detectadas": series,
            "reps_detectadas": reps,
            "mensaje": preguntas.get(campo, f"¿Podrías darme más detalles sobre tu {ejercicio}?"),
            "requiere_followup": True,
        }

    # ── workout_logs con ML sync ─────────────────────────────────────────────

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

    # ── Helpers privados ─────────────────────────────────────────────────────

    def _extraer_nombre_ejercicio(self, msg_lower: str, series_match: re.Match) -> str:
        """Extrae el nombre del ejercicio después de 'reps de X'."""
        despues = msg_lower[series_match.end():]
        m = re.search(r"(?i)(?:de\s+)(.+?)(?:\s+con\s+|\s+a\s+\d|\s*$)", despues)
        if m:
            return m.group(1).strip().title()[:80]
        return ""

    def _extraer_nombre_rutina(self, msg_lower: str) -> str:
        """Extrae el nombre de la rutina mencionada."""
        m = _RE_RUTINA_REF.search(msg_lower)
        if m:
            return m.group(1).strip().title()
        # Fallback: tomar palabras después de "rutina"
        m2 = re.search(r"rutina\s+(.+?)(?:\s*\.|\s*$)", msg_lower)
        if m2:
            return m2.group(1).strip().title()[:50]
        return "Completa"

    def _met_desde_nombre_rutina(self, nombre: str) -> float:
        """Estima MET de la rutina según el nombre."""
        n = nombre.lower()
        if any(k in n for k in ["pierna", "acero", "tren inferior"]):
            return 5.5
        if any(k in n for k in ["pecho", "explosivo"]):
            return 5.0
        if any(k in n for k in ["cardio", "infernal", "hiit"]):
            return 9.0
        if any(k in n for k in ["bestia", "total", "full", "completo"]):
            return 7.0
        return 5.0

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

    def _extraer_ejercicio_nlp(
        self, mensaje: str, msg_lower: str, peso_kg: float, _ia_engine
    ) -> Optional[Dict[str, Any]]:
        """Detecta ejercicio con MET desde diccionario interno o fallback fuerza."""
        if not (
            frase_registro_actividad_fisica(mensaje)
            or frase_vocabulario_gimnasio(mensaje)
        ):
            return None

        clave, met = resolver_met_mets_gym(msg_lower)

        if met and met > 0:
            dur    = parse_duracion_minutos(msg_lower, default=45.0)
            cal    = round(ejercicios_service.calcular_calorias(met, peso_kg, dur), 1)
            rotulo = rotulo_actividad_desde_mensaje(mensaje)
            return {
                "es_ejercicio": True, "es_comida": False,
                "calorias": cal, "proteinas_g": 0, "carbohidratos_g": 0,
                "grasas_g": 0, "fibra_g": 0, "azucar_g": 0, "sodio_mg": 0,
                "ejercicios_detectados": [f"{rotulo} ({dur:.0f} min)"],
                "alimentos_detectados": [],
                "calidad_nutricional": "Alta",
                "duracion_min": dur, "met": met,
                "origen": f"MET gym ({clave})",
            }

        if frase_vocabulario_gimnasio(mensaje):
            return extraccion_ejercicio_fallback_fuerza(mensaje, msg_lower, peso_kg)

        return None


registro_ejercicio_handler = RegistroEjercicioHandler()
