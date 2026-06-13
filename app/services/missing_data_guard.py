from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Optional

from app.services.asistente.asistente_modos import (
    RECOMENDAR_EJERCICIO,
    RECOMENDAR_NUTRICION,
    REGISTRAR_EJERCICIO,
    REGISTRAR_NUTRICION,
)


def strict_ask_missing_enabled() -> bool:
    return os.getenv("CALOFIT_STRICT_ASK_MISSING", "").strip().lower() in ("1", "true", "yes", "on")


def exercise_clarify_enabled() -> bool:
    """
    Repreguntas encadenadas para RECOMENDAR_EJERCICIO (zona, piernas, equipo, objetivo).
    Por defecto: desactivado (evita bucles y mensajes sueltos). Activar con CALOFIT_CLARIFY_EXERCISE=1.
    """
    ex = os.getenv("CALOFIT_CLARIFY_EXERCISE", "").strip().lower()
    if ex in ("1", "true", "yes", "on"):
        return True
    return False


_RX_CANTIDAD = re.compile(
    r"(?i)(\b\d+(?:[.,]\d+)?\s*(g|gr|gramos|ml|l|litro|kg|taza|vaso|botella|cda|cucharada|cdta|cucharadita|plato|porcion|porción|unidad|unid|lata|latas)\b"
    r"|\b(media|medio|un|una|uno|dos|tres|cuatro|cinco)\s+(?:lat\w+|taz\w+|vas\w+|botell\w+|cda(?:s)?|cucharad\w+|cdta(?:s)?))"
)

# Cantidad por conteo sin unidad: "2 hamburguesas", "media hamburguesa"
_RX_CANTIDAD_CONTEO = re.compile(
    r"(?i)\b(\d+(?:[.,]\d+)?|media|medio|un|una|uno|dos|tres|cuatro|cinco)\s+[a-záéíóúñ][a-záéíóúñ\s]{1,40}\b"
)
_RX_DURACION = re.compile(r"(?i)\b\d+\s*(min|mins|minutos|h|hr|hrs|hora|horas)\b")
_RX_SERIES = re.compile(r"(?i)\b\d+\s*x\s*\d+\b")  # 3x12
_RX_EQUIPO = re.compile(
    r"(?i)\b(casa|gym|gimnasio|mancuerna|mancuernas|barra|smith|polea|máquina|maquina|cinta|bici|trx)\b"
)
_RX_OBJETIVO = re.compile(
    r"(?i)\b(fuerza|hipertrofia|defin|tonificar|cardio|quemar|bajar|adelgaz|resistencia|movilidad|estir)\b"
)
_RX_FOCO_MUSCULAR = re.compile(
    r"(?i)\b(pecho|espalda|pierna|piernas|gluteo|glúteo|hombro|bicep|bícep|tricep|trícep|abdomen|core|brazos?)\b"
)
_RX_PIDE_EJERCICIO = re.compile(
    r"(?i)\b(ejercicio|ejercicios|rutina|entreno|entrenar|entrenamiento|gym|gimnasio)\b|\b(qué|que)\s+(hago|puedo\s+hacer)\b"
)


@dataclass(frozen=True)
class MissingInfo:
    missing_fields: List[str]
    question: str
    suggested_options: List[str]


def _tiene_foco_muscular(low: str) -> bool:
    return bool(_RX_FOCO_MUSCULAR.search(low))


def detectar_faltantes(modo_funcion: str, mensaje: str) -> Optional[MissingInfo]:
    """
    Detecta si faltan datos mínimos para REGISTRAR_*.
    Para recomendar (RECOMENDAR_*), no bloquea: solo sugiere si hace falta.
    """
    m = (mensaje or "").strip()
    low = m.lower()
    modo = (modo_funcion or "").strip().lower()

    if modo == REGISTRAR_NUTRICION:
        # Si menciona comida pero no hay cantidad → preguntar.
        if not (_RX_CANTIDAD.search(low) or _RX_CANTIDAD_CONTEO.search(low)):
            return MissingInfo(
                missing_fields=["cantidad"],
                question="Para registrarlo bien, ¿qué cantidad fue aproximadamente?",
                suggested_options=["Porción pequeña", "Porción mediana", "Porción grande", "100 g"],
            )
        return None

    if modo == REGISTRAR_EJERCICIO:
        # Si no hay duración ni series/reps → preguntar.
        if not (_RX_DURACION.search(low) or _RX_SERIES.search(low)):
            return MissingInfo(
                missing_fields=["duracion_o_series"],
                question="Para registrarlo bien, ¿cuánto tiempo o cuántas series/repeticiones hiciste?",
                suggested_options=["10 min", "20 min", "30 min", "3x12"],
            )
        return None

    if modo == RECOMENDAR_NUTRICION:
        # No bloquea; a lo mucho podríamos pedir "ligero vs completo", pero eso es UX.
        return None

    if modo == RECOMENDAR_EJERCICIO:
        return None

    return None


def detectar_faltantes_recomendar_ejercicio(mensaje: str) -> Optional[MissingInfo]:
    """
    Repreguntas antes de llamar al LLM cuando el usuario pide ejercicios pero falta contexto.
    """
    if not exercise_clarify_enabled():
        return None
    low = (mensaje or "").lower().strip()
    if len(low) < 3:
        return None

    # Pregunta genérica: pide ideas de entreno pero no dice zona muscular.
    if _RX_PIDE_EJERCICIO.search(low) and not _tiene_foco_muscular(low):
        return MissingInfo(
            missing_fields=["foco_muscular"],
            question="Para recomendarte bien: ¿qué zona o músculo quieres trabajar hoy?",
            suggested_options=["Pecho", "Espalda", "Piernas", "Hombros", "Brazos", "Core"],
        )

    # Piernas: distinguir frente vs atrás (cuádriceps vs femoral/glúteo posterior) si no está claro.
    if re.search(r"(?i)\bpierna|piernas|leg\b", low):
        tiene_frente = bool(
            re.search(
                r"(?i)\b(cuadriceps|cuádriceps|cuadri|vasto|recto femoral|femoral anterior|adelante|front)\b",
                low,
            )
        )
        tiene_atras = bool(
            re.search(
                r"(?i)\b(isquio|femoral posterior|femoral\s+posterior|gluteo|glúteo|posterior|atr[aá]s|gemelo|pantorrilla)\b",
                low,
            )
        )
        if not tiene_frente and not tiene_atras:
            return MissingInfo(
                missing_fields=["piernas_frente_atras"],
                question="Para piernas: ¿quieres priorizar la parte frontal (cuádriceps) o la posterior (isquios/glúteos)?",
                suggested_options=["Frontal (cuádriceps)", "Posterior (isquios/glúteos)", "Equilibrado (mixto)"],
            )

    # Equipo / lugar + objetivo (si falta alguno).
    if not _RX_EQUIPO.search(low):
        return MissingInfo(
            missing_fields=["equipo_lugar"],
            question="¿Dónde entrenas y con qué equipo cuentas?",
            suggested_options=["En casa sin pesas", "En casa con mancuernas", "En gym con máquinas", "Solo cinta/bici"],
        )
    if not _RX_OBJETIVO.search(low):
        return MissingInfo(
            missing_fields=["objetivo_entreno"],
            question="¿Cuál es tu objetivo hoy con ese entreno?",
            suggested_options=["Fuerza", "Hipertrofia", "Cardio/quemar", "Movilidad/recuperación"],
        )

    return None

