"""
Entrenador Adaptativo — CaloFit.

Genera rutinas personalizadas basadas en:
  - Perfil A/B/C (Random Forest)
  - Lesiones en medical_conditions
  - Zonas objetivo y tiempo disponible
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import text as _sql
from sqlalchemy.orm import Session

from app.services.asistente_recomendaciones import RecomendacionesHandler

# ── Configuración de sustituciones por lesión ──────────────────────────────────

_LESIONES_SUSTITUCION: Dict[str, Dict[str, Any]] = {
    "rodilla": {
        "keywords": ["rodilla", "menisco", "ligamento cruzado", "rótula"],
        "grupos_restringidos": ["Piernas"],
        "sustituir": {
            "sentadilla": ("extensiones_cuadriceps_maquina", "Extensiones de Cuádriceps (sin impacto)"),
            "prensa":     ("extensiones_cuadriceps_maquina", "Extensiones de Cuádriceps (sin impacto)"),
            "default":    ("curl_femoral_maquina",           "Curl Femoral en Máquina (bajo impacto)"),
        },
        "justificacion": "lesión de rodilla — se sustituyeron ejercicios de alto impacto por variantes de máquina sin compresión articular",
    },
    "espalda": {
        "keywords": ["espalda", "lumbar", "hernia", "discopatía", "lumbalgia", "ciática"],
        "grupos_restringidos": ["Espalda", "Piernas"],
        "sustituir": {
            "peso muerto": ("face_pull",    "Face Pull (sin carga lumbar)"),
            "remo":        ("remo_en_polea_baja", "Remo Polea (sin inclinación libre)"),
            "default":     ("dead_bug",     "Dead Bug (fortalece core sin comprimir columna)"),
        },
        "justificacion": "lesión de espalda/lumbar — se sustituyeron cargas libres por variantes de máquina o ejercicios de estabilización",
    },
    "hombro": {
        "keywords": ["hombro", "manguito rotador", "supraespinoso", "infraespinoso", "luxación"],
        "grupos_restringidos": ["Hombros", "Pecho"],
        "sustituir": {
            "press": ("face_pull",           "Face Pull (rehabilitación manguito rotador)"),
            "vuelo": ("pajaros_inclinado",   "Pájaros con peso ligero — solo si sin dolor"),
            "default": ("face_pull",         "Face Pull (movilidad y fuerza sin impacto)"),
        },
        "justificacion": "lesión de hombro — se eliminaron ejercicios de press y se priorizó la rehabilitación del manguito rotador",
    },
    "codo": {
        "keywords": ["codo", "epicóndilo", "codo de tenista", "codo de golfista", "tendinitis codo"],
        "grupos_restringidos": ["Bíceps", "Tríceps"],
        "sustituir": {
            "default": ("face_pull", "Face Pull (isométrico suave)"),
        },
        "justificacion": "lesión de codo — se redujeron ejercicios de flexo-extensión de codo y se sugieren cargas mínimas",
    },
}

# ── Nombres creativos por zona ─────────────────────────────────────────────────

_NOMBRES_RUTINA: Dict[str, List[str]] = {
    "Piernas":         ["Piernas de Acero", "Tormenta de Cuádriceps", "Rey de Sentadillas", "Leyenda del Tren Inferior"],
    "Pecho":           ["Pecho Explosivo", "Escudo de Titanio", "Pectorales de Campeón", "Fuerza de Impacto"],
    "Espalda":         ["Espalda Invencible", "Alas de Águila", "Columna de Fuego", "Fortaleza Posterior"],
    "Hombros":         ["Hombros de Titán", "Montaña de Deltoides", "Brazos al Cielo", "Cannonball Shoulders"],
    "Bíceps":          ["Brazos de Titán", "Curls de Hierro", "Bíceps Supremo", "Fuerza en Flexión"],
    "Tríceps":         ["Tríceps de Élite", "Empuje Mortal", "Brazos de Acero", "Triple Amenaza"],
    "Core":            ["Núcleo de Acero", "Core Blindado", "Centro de Poder", "Abdomen Indestructible"],
    "Glúteos":         ["Glúteos de Fuego", "Power Booty", "Activación Total", "Caderas Explosivas"],
    "Cardio":          ["Cardio Infernal", "Quema Máxima", "Resistencia Extrema", "Turbo Cardio"],
    "Cuerpo Completo": ["Bestia Total", "Full Body Extremo", "Guerrero Completo", "Máquina Humana"],
}

# ── Series/reps por perfil ─────────────────────────────────────────────────────

_CONFIG_PERFIL: Dict[str, Dict[str, Any]] = {
    "PERFIL_A": {
        "series": 4, "reps": 10, "descanso_seg": 60,
        "intensidad": "Alta", "nivel_filtro": ["Avanzado", "Intermedio", "Principiante"],
    },
    "PERFIL_B": {
        "series": 3, "reps": 12, "descanso_seg": 90,
        "intensidad": "Media", "nivel_filtro": ["Intermedio", "Principiante"],
    },
    "PERFIL_C": {
        "series": 3, "reps": 15, "descanso_seg": 120,
        "intensidad": "Baja-Media", "nivel_filtro": ["Principiante"],
    },
}

# ── Mapeo workout_type → zonas musculares ──────────────────────────────────────
# Claves: primera palabra del campo workout_type en minúsculas.
# "Fuerza (Pesas, Gym)" → "fuerza" → zonas de fuerza.
_WORKOUT_TYPE_TO_ZONES: Dict[str, List[str]] = {
    # Español (valores del dropdown Flutter)
    "fuerza":    ["Pecho", "Espalda", "Hombros", "Piernas"],
    "cardio":    ["Cardio"],
    "hiit":      ["Cardio", "Core", "Piernas"],
    "funcional": ["Cardio", "Core", "Piernas"],
    "yoga":      ["Core", "Glúteos", "Piernas"],
    "pilates":   ["Core", "Glúteos", "Piernas"],
    "mixto":     ["Pecho", "Espalda", "Cardio", "Core"],
    # Inglés (valores guardados por el pipeline ML — _WT_MAP en asistente_recomendaciones.py)
    "strength":  ["Pecho", "Espalda", "Hombros", "Piernas"],
}


def zonas_desde_workout_type(workout_type: Optional[str]) -> List[str]:
    """Mapea workout_type del perfil a zonas para generar_rutina_inteligente()."""
    if not workout_type:
        return ["Cuerpo Completo"]
    primera = workout_type.strip().lower().split()[0]
    return _WORKOUT_TYPE_TO_ZONES.get(primera, ["Cuerpo Completo"])


# ── Calculadora de ejercicios por tiempo ──────────────────────────────────────

def _ejercicios_por_tiempo(tiempo_min: int, series: int, reps: int, descanso_seg: int) -> int:
    """Estima cuántos ejercicios caben en el tiempo disponible."""
    seg_por_rep = 3
    seg_por_set = reps * seg_por_rep + descanso_seg
    seg_por_ejercicio = series * seg_por_set + 30  # 30s transición
    return max(2, min(8, tiempo_min * 60 // seg_por_ejercicio))


def _detectar_lesiones(medical_conditions: List[str]) -> List[str]:
    """Devuelve lista de claves de lesión activas."""
    texto = " ".join(c.lower() for c in (medical_conditions or []))
    activas = []
    for clave, cfg in _LESIONES_SUSTITUCION.items():
        for kw in cfg["keywords"]:
            if kw in texto:
                activas.append(clave)
                break
    return activas


def _grupos_restringidos(lesiones_activas: List[str]) -> List[str]:
    restringidos = []
    for lesion in lesiones_activas:
        restringidos.extend(_LESIONES_SUSTITUCION[lesion]["grupos_restringidos"])
    return list(set(restringidos))


def _sustituir_ejercicio(
    ejercicio_id: str,
    ejercicio_nombre: str,
    lesion: str,
    db: Session,
) -> Tuple[str, str, str]:
    """
    Retorna (nuevo_id, nuevo_nombre, justificacion).
    Busca la clave de sustitución que mejor coincide con el nombre del ejercicio.
    """
    sustit = _LESIONES_SUSTITUCION[lesion]["sustituir"]
    just   = _LESIONES_SUSTITUCION[lesion]["justificacion"]
    nombre_lower = ejercicio_nombre.lower()

    for patron, (nuevo_id, nuevo_nombre) in sustit.items():
        if patron != "default" and patron in nombre_lower:
            return nuevo_id, nuevo_nombre, just

    nuevo_id, nuevo_nombre = sustit["default"]
    return nuevo_id, nuevo_nombre, just


# Tipos que NO pertenecen a una rutina de gym/pesas — siempre excluidos.
_TIPOS_EXCLUIDOS_BASE: List[str] = ["Strongman", "Cardio Ligero"]

# Tipos extra excluidos según el workout_type del perfil.
_TIPOS_EXCLUIDOS_POR_WORKOUT: Dict[str, List[str]] = {
    "fuerza":    ["Cardio", "Metabólico/HIIT"],
    "strength":  ["Cardio", "Metabólico/HIIT"],
    "cardio":    ["Halterofilia", "Powerlifting"],
    "hiit":      ["Powerlifting"],
    "funcional": ["Powerlifting"],
}


def _consultar_ejercicios(
    zonas: List[str],
    nivel_filtro: List[str],
    n: int,
    db: Session,
    tipos_excluidos: Optional[List[str]] = None,
) -> List[Dict]:
    """Consulta ejercicios filtrando por grupo_padre, nivel y tipos inapropiados."""
    if not zonas:
        return []

    todos_excluidos = list(_TIPOS_EXCLUIDOS_BASE) + list(tipos_excluidos or [])

    placeholders_zonas   = ", ".join(f":z{i}" for i in range(len(zonas)))
    placeholders_niveles = ", ".join(f":n{i}" for i in range(len(nivel_filtro)))
    placeholders_excl    = ", ".join(f":x{i}" for i in range(len(todos_excluidos)))

    params = {f"z{i}": z for i, z in enumerate(zonas)}
    params.update({f"n{i}": nv for i, nv in enumerate(nivel_filtro)})
    params.update({f"x{i}": t  for i, t  in enumerate(todos_excluidos)})
    params["lim"] = n

    rows = db.execute(_sql(f"""
        SELECT id, nombre, musculo_principal, tipo, nivel, met, tecnica, tipo_metrica, grupo_padre
        FROM ejercicios
        WHERE grupo_padre IN ({placeholders_zonas})
          AND nivel       IN ({placeholders_niveles})
          AND tipo        NOT IN ({placeholders_excl})
          AND id NOT IN (
            'huayno_zapateo', 'pichanga_futbol', 'voley_calle',
            'trote_ligero', 'subir_escaleras_cerro'
          )
        ORDER BY RANDOM()
        LIMIT :lim
    """), params).fetchall()

    return [
        {
            "id": r[0], "nombre": r[1], "musculo_principal": r[2],
            "tipo": r[3], "nivel": r[4], "met": r[5],
            "instrucciones": r[6] or "", "tipo_metrica": r[7] or "peso_reps",
            "grupo_padre": r[8],
        }
        for r in rows
    ]


def _nombre_rutina(zonas: List[str], perfil: str, lesiones: List[str]) -> str:
    zona_principal = zonas[0] if zonas else "Cuerpo Completo"
    opciones = _NOMBRES_RUTINA.get(zona_principal, _NOMBRES_RUTINA["Cuerpo Completo"])
    idx = hash(perfil + zona_principal) % len(opciones)
    nombre = opciones[idx]
    if lesiones:
        nombre += " (Adaptada)"
    return nombre


# ── API pública ────────────────────────────────────────────────────────────────

async def generar_rutina_inteligente(
    user_id: int,
    zonas_objetivo: List[str],
    tiempo_min: int,
    db: Session,
) -> Dict[str, Any]:
    """
    Genera una rutina personalizada según perfil ML, lesiones y zonas objetivo.

    Returns:
        {
          "nombre_rutina": str,
          "perfil": str,
          "confianza_perfil": float,
          "ejercicios": [...],
          "series": int,
          "reps": int,
          "descanso_seg": int,
          "intensidad": str,
          "lesiones_detectadas": [...],
          "advertencias": [...],
          "tiempo_estimado_min": int,
        }
    """
    from app.models.client import Client

    perfil_obj = db.query(Client).filter(Client.id == user_id).first()
    if not perfil_obj:
        return {"error": f"Usuario {user_id} no encontrado"}

    # Clasificar perfil A/B/C
    handler = RecomendacionesHandler()
    perfil_str, confianza = handler.predecir_perfil(perfil_obj, db)
    cfg = _CONFIG_PERFIL.get(perfil_str, _CONFIG_PERFIL["PERFIL_B"])

    # Detectar lesiones
    conditions = list(perfil_obj.medical_conditions or [])
    lesiones_activas = _detectar_lesiones(conditions)
    grupos_bloqueados = _grupos_restringidos(lesiones_activas)

    # Filtrar zonas seguras
    zonas_seguras    = [z for z in zonas_objetivo if z not in grupos_bloqueados]
    zonas_restringidas = [z for z in zonas_objetivo if z in grupos_bloqueados]

    n_ejercicios = _ejercicios_por_tiempo(tiempo_min, cfg["series"], cfg["reps"], cfg["descanso_seg"])

    # Tipos excluidos según el workout_type del perfil
    _wt_primera  = (perfil_obj.workout_type or "").strip().lower().split()[0] if perfil_obj.workout_type else ""
    _tipos_extra = _TIPOS_EXCLUIDOS_POR_WORKOUT.get(_wt_primera, [])

    # Distribuir ejercicios entre zonas seguras
    if zonas_seguras:
        n_por_zona = max(1, n_ejercicios // len(zonas_seguras))
        ejercicios_raw = []
        for zona in zonas_seguras:
            ejercicios_raw.extend(
                _consultar_ejercicios([zona], cfg["nivel_filtro"], n_por_zona, db, tipos_excluidos=_tipos_extra)
            )
    else:
        # Todas las zonas tienen lesión → dar rutina de core o upper-body seguro
        ejercicios_raw = _consultar_ejercicios(["Core"], cfg["nivel_filtro"], n_ejercicios, db, tipos_excluidos=_tipos_extra)

    # Aplicar sustituciones para zonas restringidas
    advertencias = []
    sustituciones_aplicadas = []

    for zona in zonas_restringidas:
        for lesion in lesiones_activas:
            if zona in _LESIONES_SUSTITUCION[lesion]["grupos_restringidos"]:
                just = _LESIONES_SUSTITUCION[lesion]["justificacion"]
                # Añadir 1-2 ejercicios de sustitución
                nuevo_id, nuevo_nombre = list(
                    _LESIONES_SUSTITUCION[lesion]["sustituir"].values()
                )[-1]  # default
                row = db.execute(_sql(
                    "SELECT id, nombre, musculo_principal, tipo, nivel, met, tecnica, tipo_metrica, grupo_padre "
                    "FROM ejercicios WHERE id = :eid LIMIT 1"
                ), {"eid": nuevo_id}).fetchone()

                if row:
                    sustituciones_aplicadas.append({
                        "id": row[0], "nombre": row[1], "musculo_principal": row[2],
                        "tipo": row[3], "nivel": row[4], "met": row[5],
                        "instrucciones": row[6] or "", "tipo_metrica": row[7] or "peso_reps",
                        "grupo_padre": row[8], "es_sustitucion": True,
                    })
                advertencias.append(
                    f"⚠ Zona '{zona}' restringida por {just}. "
                    f"Se añadió '{nuevo_nombre}' como alternativa segura."
                )
                break

    ejercicios_final = ejercicios_raw[:n_ejercicios] + sustituciones_aplicadas

    # Calcular tiempo estimado
    seg_por_ejercicio = (
        cfg["series"] * (cfg["reps"] * 3 + cfg["descanso_seg"]) + 30
    )
    tiempo_estimado = round(len(ejercicios_final) * seg_por_ejercicio / 60)

    return {
        "nombre_rutina":        _nombre_rutina(zonas_objetivo, perfil_str, lesiones_activas),
        "perfil":               perfil_str,
        "confianza_perfil":     round(confianza, 1),
        "ejercicios":           ejercicios_final,
        "series":               cfg["series"],
        "reps":                 cfg["reps"],
        "descanso_seg":         cfg["descanso_seg"],
        "intensidad":           cfg["intensidad"],
        "lesiones_detectadas":  lesiones_activas,
        "zonas_solicitadas":    zonas_objetivo,
        "zonas_seguras":        zonas_seguras,
        "zonas_restringidas":   zonas_restringidas,
        "advertencias":         advertencias,
        "tiempo_estimado_min":  tiempo_estimado,
    }
