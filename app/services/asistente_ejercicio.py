"""
Lógica de ejercicio del asistente (preferencias, detección de payload de rutina).

Separado de ``asistente_nutricion.py`` para poder cambiar entrenamiento sin tocar comidas.
"""
from __future__ import annotations

import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import func as sql_func
from sqlalchemy.orm import Session

from app.core.cache import set_consulta_cached
from app.core.mets_gym import METS_GYM
from app.models.preferencias import PreferenciaEjercicio
from app.services.ejercicios_service import ejercicios_service


def parse_duracion_minutos(texto: str, default: float = 45.0) -> float:
    """
    Extrae minutos de frases tipo «30 min», «45 minutos», «1 hora», «1h».
    Si no hay unidad reconocible, devuelve ``default``.
    """
    if not texto:
        return default
    t = texto.lower()
    match_dur = re.search(r"(\d+)\s*(min(?:utos?)?|h(?:oras?|rs?)?)", t)
    if not match_dur:
        return default
    valor = int(match_dur.group(1))
    unidad = match_dur.group(2)
    return float(valor * 60 if unidad.startswith("h") else valor)


def resolver_met_mets_gym(texto_lower: str) -> Tuple[Optional[str], Optional[float]]:
    """Mayor coincidencia por longitud de clave (evita que «press» gane a «press banca»)."""
    best_key: Optional[str] = None
    best_met: Optional[float] = None
    best_len = -1
    for key, met in sorted(METS_GYM.items(), key=lambda x: -len(x[0])):
        if key in texto_lower and len(key) > best_len:
            best_len = len(key)
            best_key = key
            best_met = float(met)
    return best_key, best_met


_RE_HABITO_EJERCICIO = re.compile(
    r"(?i)\b(hice|realicé|realice|entren[ée]|entrené|entrene|corrí|corri|trot[ée]|trote"
    r"|camin[ée]|camine|practiqué|practique|nadé|nade|subí|subi|pedale|remé|boxe)\b"
)
_RE_TIEMPO_O_VOLUMEN = re.compile(
    r"(?i)(?:\d+\s*(?:min(?:utos?)?|h(?:ora)?s?|h\b)|\bseries\b|\brep(?:es|eticiones)?\b|\bmin\b|\bkm\b)"
)
_RE_VERBO_COMIDA = re.compile(
    r"(?i)\b(comí|comi|desayun|almorc|almorz|cené|cene|bebí|bebi|ingerí|ingeri|tragué|trag|"
    r"comí\w*|desayun\w*)\b"
)
_RE_LEX_GYM = re.compile(
    r"(?i)(pierna|brazos?|pecho|espalda|core|abdom|press|curl|remo|sentad|extens|elev|femoral|"
    r"cu[aá]driceps|gemelo|gl[uú]te|hiit|cardio|trot|corr|burpee|flexi|dominad|"
    r"jal[oó]n|pesas|mancuern|barra|gym|m[aá]quina|bici|el[ií]ptic|soga|saltos?|zancad|"
    r"rowing|adduct|abd\w*|bis\w*|tris\w*|hip\s*thrust|kettle|snatch|clean|thruster|"
    r"ejercicio|entreno|fuerza|aer[óo]bic)"
    r"|(?<!a la )(?<!la )\bplancha\b"  # 'plancha' solo si NO es 'a la plancha' (cocción)
)


def frase_registro_actividad_fisica(texto: str) -> bool:
    """'Hice X … min' sin verbos de comida: priorizar gasto calórico, no macros."""
    t = texto.lower()
    if not _RE_HABITO_EJERCICIO.search(t):
        return False
    if _RE_VERBO_COMIDA.search(t):
        return False
    if not _RE_TIEMPO_O_VOLUMEN.search(t):
        return False
    return True


def frase_vocabulario_gimnasio(texto: str) -> bool:
    return bool(_RE_LEX_GYM.search(texto))


def rotulo_actividad_desde_mensaje(texto: str) -> str:
    t = texto.strip()
    m = re.search(
        r"(?i)(?:\b(hice|realicé|realice|entren[ée]|entrené|entrene|practiqué|practique)\s+)(.+?)"
        r"(?=\s+por\s+\d|\s+\d+\s*min|\s+\d+\s*minut|$)",
        t,
    )
    if m:
        s = m.group(2).strip(" .,;—-")
        if len(s) >= 2:
            return s[:120]
    m2 = re.search(r"(?i)(?:\b(corrí|corri|trot[ée]|trote|nadé|nade|camin[ée]|camine)\s+)(.+?)(?=\s+\d+\s*min|\s+por|\s+\d+\s*km|$)", t)
    if m2:
        s = m2.group(2).strip(" .,;—-")
        if len(s) >= 2:
            return s[:120]
    return t[:120]


def extraccion_ejercicio_fallback_fuerza(
    texto_original: str, texto_lower: str, peso_kg: float, met_default: float = 5.0
) -> Dict[str, Any]:
    """Fuerza/cardio general cuando el texto es claramente entreno pero no hubo clave MET."""
    dur = parse_duracion_minutos(texto_lower, default=45.0)
    cal = round(ejercicios_service.calcular_calorias(met_default, peso_kg, dur), 1)
    rotulo = rotulo_actividad_desde_mensaje(texto_original)
    return {
        "es_ejercicio": True,
        "es_comida": False,
        "calorias": cal,
        "proteinas_g": 0,
        "carbohidratos_g": 0,
        "grasas_g": 0,
        "fibra_g": 0,
        "azucar_g": 0,
        "sodio_mg": 0,
        "ejercicios_detectados": [f"{rotulo} ({dur:.0f} min)"],
        "alimentos_detectados": [],
        "calidad_nutricional": "Alta",
        "duracion_min": dur,
        "met": met_default,
        "origen": "Fallback gimnasio (MET genérico)",
    }


def procesar_secciones_ejercicio(respuesta_estructurada: Dict[str, Any], perfil: Any) -> None:
    """
    Recalcula kcal de tarjetas POWER con la misma fórmula MET que el registro por texto,
    cachea ``consulta_id`` y alinea ``macros`` con lo que verá el usuario al confirmar.
    """
    secciones: List[Dict[str, Any]] = respuesta_estructurada.get("secciones") or []
    if not isinstance(secciones, list):
        return

    peso = float(getattr(perfil, "weight", None) or 70.0)

    for sec in secciones:
        if not isinstance(sec, dict) or sec.get("tipo") != "ejercicio":
            continue

        partes = [
            str(sec.get("nombre") or ""),
            " ".join(sec.get("ejercicios") or []),
            str(sec.get("macros") or ""),
            str(sec.get("gasto_calorico_estimado") or ""),
        ]
        haystack = " ".join(partes).lower()
        _, met = resolver_met_mets_gym(haystack)
        if met is None or met <= 0:
            # Tarjeta POWER ya es ejercicio; a veces el título no coincide con claves MET
            met = 5.0

        dur = parse_duracion_minutos(haystack, default=45.0)
        kcal = round(ejercicios_service.calcular_calorias(met, peso, dur), 1)
        nombre_tarjeta = (sec.get("nombre") or "Entrenamiento").strip()

        stats_line = (
            f"Cal: {kcal} kcal | Dur: {dur:.0f} min | MET ~{met}"
        )
        sec["macros"] = stats_line
        sec["gasto_calorico_estimado"] = stats_line
        sec["macros_normalizados"] = {
            "kcal": kcal,
            "proteinas_g": 0.0,
            "carbohidratos_g": 0.0,
            "grasas_g": 0.0,
        }

        consulta_id = str(uuid.uuid4())
        ej_list = sec.get("ejercicios") or []
        if not ej_list:
            ej_list = [nombre_tarjeta]
        payload = {
            "nombre": nombre_tarjeta,
            "ejercicios": ej_list,
            "calorias": kcal,
            "calorias_quemadas": kcal,
            "duracion": int(round(dur)),
            "met": met,
        }
        set_consulta_cached(consulta_id, payload)
        sec["consulta_id"] = consulta_id


def es_payload_ejercicio(payload: Dict[str, Any]) -> bool:
    """True si el payload de tarjeta corresponde a ejercicio (POWER), no a comida."""
    return bool(
        "ejercicios" in payload
        or "gasto_calorico" in payload
        or "duracion" in payload
    )


def registrar_preferencias_ejercicios(extraccion: Dict[str, Any], perfil: Any, db: Session) -> None:
    """Auto-aprendizaje: frecuencia, puntuación, última vez; kcal de la fila = última sesión."""

    ejercicios_detectados = extraccion.get("ejercicios_detectados", [])
    if not ejercicios_detectados:
        ejercicios_detectados = extraccion.get("alimentos_detectados", [])
    if not ejercicios_detectados:
        return

    cals = float(extraccion.get("calorias") or 0)
    n = max(1, len(ejercicios_detectados))
    sc = cals / n
    ahora = datetime.now()

    for ejercicio in ejercicios_detectados:
        e_low = (ejercicio or "actividad").lower().strip()[:200]
        pref_existente = db.query(PreferenciaEjercicio).filter(
            PreferenciaEjercicio.client_id == perfil.id,
            sql_func.lower(PreferenciaEjercicio.ejercicio) == e_low,
        ).first()

        if pref_existente:
            pref_existente.frecuencia += 1
            pref_existente.ultima_vez = ahora
            pref_existente.puntuacion = min(5.0, pref_existente.puntuacion + 0.1)
            try:
                if hasattr(pref_existente, "calorias_quemadas"):
                    pref_existente.calorias_quemadas = sc
            except Exception:
                pass
        else:
            db.add(
                PreferenciaEjercicio(
                    client_id=perfil.id,
                    ejercicio=e_low,
                    frecuencia=1,
                    puntuacion=1.0,
                    ultima_vez=ahora,
                    calorias_quemadas=sc,
                )
            )


def registrar_ejercicio_desde_payload_tarjeta(
    payload: Dict[str, Any],
    perfil: Any,
    progreso: Any,
    db: Session,
) -> Dict[str, Any]:
    """
    Suma calorías quemadas al progreso y hace upsert de PreferenciaEjercicio (tarjeta o confirmación).
    No hace commit.
    """
    calorias = float(payload.get("calorias", 0) or payload.get("calorias_quemadas", 0) or 0)
    nombre = (payload.get("nombre") or "Actividad").strip()

    progreso.calorias_quemadas = (progreso.calorias_quemadas or 0) + calorias

    pref = (
        db.query(PreferenciaEjercicio)
        .filter(
            PreferenciaEjercicio.client_id == perfil.id,
            sql_func.lower(PreferenciaEjercicio.ejercicio) == nombre.lower(),
        )
        .first()
    )
    if pref:
        pref.frecuencia += 1
        pref.ultima_vez = datetime.now()
        pref.calorias_quemadas = calorias
    else:
        db.add(
            PreferenciaEjercicio(
                client_id=perfil.id,
                ejercicio=nombre.lower(),
                frecuencia=1,
                puntuacion=1.0,
                ultima_vez=datetime.now(),
                calorias_quemadas=calorias,
            )
        )

    return {
        "tipo_detectado": "ejercicio",
        "nombre": nombre,
        "calorias": calorias,
        "duracion_estimada": payload.get("duracion", 0),
    }


def aplicar_extraccion_nlp_ejercicio_a_progreso(extraccion: Dict[str, Any], progreso: Any) -> None:
    """Suma calorías quemadas cuando la extracción NLP no es comida (ejercicio / actividad)."""
    if extraccion.get("es_comida"):
        return
    calorias = extraccion.get("calorias", 0) or 0
    progreso.calorias_quemadas = (progreso.calorias_quemadas or 0) + calorias
