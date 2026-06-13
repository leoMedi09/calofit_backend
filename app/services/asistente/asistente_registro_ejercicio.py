"""
Registro de ejercicios por NLP — Integración MET + workout_logs.

Capacidades:
  - Registro directo: "Hice 3 series de 10 reps de Press de Banca con 50kg"
  - Por referencia de rutina: "Hice la rutina Piernas de Acero"
  - Preguntas de seguimiento cuando faltan datos (peso/tiempo)
  - ML sync: workout_logs ← calorias_quemadas, session_duration_min, intensity
"""
from __future__ import annotations

import json
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

# ── Muletillas de voz a eliminar antes de procesar ────────────────────────────
_RE_MULETILLAS = re.compile(
    r"(?i)\b(mm+h?|eeh?|aah?|uh+|uhm+|hmm+|o\s+sea(\s+que)?|como\s+que"
    r"|bueno\s+pues|pues\s+(?=\w)|la\s+verdad\s+(es\s+)?(que\s+)?"
    r"|y\s+este\s+|este\s+que\s+|este\s+(?=mm|este|o\s+sea|como\s+que))\b[,.]?\s*"
)
# "este"/"pues"/"bueno"/"oye" solos al inicio (muy común al hablar)
_RE_INICIO_MULETILLA = re.compile(r"(?i)^(este|pues|bueno|oye\s+pues)\s*[,.]?\s*")

# ── Números en palabras → dígitos ─────────────────────────────────────────────
# Soporta: unidades, decenas, "X y Y" (setenta y cinco → 75),
# centenas (doscientos, quinientos…) y fracciones ("y medio" → .5).
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

# "un/uno/una" como artículo NO debe convertirse a "1" cuando aparece solo.
# Solo se usa en compuestos ("treinta y uno", "veintiún").
# "con un peso de 50" → NO normalizar (sino "con 1 peso de 50" confunde el LLM).
_UNIDADES_STANDALONE = {k: v for k, v in _UNIDADES_MAP.items()
                         if k not in ("un", "uno", "una")}

# Regex para capturar números compuestos en español hablado
_RE_NUMERO_COMPUESTO = re.compile(
    r"(?i)\b("
    + "|".join(sorted(_CENTENAS_MAP, key=len, reverse=True))
    + r")(?:\s+(" + "|".join(sorted(_DECENAS_MAP, key=len, reverse=True))
    + r"))?(?:\s+y\s+(" + "|".join(sorted(_UNIDADES_MAP, key=len, reverse=True))
    + r"))?(?:\s+(y\s+medi[ao]|y\s+cuarto|y\s+tres\s+cuartos))?\b"
    + r"|(?i)\b(" + "|".join(sorted(_DECENAS_MAP, key=len, reverse=True))
    + r")(?:\s+y\s+(" + "|".join(sorted(_UNIDADES_MAP, key=len, reverse=True))
    + r"))?(?:\s+(y\s+medi[ao]|y\s+cuarto|y\s+tres\s+cuartos))?\b"
    + r"|(?i)\b(" + "|".join(sorted(_UNIDADES_STANDALONE, key=len, reverse=True))
    + r")(?:\s+(y\s+medi[ao]|y\s+cuarto|y\s+tres\s+cuartos))?\b"
)


def _resolver_numero_compuesto(m: re.Match) -> str:
    """Convierte grupos capturados de número compuesto a dígito."""
    g = [x.lower().strip() if x else None for x in m.groups()]
    # Grupo centena+decena+unidad+fraccion (grupos 0-3)
    if g[0] and g[0] in {k.lower() for k in _CENTENAS_MAP}:
        val = _CENTENAS_MAP.get(g[0], 0)
        if g[1]:
            val += _DECENAS_MAP.get(g[1], 0)
        if g[2]:
            val += _UNIDADES_MAP.get(g[2], 0)
        frac = _FRACCION_MAP.get(g[3] or "", "")
        return str(val) + frac
    # Grupo decena+unidad+fraccion (grupos 4-6)
    if g[4] and g[4] in {k.lower() for k in _DECENAS_MAP}:
        val = _DECENAS_MAP.get(g[4], 0)
        if g[5]:
            val += _UNIDADES_MAP.get(g[5], 0)
        frac = _FRACCION_MAP.get(g[6] or "", "")
        return str(val) + frac
    # Grupo unidad+fraccion (grupos 7-8)
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


def _normalizar_numeros(texto: str) -> str:
    """Normaliza texto de voz y convierte números hablados a dígitos."""
    return _normalizar_voz(texto)


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

# "por 20 min", "durante 45 minutos", "por 1 hora", "por 30 segundos"
_RE_DURACION = re.compile(
    r"(?i)\b(?:por|durante)\s+(\d+(?:[.,]\d+)?)\s*"
    r"(min(?:utos?)?|h(?:oras?)?|seg(?:undos?)?)\b"
)

# Intensidad por MET
def _met_a_intensity(met: float) -> str:
    if met >= 8.0:
        return "Alta"
    if met >= 5.0:
        return "Media"
    return "Baja"


class RegistroEjercicioHandler:
    """Orquesta el registro de ejercicios por NLP y el log de entrenamiento."""

    # Caché de ejercicio pendiente por usuario (para follow-up de datos_incompletos).
    # Cuando el asistente pregunta "¿cuántas series?" y el usuario responde
    # "3 series de 10 con 50kg", este dict provee el nombre del ejercicio.
    _ejercicio_pendiente: dict[int, str] = {}

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
        Flujo:
          1. Rutina nombrada  (regex — inequívoco)
          2. Extractor LLM    (primario — lenguaje libre, cualquier orden)
          3. Regex series/reps (fallback rígido)
          4. NLP estándar MET  (fallback genérico)
        Fórmula calorías: MET × peso_corporal × 3.5 / 200 × minutos
        """
        msg_lower = _normalizar_numeros((mensaje or "").lower().strip())
        peso_kg   = float(getattr(perfil, "weight", None) or 70.0)

        # ── 1. Referencia a rutina nombrada ───────────────────────────────────
        if _RE_MENCIONA_RUTINA.search(msg_lower):
            return self._procesar_referencia_rutina(mensaje, msg_lower, perfil, db)

        # ── 2. Extractor LLM (primario — uno o varios ejercicios) ────────────
        lista_llm = await self._llm_extraer_ejercicio(mensaje, ia_engine)

        # Follow-up context: si el usuario responde a la pregunta de seguimiento
        # ("¿cuántas series?") el LLM puede no saber el nombre del ejercicio.
        # Usar el caché del ejercicio pendiente para completar el contexto.
        _perfil_id = getattr(perfil, "id", None)
        _ejercicio_previo = self._ejercicio_pendiente.get(_perfil_id)
        if lista_llm and _ejercicio_previo:
            for _item in lista_llm:
                _tiene_datos = _item.get("series", 0) > 0 or _item.get("reps", 0) > 0
                _nombre_generico = any(
                    kw in (_item.get("ejercicio") or "").lower()
                    for kw in ("repeticion", "serie", "peso", "reps", "sets",
                               "entrenamiento", "ejercicio general")
                )
                if _tiene_datos and _nombre_generico:
                    _item["ejercicio"] = _ejercicio_previo

        if lista_llm:
            return await self._procesar_datos_llm(lista_llm, perfil, db, msg_lower, ia_engine)

        # ── 3. Fallback: series/reps explícitos por regex ─────────────────────
        series_match = _RE_SERIES_REPS.search(msg_lower)
        if series_match:
            return self._procesar_series_directo(mensaje, msg_lower, series_match, perfil, db)

        # ── 4. Flujo NLP estándar (MET por vocabulario) ───────────────────────
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

        dur_min    = extraccion.get("duracion_min", 45.0)
        met        = extraccion.get("met", 5.0)
        cal        = extraccion.get("calorias", 0.0)
        nombres    = extraccion.get("ejercicios_detectados", [])
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

    # ── Extractor LLM (lenguaje libre, múltiples ejercicios) ────────────────

    async def _llm_extraer_ejercicio(
        self, mensaje: str, ia_engine
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Extrae UNO O VARIOS ejercicios de cualquier phrasing natural usando Groq.
        Devuelve lista de dicts o None si no hay ejercicio en el mensaje.
        Soporta:
          - orden variable series/reps  ("5 reps por 2 series")
          - 'kilos' / 'kgs' / 'kg'
          - duraciones en texto ("media hora", "45 minutos")
          - saludo + acción ("hola amigo, hoy hice press y curl")
          - múltiples ejercicios en un solo mensaje
        """
        # Pre-normalizar texto de voz antes de enviar al LLM
        mensaje_norm = _normalizar_voz(mensaje)
        prompt = (
            "Eres un asistente de gimnasio. Extrae TODOS los ejercicios del mensaje.\n"
            "Responde SOLO con un JSON array válido (sin texto adicional).\n\n"
            "Schema por ejercicio:\n"
            '{"ejercicio":"<nombre canónico es>","series":<int>,"reps":<int>,'
            '"peso_kg":<float>,"duracion_min":<int>,"tipo":"fuerza|cardio|funcional"}\n\n'
            "REGLAS:\n"
            "- Si no menciona un campo, usa 0.\n"
            "- '70 kilos'/'70 kgs'/'setenta kilos' → peso_kg:70.\n"
            "- 'media hora'/'treinta minutos' → duracion_min:30.\n"
            "- 'N reps por M series' → series:M, reps:N  (el orden puede invertirse).\n"
            "- 'N sets de M reps' → series:N, reps:M.\n"
            "- Si hay VARIOS ejercicios, devuelve UN objeto por cada uno.\n"
            "- Si el mensaje NO contiene ningún ejercicio físico → devuelve: []\n"
            "- Saludo o muletillas + ejercicio → ignora el saludo/muletilla, extrae el ejercicio.\n"
            "- El texto puede ser transcripción de voz: sin puntuación, números escritos, muletillas.\n\n"
            "Ejemplos VOZ:\n"
            "'hice press banca setenta kilos cinco reps por dos series y luego curl biceps veinte kilos tres por doce'\n"
            '  → [{"ejercicio":"Press Banca","series":2,"reps":5,"peso_kg":70,"duracion_min":0,"tipo":"fuerza"},'
            '{"ejercicio":"Curl Biceps","series":3,"reps":12,"peso_kg":20,"duracion_min":0,"tipo":"fuerza"}]\n'
            "'me fui a correr como media hora por el parque'\n"
            '  → [{"ejercicio":"Trote","series":0,"reps":0,"peso_kg":0,"duracion_min":30,"tipo":"cardio"}]\n'
            "'este mmm hice sentadillas cuatro series de ocho con noventa kilos y dominadas tres por diez sin peso'\n"
            '  → [{"ejercicio":"Sentadilla","series":4,"reps":8,"peso_kg":90,"duracion_min":0,"tipo":"fuerza"},'
            '{"ejercicio":"Dominadas","series":3,"reps":10,"peso_kg":0,"duracion_min":0,"tipo":"fuerza"}]\n\n'
            f"Mensaje: {mensaje_norm[:500]}\n"
            "JSON array:"
        )
        try:
            raw = await ia_engine._llamar_groq(prompt, max_tokens=300, temp=0.05)
            raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
            m = re.search(r"\[.*\]", raw, re.DOTALL)
            if not m:
                return None
            lista = json.loads(m.group(0))
            if not isinstance(lista, list) or not lista:
                return None
            resultado = []
            for item in lista:
                if not isinstance(item, dict) or not item.get("ejercicio"):
                    continue
                resultado.append({
                    "ejercicio":    str(item.get("ejercicio", "")).strip().title(),
                    "series":       max(0, int(item.get("series", 0) or 0)),
                    "reps":         max(0, int(item.get("reps", 0) or 0)),
                    "peso_kg":      max(0.0, float(item.get("peso_kg", 0) or 0)),
                    "duracion_min": max(0, int(item.get("duracion_min", 0) or 0)),
                    "tipo":         str(item.get("tipo", "fuerza")).lower(),
                })
            return resultado or None
        except Exception:
            return None

    async def _estimar_met_groq(
        self, nombre_ejercicio: str, tipo: str, ia_engine
    ) -> float:
        """
        Estima el MET de un ejercicio desconocido usando Groq.
        Fallback cuando el catálogo interno no tiene el ejercicio.
        MET estándar: fuerza ~3-8, cardio ~6-14, funcional ~5-8.
        """
        prompt = (
            f"Indica el MET (Metabolic Equivalent of Task) para el ejercicio: '{nombre_ejercicio}'.\n"
            f"Tipo: {tipo}. Responde SOLO con el número decimal (ej: 5.5). "
            "Sin texto adicional. Usa valores de compendio de actividades físicas."
        )
        try:
            raw = await ia_engine._llamar_groq(prompt, max_tokens=10, temp=0.0)
            raw = raw.strip().replace(",", ".")
            val = float(re.search(r"\d+(?:\.\d+)?", raw).group(0))
            # Clamp a rango fisiológico razonable
            return max(2.0, min(val, 18.0))
        except Exception:
            return {"fuerza": 5.0, "cardio": 8.0, "funcional": 6.5}.get(tipo, 5.0)

    async def _procesar_datos_llm(
        self,
        lista_ejercicios: List[Dict[str, Any]],
        perfil,
        db: Session,
        msg_lower: str,
        ia_engine,
    ) -> Dict[str, Any]:
        """
        Registra UNO O VARIOS ejercicios extraídos por _llm_extraer_ejercicio.
        Acumula calorías totales y genera un mensaje de confirmación combinado.
        """
        peso_corporal = float(getattr(perfil, "weight", None) or 70.0)

        # Duración explícita en texto (compartida si el mensaje la dice una vez)
        dur_match = _RE_DURACION.search(msg_lower)
        dur_texto: Optional[float] = None
        if dur_match:
            _val = float(dur_match.group(1).replace(",", "."))
            _u   = dur_match.group(2).lower()
            dur_texto = (_val * 60 if _u.startswith("h")
                         else _val / 60 if _u.startswith("seg") else _val)

        cal_total   = 0.0
        nombres_reg = []
        detalles    = []

        for datos in lista_ejercicios:
            ejercicio = datos["ejercicio"]
            series_raw = datos["series"]
            reps_raw   = datos["reps"]
            peso_kg_e  = datos["peso_kg"] or None
            tipo       = datos["tipo"]
            dur_llm    = datos["duracion_min"]

            # ── Follow-up: ejercicio de fuerza sin series/reps ni duración ────────
            # "hice press banca" sin datos → preguntar en lugar de registrar 1×1.
            # Solo aplica a fuerza (cardio puede no tener reps).
            _es_fuerza_tipo = tipo == "fuerza" or any(t in msg_lower for t in [
                "press", "curl", "remo", "sentad", "peso muerto", "jalon",
                "dominad", "fondos", "hip thrust", "extension",
            ])
            if (
                series_raw == 0 and reps_raw == 0 and dur_llm == 0
                and _es_fuerza_tipo
                and dur_texto is None
                and len(lista_ejercicios) == 1
            ):
                # Guardar ejercicio en caché para el próximo mensaje (follow-up)
                _pid = getattr(perfil, "id", None)
                if _pid:
                    RegistroEjercicioHandler._ejercicio_pendiente[_pid] = ejercicio
                return self._pregunta_faltante("series_reps", ejercicio, 0, 0)

            # Cuando el usuario especifica duración pero no series/reps,
            # registrar con 0×0 (honesto) en lugar de forzar 1×1 (engañoso).
            # El kcal se calcula por MET×duración, no por series×reps.
            series    = series_raw  # puede ser 0 si no se especificó
            reps      = reps_raw    # puede ser 0 si no se especificó

            # Duración por ejercicio: texto explícito → LLM → estimado por series/reps
            if dur_texto is not None and len(lista_ejercicios) == 1:
                dur_min = dur_texto
            elif dur_llm > 0:
                dur_min = float(dur_llm)
            elif series > 0 and reps > 0:
                _seg_rep  = 4 if _es_fuerza_tipo else 3
                _seg_desc = 90 if _es_fuerza_tipo else 60
                dur_min = max(series * (reps * _seg_rep + _seg_desc) / 60, 5.0)
            else:
                # Sin duración ni series/reps → usar mínimo de 10 min
                dur_min = 10.0

            # MET: catálogo gym → estimación Groq para ejercicios no reconocidos
            _clave, met = resolver_met_mets_gym(ejercicio.lower() + " " + msg_lower)
            if not met:
                met = await self._estimar_met_groq(ejercicio, tipo, ia_engine)

            cal = round(ejercicios_service.calcular_calorias(met, peso_corporal, dur_min), 1)

            self._registrar_workout_log_completo(
                client_id=perfil.id,
                ejercicio=ejercicio,
                series=series, reps=reps, peso_kg=peso_kg_e,
                calorias_quemadas=cal,
                session_duration_min=dur_min,
                met=met, db=db,
            )
            self._sumar_calorias_progreso(perfil.id, cal, db)

            cal_total += cal
            nombres_reg.append(ejercicio)
            det = f"{ejercicio}"
            if series and reps:
                det += f" {series}×{reps}"
            elif dur_min and dur_min > 0:
                det += f" {dur_min:.0f} min"
            if peso_kg_e:
                det += f" @{peso_kg_e}kg"
            det += f" ({cal:.0f} kcal)"
            detalles.append(det)

        _hoy  = get_peru_date()
        _prog = db.query(ProgresoCalorias).filter(
            ProgresoCalorias.client_id == perfil.id,
            ProgresoCalorias.fecha == _hoy,
        ).first()

        # Registro exitoso → limpiar caché de ejercicio pendiente
        RegistroEjercicioHandler._ejercicio_pendiente.pop(perfil.id, None)

        msg_conf = " | ".join(detalles)
        return {
            "success": True,
            "tipo_detectado": "ejercicio_llm",
            "ejercicios": nombres_reg,
            "datos": {
                "calorias":     round(cal_total, 1),
                "ejercicios":   detalles,
            },
            "balance_actualizado": {
                "consumido": _prog.calorias_consumidas if _prog else 0,
                "quemado":   _prog.calorias_quemadas   if _prog else cal_total,
            },
            "mensaje": f"✅ Registré: {msg_conf} → Total: {cal_total:.0f} kcal quemadas.",
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

        # Calcular duración: usar tiempo real del texto si está disponible
        _dur_match = _RE_DURACION.search(msg_lower)
        if _dur_match:
            _val = float(_dur_match.group(1).replace(",", "."))
            _u   = _dur_match.group(2).lower()
            dur_min      = _val * 60 if _u.startswith("h") else (_val / 60 if _u.startswith("seg") else _val)
            _dur_es_real = True
        else:
            # Fallback estimado con parámetros realistas por tipo de ejercicio
            _es_fuerza_est = any(t in msg_lower for t in [
                "press", "curl", "remo", "sentad", "peso muerto", "jalon",
                "jalón", "extensi", "dominad", "fondos", "hip thrust",
            ])
            _seg_rep  = 4 if _es_fuerza_est else 3   # 4 seg/rep fuerza, 3 funcional/cardio
            _seg_desc = 90 if _es_fuerza_est else 60  # 90 seg descanso fuerza, 60 resto
            dur_min      = series * (reps * _seg_rep + _seg_desc) / 60
            _dur_es_real = False

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

        # Re-leer progreso actualizado para balance_actualizado
        _hoy = get_peru_date()
        _prog = db.query(ProgresoCalorias).filter(
            ProgresoCalorias.client_id == perfil.id,
            ProgresoCalorias.fecha == _hoy,
        ).first()

        detalle_peso = f" con {peso_kg} kg" if peso_kg else ""
        _origen_dur  = "del tiempo que indicaste" if _dur_es_real else "estimado por series/reps"
        return {
            "success": True,
            "tipo_detectado": "ejercicio_series",
            "ejercicios": [ejercicio_nombre or clave],
            "datos": {
                "series": series, "reps": reps, "peso_kg": peso_kg,
                "calorias": cal, "duracion_min": round(dur_min, 1), "met": met,
                "duracion_origen": _origen_dur,
            },
            "balance_actualizado": {
                "consumido": _prog.calorias_consumidas if _prog else 0,
                "quemado":   _prog.calorias_quemadas   if _prog else cal,
            },
            "mensaje": (
                f"✅ Registré: {ejercicio_nombre or clave} — "
                f"{series}×{reps}{detalle_peso} | {dur_min:.0f} min ({_origen_dur})"
                f" → {cal:.0f} kcal quemadas."
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

        _hoy = get_peru_date()
        _prog = db.query(ProgresoCalorias).filter(
            ProgresoCalorias.client_id == perfil.id,
            ProgresoCalorias.fecha == _hoy,
        ).first()

        return {
            "success": True,
            "tipo_detectado": "rutina_referencia",
            "ejercicios": [f"Rutina: {rutina_nombre}"],
            "datos": {
                "calorias": cal,
                "duracion_min": dur_min,
                "met": met_rutina,
            },
            "balance_actualizado": {
                "consumido": _prog.calorias_consumidas if _prog else 0,
                "quemado":   _prog.calorias_quemadas   if _prog else cal,
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
            # Nuevo: cuando el usuario no especificó series ni reps
            "series_reps": f"Anotado que hiciste {ejercicio}. "
                           f"¿Cuántas series y repeticiones hiciste, y con qué peso?",
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

    # ── Helpers privados ─────────────────────────────────────────────────────

    def _extraer_nombre_ejercicio(self, msg_lower: str, series_match: re.Match) -> str:
        """Extrae el nombre del ejercicio después de 'reps de X', sin incluir duración."""
        despues = msg_lower[series_match.end():]
        # Quitar "por X min/horas/seg" para evitar que el tiempo contamine el nombre
        despues_limpio = _RE_DURACION.sub("", despues).strip()
        m = re.search(r"(?i)(?:de\s+)(.+?)(?:\s+con\s+|\s+a\s+\d|\s*$)", despues_limpio)
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

        # Fallback: si el usuario usó un verbo de ejercicio + tiempo (aunque el ejercicio
        # específico no esté en el vocabulario GYM), registrar con MET genérico.
        if frase_vocabulario_gimnasio(mensaje) or frase_registro_actividad_fisica(mensaje):
            return extraccion_ejercicio_fallback_fuerza(mensaje, msg_lower, peso_kg)

        return None


registro_ejercicio_handler = RegistroEjercicioHandler()
