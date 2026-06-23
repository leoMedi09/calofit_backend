"""
╔══════════════════════════════════════════════════════════════════════╗
║  CaloFit — IA Service                                               ║
║  Arquitectura de Inteligencia Nutricional                            ║
║                                                                      ║
║  Pilar 1 → Mifflin-St Jeor (Base Clínica Nutricional)               ║
║  Pilar 2 → Random Forest (Perfilamiento de Adherencia)              ║
║  Pilar 3 → K-Nearest Neighbors (Recomendador de Alimentos)          ║
║  Pilar 4 → LLM Llama-3 vía Groq (Procesamiento NLP y Diálogo)       ║
╚══════════════════════════════════════════════════════════════════════╝

API Pública:
  calcular_requerimiento(genero, edad, peso, talla, nivel_act, objetivo) → float
  calcular_macros_completos(...)            → (cal, prot, carb, gras)
  calcular_macros_optimizados(cal, obj, peso) → dict
  generar_alerta_fuzzy(adh_pct, prog_pct)  → dict
  extraer_macros_de_texto(texto)           → dict
  identificar_intencion_salud(texto)       → str
  interpretar_comando_nlp(comando)         → dict
  recomendar_alimentos_con_groq(perfil)    → str
"""

import os
import json
import re
import asyncio
import pandas as pd
import numpy as np
import joblib
import httpx
from typing import List, Optional, Dict, Tuple

try:
    from groq import AsyncGroq
except ImportError:
    AsyncGroq = None

genai = None  # Gemini eliminado — solo Groq

try:
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
except ImportError:
    fuzz = None
    ctrl = None

from app.core.config import settings
from app.core.mets_gym import METS_GYM
from app.services.nutricion_service import nutricion_service

# Mensaje estable si httpx/Groq corta por tiempo (evitar "[Error: Request timed out.]" en el chat).
FALLBACK_MSG_IA_TIMEOUT = (
    "No pudimos completar la respuesta a tiempo. "
    "Por favor envía tu mensaje otra vez en unos segundos."
)

# Mensaje cuando Groq alcanza el límite diario de tokens (429)
FALLBACK_MSG_RATE_LIMIT = (
    "El asistente está temporalmente ocupado (límite de consultas alcanzado). "
    "Espera unos minutos y vuelve a intentarlo. "
    "Si el problema persiste, intenta más tarde."
)

# Constantes de Salud
CONDICIONES_CRITICAS = [
    "diabetes", "hipertensión", "hipertension", "renal", "cardíaca",
    "cardiaca", "embarazo", "lactancia", "celiaco", "celíaco"
]

# Porciones estándar según hora del día (Perú) y tipo de plato
PORCIONES_ESTANDAR = {
    # Momento del día
    "desayuno": 300, "almuerzo": 400, "cena": 300, "snack": 150,
    "merienda": 150, "media mañana": 150, "lonche": 200,
    # Tipo de plato
    "plato": 350, "plato de": 350, "porción": 200, "porcion": 200,
    "tazón": 300, "tazon": 300, "sopa": 300, "caldo": 300,
    "ensalada": 200, "fruta": 150, "pan": 80,
    "vaso": 250, "taza": 240, "botella": 500,
    "presa": 150, "filete": 150, "bistec": 180,
    "huevo": 55, "huevos": 110,
    "porcion pequeña": 150, "porcion grande": 450,
}


class IAService:
    """Motor de IA de CaloFit — Simplificado para Tesis (Random Forest + KNN + Llama-3)."""

    def __init__(self):
        # Groq — motor principal (14,400 req/día gratis, 30 RPM)
        self.groq_client = None
        if AsyncGroq and getattr(settings, "GROQ_API_KEY", None):
            self.groq_client = AsyncGroq(
                api_key=settings.GROQ_API_KEY,
                timeout=httpx.Timeout(180.0, connect=30.0),
                max_retries=2,
            )
            print("✅ Groq Llama-3.3-70B: Motor principal de IA inicializado.")
        else:
            print("⚠️  Groq no configurado.")

        self.gemini_model = None
        self.gemini_model_fast = None

        # FatSecret (Respaldo de macros)
        self._fs_client_id     = getattr(settings, "FATSECRET_CLIENT_ID", None)
        self._fs_client_secret = getattr(settings, "FATSECRET_CLIENT_SECRET", None)
        self._fs_token         = None

        # Motor de Lógica Difusa (Diagnóstico)
        self._alerta_sim = self._setup_fuzzy_logic()

    # ══════════════════════════════════════════════════════════════════
    # PILAR 1: CÁLCULO CLÍNICO (Mifflin-St Jeor)
    # ══════════════════════════════════════════════════════════════════

    def calcular_requerimiento(
        self, genero: int, edad: int, peso: float,
        talla: float, nivel_actividad: float, objetivo: str
    ) -> float:
        """
        Calcula el Gasto Energético Total (GET) basado en Mifflin-St Jeor.
        genero: 1=Hombre, 2=Mujer.
        """
        # TMB (Tasa Metabólica Basal)
        s = 5 if genero == 1 else -161
        tmb = (10 * peso) + (6.25 * talla) - (5 * edad) + s

        # GET = TMB * Nivel de Actividad
        mantenimiento = tmb * nivel_actividad

        # Ajuste por objetivo nutricional
        ajustes = {
            "perder": -500, "perder peso": -500, "perder_leve": -300,
            "mantener": 0,  "mantener peso": 0,
            "ganar_leve": 250,
            "ganar": 500,   "ganar masa": 500,
        }
        return round(mantenimiento + ajustes.get(objetivo.lower(), 0), 2)

    def calcular_macros_completos(
        self, genero: int, edad: int, peso: float,
        talla: float, nivel_actividad: float, objetivo: str
    ) -> Tuple[float, float, float, float]:
        """Retorna calorías y P/C/G (misma lógica que plan/dashboard: ``macros_desde_calorias_peso_objetivo``)."""
        from app.core.macros_diarios import macros_desde_calorias_peso_objetivo

        calorias = self.calcular_requerimiento(genero, edad, peso, talla, nivel_actividad, objetivo)
        m = macros_desde_calorias_peso_objetivo(calorias, objetivo, peso)
        return (
            calorias,
            m["proteinas_g"],
            m["carbohidratos_g"],
            m["grasas_g"],
        )

    def calcular_macros_optimizados(
        self, calorias: float, objetivo: str, peso: float = 70.0
    ) -> Dict:
        """Distribución de macros (g/kg + reparto grasa); ver app.core.macros_diarios."""
        from app.core.macros_diarios import macros_desde_calorias_peso_objetivo

        return macros_desde_calorias_peso_objetivo(calorias, objetivo, peso)

    # ══════════════════════════════════════════════════════════════════
    # LÓGICA DIFUSA (Diagnóstico de Adherencia)
    # ══════════════════════════════════════════════════════════════════

    def _setup_fuzzy_logic(self):
        if not (fuzz and ctrl): return None
        try:
            adherencia = ctrl.Antecedent(np.arange(0, 101, 1), "adherencia")
            progreso   = ctrl.Antecedent(np.arange(0, 101, 1), "progreso")
            alerta     = ctrl.Consequent(np.arange(0, 101, 1), "alerta")

            adherencia["baja"]  = fuzz.trimf(adherencia.universe, [0,   0,  50])
            adherencia["media"] = fuzz.trimf(adherencia.universe, [25, 50,  75])
            adherencia["alta"]  = fuzz.trimf(adherencia.universe, [50, 100, 100])

            progreso["lento"]  = fuzz.trimf(progreso.universe, [0,   0,  50])
            progreso["normal"] = fuzz.trimf(progreso.universe, [25, 50,  75])
            progreso["rapido"] = fuzz.trimf(progreso.universe, [50, 100, 100])

            alerta["suave"]    = fuzz.trimf(alerta.universe, [0,   0,  40])
            alerta["moderada"] = fuzz.trimf(alerta.universe, [30, 50,  70])
            alerta["estricta"] = fuzz.trimf(alerta.universe, [60, 100, 100])

            rules = [
                ctrl.Rule(adherencia["alta"]  & progreso["rapido"], alerta["suave"]),
                ctrl.Rule(adherencia["media"] & progreso["normal"], alerta["moderada"]),
                ctrl.Rule(adherencia["baja"]  | progreso["lento"],  alerta["estricta"]),
            ]
            return ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))
        except: return None

    def generar_alerta_fuzzy(self, adh_pct: float, prog_pct: float) -> Dict:
        if not self._alerta_sim: return {"nivel": "N/A", "score": 50, "mensaje": "Estándar."}
        try:
            self._alerta_sim.input["adherencia"] = max(0, min(100, adh_pct))
            self._alerta_sim.input["progreso"]   = max(0, min(100, prog_pct))
            self._alerta_sim.compute()
            score = self._alerta_sim.output["alerta"]
            if score < 40: nivel, msg = "Bajo",  "Excelente ritmo."
            elif score < 70: nivel, msg = "Medio", "Estable, sigue así."
            else: nivel, msg = "Alto",  "Necesitas refuerzo motivaional."
            return {"nivel": nivel, "score": round(float(score), 2), "mensaje": msg}
        except: return {"nivel": "N/A", "score": 50, "mensaje": "Estándar."}

    # ══════════════════════════════════════════════════════════════════
    # PROCESAMIENTO NLP (Llama-3 vía Groq)
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def es_fallo_respuesta_llm(text: Optional[str]) -> bool:
        t = (text or "").strip()
        if not t or t == "[Modo Offline]":
            return True
        if t.startswith("[Error:"):
            return True
        if t.startswith("No pudimos completar la respuesta a tiempo"):
            return True
        if t.startswith("El asistente está temporalmente ocupado"):
            return True
        return False

    @staticmethod
    def normalizar_etiqueta_modo_llm(raw: str) -> Optional[str]:
        """Extrae etiqueta válida del LLM — también detecta equivalentes semánticos."""
        from app.services.asistente.asistente_modos import MODOS_ASISTENTE

        t = (raw or "").strip().lower()
        if not t:
            return None
        t = re.sub(r"^```[a-z0-9]*\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```\s*$", "", t).strip()

        # Detección directa de etiquetas exactas
        for modo in sorted(MODOS_ASISTENTE, key=len, reverse=True):
            if re.search(rf"(?<![a-z0-9_]){re.escape(modo)}(?![a-z0-9_])", t, re.IGNORECASE):
                return modo

        # Mapeo semántico: palabras que el LLM usa en vez de la etiqueta exacta
        _SEMANTICO = {
            "registrar_nutricion": [
                "registro", "registrar", "comida", "comer", "ingesta", "alimento",
                "nutricion", "nutrición", "log", "anotar", "apuntar",
            ],
            "recomendar_nutricion": [
                "recomienda", "recomendacion", "recomendación", "sugerencia",
                "opciones", "qué comer", "que comer", "menú", "menu",
            ],
            "registrar_ejercicio": [
                "ejercicio registrado", "actividad física", "actividad fisica",
                "entrenamiento registrado", "cardio", "workout",
            ],
            "recomendar_ejercicio": [
                "rutina", "ejercicios recomendados", "plan de ejercicio",
            ],
        }
        for modo, palabras in _SEMANTICO.items():
            if any(p in t for p in palabras):
                return modo
        return None

    async def clasificar_modo_asistente(self, mensaje: str, historial: list = None) -> Optional[str]:
        """
        Clasificación semántica vía Groq (llamada breve, baja temperatura).
        Si devuelve ``None`` (sin API, error, respuesta inválida o desactivado por env),
        ``resolver_modo_funcion`` usa ``detectar_modo_funcion``.

        ``historial``: últimos turnos de la conversación (opcional). Sin esto, una
        pregunta de seguimiento como "¿cuál de esas tiene más proteína?" se ve
        aislada y el clasificador la confunde con una NUEVA petición de
        recomendación — con contexto, entiende que es una pregunta sobre algo
        ya mencionado y la manda a "otro" (conversación libre con memoria).

        Desactivar la llamada extra (p. ej. en benchmarks): ``CALOFIT_DISABLE_CLASIFICAR_MODO_LLM=1``.
        """
        if getattr(settings, "CALOFIT_DISABLE_CLASIFICAR_MODO_LLM", False):
            return None
        if not self.groq_client:
            return None
        m = (mensaje or "").strip()
        if len(m) < 2:
            return None
        _contexto_historial = ""
        if historial:
            _ultimos = historial[-4:]
            _lineas = "\n".join(
                f"{'Usuario' if h.get('role') == 'user' else 'Asistente'}: {str(h.get('content', ''))[:200]}"
                for h in _ultimos
            )
            _contexto_historial = (
                f"\n━━ CONVERSACIÓN RECIENTE (para contexto) ━━\n{_lineas}\n"
            )
        prompt = (
            "Eres el clasificador de intenciones de CaloFit, app de nutrición y ejercicio en Perú.\n"
            "Responde ÚNICAMENTE con una de estas 5 palabras (nada más, sin explicación):\n"
            "  registrar_nutricion | recomendar_nutricion | registrar_ejercicio | recomendar_ejercicio | otro\n\n"
            "━━ DEFINICIONES ━━\n"
            "registrar_nutricion — el usuario INFORMA que ya comió/bebió (tiempo pasado)\n"
            "  Señal clave: verbos pasados (comí, tomé, desayuné, almorcé, cené, bebí, me jalé, me tomé)\n"
            "  ✓ 'comí arroz con pollo'  ✓ 'desayuné avena con leche'  ✓ 'me jalé un cuy frito con papas'\n"
            "  ✓ 'almorcé lomo saltado'  ✓ 'tomé un batido después del gym'  ✓ 'cené menestra con arroz'\n\n"
            "recomendar_nutricion — el usuario PIDE ideas de qué comer (futuro/duda)\n"
            "  Señal clave: preguntas o expresión de hambre/duda sobre alimentos\n"
            "  ✓ '¿qué ceno?'  ✓ 'tengo hambre'  ✓ 'dame opciones para almorzar'  ✓ 'qué me recomiendas'\n\n"
            "registrar_ejercicio — el usuario INFORMA que ya entrenó (tiempo pasado)\n"
            "  Señal clave: verbos pasados de ejercicio (hice, entrené, corrí, caminé, nadé, fui al gym)\n"
            "  ✓ 'hice press banca 3x10 con 70 kilos'  ✓ 'entrené piernas hoy'\n"
            "  ✓ 'corrí 5km en el parque'  ✓ 'fui al gym'  ✓ 'acabo de terminar mi rutina de pecho'\n"
            "  ✓ 'hola amigo tiré sentadillas con 80kg tres por diez'  ✓ 'hice tres series de dominadas'\n\n"
            "recomendar_ejercicio — el usuario PIDE ejercicios o EXPRESA deseo de entrenar\n"
            "  Señal clave: petición de rutina, o 'quiero/necesito + entrenar'\n"
            "  ✓ 'dame rutina de pecho'  ✓ 'quiero hacer ejercicio para bajar de peso'\n"
            "  ✓ 'qué ejercicios hago hoy'  ✓ 'arma mi entrenamiento'  ✓ 'necesito empezar a entrenar'\n\n"
            "otro — preguntas de permiso, saludos, consultas informativas, progreso del día\n"
            "  Señal clave: 'puedo + infinitivo', 'se puede', 'es bueno/malo', '¿cómo voy?', saludo\n"
            "  ✓ 'puedo ir a nadar'          → otro  (permiso, NO registro)\n"
            "  ✓ 'puedo realizar un trote'   → otro  (infinitivo = intención futura, no acción pasada)\n"
            "  ✓ 'puedo trotar con lesión'   → otro\n"
            "  ✓ 'se puede nadar todos los días' → otro\n"
            "  ✓ 'es bueno correr en ayunas' → otro\n"
            "  ✓ 'puedo comer carbohidratos de noche' → otro\n"
            "  ✓ '¿cuánto llevo hoy?'        → otro\n"
            "  ✓ 'hola, cómo estás'          → otro\n"
            "  ✓ 'cuántas calorías tiene una palta' → otro\n\n"
            "━━ REGLAS CRÍTICAS ━━\n"
            "R1. INFINITIVO ≠ PASADO: 'puedo realizar/trotar/nadar' (infinitivo) = otro.\n"
            "    'realicé/corrí/nadé' (pretérito) = registrar_ejercicio.\n"
            "R2. MODAL = PERMISO: cualquier 'puedo/se puede/podría/es posible + verbo' = otro.\n"
            "R3. MEZCLA saludo+acción: 'hola amigo hoy hice press' → clasifica por la ACCIÓN (registrar_ejercicio).\n"
            "R4. VOZ: ignora muletillas (mmm, este, pues, o sea) y números en palabras (setenta kilos).\n"
            "R5. AMBIGUO: si no estás seguro, prefiere 'otro' antes que un registro incorrecto.\n"
            "R6. PREGUNTA DE SEGUIMIENTO: si el mensaje hace referencia a algo YA mencionado en "
            "la conversación reciente (palabras como 'esas', 'esa', 'ese', 'la anterior', 'cuál de "
            "los/las', o compara/pregunta sobre opciones sin nombrar un alimento/ejercicio nuevo), "
            "clasifica como 'otro' — es una pregunta sobre la respuesta anterior, no una nueva "
            "petición. Ejemplo: si el asistente ya dio 3 platos y el usuario pregunta '¿cuál tiene "
            "más proteína?' o '¿cuál es más barato?', eso es 'otro', NO recomendar_nutricion.\n"
            "R7. CORRECCIÓN/ADICIÓN A UN REGISTRO RECIENTE: si el usuario pide agregar, sumar, "
            "completar o corregir algo en una comida que YA registró en esta conversación (ej. "
            "'agrégalo al registro', 'súmale eso', 'olvidé decir que también comí X', 'le faltó "
            "el huevo, regístralo'), clasifica como 'registrar_nutricion' — es una ACCIÓN sobre "
            "el registro (persistir algo nuevo), no una pregunta de seguimiento como en R6.\n"
            "  ✓ 'agrégalo al registro' (tras mencionar un alimento olvidado) → registrar_nutricion\n"
            "  ✓ 'súmale el huevo que me faltó' → registrar_nutricion\n"
            "  ✓ 'también comí una manzana, anótala' → registrar_nutricion\n"
            "R8. PREGUNTA SÍ/NO SOBRE UN ALIMENTO YA NOMBRADO ≠ PEDIR IDEAS: si el usuario "
            "nombra un alimento ESPECÍFICO y pregunta si es recomendable/bueno/apropiado "
            "comerlo (a cierta hora, en su dieta, etc.), eso es 'otro' — quiere una respuesta "
            "sí/no sobre ESE alimento, no una lista de opciones nuevas. 'Qué me recomiendas' "
            "SOLO es recomendar_nutricion cuando NO nombra ningún alimento (pregunta abierta).\n"
            "  ✓ '¿me recomiendas comer chancho a esta hora?' → otro (alimento ya nombrado: chancho)\n"
            "  ✓ '¿es bueno comer palta de noche?' → otro\n"
            "  ✓ '¿qué me recomiendas para la cena?' → recomendar_nutricion (sin nombrar alimento)\n"
            f"{_contexto_historial}\n"
            f"Mensaje a clasificar: \"{m[:500]}\"\n"
            "Respuesta (una sola palabra exacta):"
        )
        # Clasificación con 70B — salida ~1 token, ~400 tokens totales por llamada
        # Consume ~400/6000 TPM del límite 70B → ~15 clasificaciones/min (suficiente para demo)
        raw = await self._llamar_groq(prompt, max_tokens=10, temp=0.0,
                                      model="llama-3.3-70b-versatile")
        if self.es_fallo_respuesta_llm(raw):
            # Fallback al 8B si el 70B falla (rate limit, timeout)
            raw = await self._llamar_groq(prompt, max_tokens=10, temp=0.0)
        if self.es_fallo_respuesta_llm(raw):
            return None
        return IAService.normalizar_etiqueta_modo_llm(raw)

    async def _llamar_gemini(self, prompt: str, fast: bool = False) -> str:
        """Llama a Gemini 2.0 Flash."""
        if not self.gemini_model:
            return "[Modo Offline — configura GEMINI_API_KEY]"
        try:
            loop = asyncio.get_event_loop()
            resp = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: self.gemini_model.generate_content(prompt)),
                timeout=30.0
            )
            return resp.text.strip()
        except asyncio.TimeoutError:
            return FALLBACK_MSG_IA_TIMEOUT
        except Exception as e:
            err = str(e).lower()
            if "429" in err or "quota" in err or "exhausted" in err:
                print(f"[Gemini] Cuota agotada — espera 1 minuto: {e}")
                return FALLBACK_MSG_RATE_LIMIT
            if "timeout" in err:
                return FALLBACK_MSG_IA_TIMEOUT
            print(f"[Gemini] Error inesperado: {e}")
            return FALLBACK_MSG_IA_TIMEOUT

    async def _llamar_groq(self, prompt: str, max_tokens: int = 800, temp: float = 0.7,
                           model: str | None = None) -> str:
        """
        Motor de IA con selección de modelo:
        - llama-3.1-8b-instant (default): 200,000 TPM — respuestas principales, NLP, macros
        - llama-3.3-70b-versatile (model param): clasificación de intención solamente
          (~10 tokens output → ~400 tokens por llamada → ~15 clasificaciones/min)
        """
        if not self.groq_client:
            return "[Modo Offline]"
        modelo = model or "llama-3.1-8b-instant"
        try:
            r = await self.groq_client.chat.completions.create(
                model=modelo,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temp,
            )
            return r.choices[0].message.content.strip()
        except Exception as e:
            err = str(e).lower()
            if "timed out" in err or "timeout" in err:
                return FALLBACK_MSG_IA_TIMEOUT
            if "429" in err or "rate_limit" in err or "rate limit" in err:
                return FALLBACK_MSG_RATE_LIMIT
            return f"[Error: {e}]"

    async def recomendar_alimentos_con_groq(
        self, perfil_usuario: Dict, comando_texto: str = None,
        adherencia_pct: float = 100, progreso_pct: float = 50,
    ) -> str:
        """Genera plan nutricional usando el contexto del usuario y datos de los modelos ML."""
        genero = 1 if perfil_usuario.get("gender", "M") == "M" else 2
        calorias = self.calcular_requerimiento(
            genero, perfil_usuario.get("age", 25), perfil_usuario.get("weight", 70),
            perfil_usuario.get("height", 170), perfil_usuario.get("activity_level", 1.2),
            perfil_usuario.get("goal", "mantener")
        )
        macros = self.calcular_macros_optimizados(calorias, perfil_usuario.get("goal", "mantener"), perfil_usuario.get("weight", 70))
        
        # Recuperar y formatear condiciones médicas si existen
        med_conditions = perfil_usuario.get("medical_conditions", [])
        cond_texto = ", ".join(med_conditions) if med_conditions else "Ninguna"
        
        alerta_clinica = ""
        if cond_texto != "Ninguna" and cond_texto != "":
            alerta_clinica = f"\n⚠️ CONDICIONES MÉDICAS CRÍTICAS: {cond_texto}. ESTÁ PROHIBIDO RECOMENDAR ALIMENTOS DAÑINOS PARA ESTAS PATOLOGÍAS."

        prompt = f"""Eres un Nutricionista Clínico y Deportivo muy profesional especializado en gastronomía peruana.
CLIENTE: {perfil_usuario.get('first_name', 'Cliente')} | OBJETIVO: {perfil_usuario.get('goal', 'mantener')}
METAS: {calorias} kcal | P: {macros['proteinas_g']}g | C: {macros['carbohidratos_g']}g | G: {macros['grasas_g']}g{alerta_clinica}

REGLAS INFLEXIBLES:
1. Usa principalmente alimentos disponibles en Perú (Quinua, Pollo, Camote, Atún, etc).
2. Si el cliente tiene Condiciones Médicas Críticas, MENCIONA BREVEMENTE POR QUÉ evitaste ciertos alimentos en base a su patología.
3. Sé empático pero sumamente riguroso clínicamente.
{f'COMANDO DEL CLIENTE: {comando_texto}' if comando_texto else ''}"""
        return await self._llamar_groq(prompt)

    async def sugerir_guia_estrategica(self, perfil_usuario: Dict, alertas_salud: Optional[List[Dict]] = None) -> Dict:
        """
        Genera una guía estratégica mensual para el Nutricionista.
        Respuesta estable para frontend:
          - ai_strategic_focus: str
          - recommended_foods: List[str]
          - forbidden_foods: List[str]
        """
        alertas_salud = alertas_salud or []
        condiciones = perfil_usuario.get("medical_conditions") or []
        objetivo = str(perfil_usuario.get("goal", "mantener")).lower()

        # Fallback determinístico si no hay LLM disponible.
        fallback_focus = "Control nutricional y adherencia sostenida"
        if "perder" in objetivo:
            fallback_focus = "Déficit calórico moderado con alta saciedad"
        elif "ganar" in objetivo:
            fallback_focus = "Superávit controlado y ganancia muscular limpia"

        recommended_default = ["Pollo", "Pescado", "Huevo", "Quinua", "Avena", "Verduras", "Frutas enteras"]
        forbidden_default = ["Bebidas azucaradas", "Frituras", "Ultraprocesados", "Alcohol excesivo"]

        if any("diabetes" in str(c).lower() for c in condiciones):
            forbidden_default.extend(["Jugos azucarados", "Postres con azúcar refinada"])
        if any("hipert" in str(c).lower() for c in condiciones):
            forbidden_default.append("Snacks altos en sodio")

        # Sanitiza duplicados conservando orden.
        forbidden_default = list(dict.fromkeys(forbidden_default))

        if not self.groq_client:
            return {
                "ai_strategic_focus": fallback_focus,
                "recommended_foods": recommended_default[:8],
                "forbidden_foods": forbidden_default[:8],
            }

        alertas_txt = "; ".join(
            [f"{a.get('tipo', 'N/A')}: {a.get('descripcion', '')} (sev={a.get('severidad', 'N/A')})" for a in alertas_salud]
        ) or "Sin alertas relevantes recientes"
        cond_txt = ", ".join([str(c) for c in condiciones]) if condiciones else "Ninguna"
        peso_hist = perfil_usuario.get("weight_history") or []
        peso_hist_txt = ", ".join([f"{h.get('valor')}kg" for h in peso_hist[-6:]]) if peso_hist else "Sin historial"

        prompt = f"""Eres un nutricionista clínico experto. Debes responder SOLO JSON válido.
Perfil:
- Nombre: {perfil_usuario.get('full_name', 'Paciente')}
- Sexo: {perfil_usuario.get('gender', 'N/A')}
- Edad: {perfil_usuario.get('age', 'N/A')}
- Peso actual: {perfil_usuario.get('current_weight', 'N/A')} kg
- Talla: {perfil_usuario.get('current_height', 'N/A')} cm
- IMC: {perfil_usuario.get('imc', 'N/A')}
- Actividad: {perfil_usuario.get('activity_level', 'N/A')}
- Objetivo: {perfil_usuario.get('goal', 'N/A')}
- Condiciones médicas: {cond_txt}
- Historial de peso reciente: {peso_hist_txt}
- Alertas de salud (15 días): {alertas_txt}

Devuelve exactamente este esquema:
{{
  "ai_strategic_focus": "frase breve y accionable",
  "recommended_foods": ["alimento1", "alimento2", "alimento3", "alimento4", "alimento5"],
  "forbidden_foods": ["alimento1", "alimento2", "alimento3", "alimento4", "alimento5"]
}}

Reglas:
1) Usa alimentos comunes en Perú.
2) Si hay condiciones médicas, ajusta recomendaciones y restricciones.
3) No uses markdown, no agregues texto fuera del JSON.
"""
        raw = await self._llamar_groq(prompt, max_tokens=400, temp=0.3)
        try:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            parsed = json.loads(m.group()) if m else json.loads(raw)
            focus = str(parsed.get("ai_strategic_focus", fallback_focus)).strip() or fallback_focus
            rec = parsed.get("recommended_foods", recommended_default)
            forb = parsed.get("forbidden_foods", forbidden_default)

            if not isinstance(rec, list): rec = recommended_default
            if not isinstance(forb, list): forb = forbidden_default
            rec = [str(x).strip() for x in rec if str(x).strip()]
            forb = [str(x).strip() for x in forb if str(x).strip()]

            # Salida con tamaño y contenido predecible para el frontend.
            if not rec: rec = recommended_default
            if not forb: forb = forbidden_default
            return {
                "ai_strategic_focus": focus,
                "recommended_foods": list(dict.fromkeys(rec))[:8],
                "forbidden_foods": list(dict.fromkeys(forb))[:8],
            }
        except Exception:
            return {
                "ai_strategic_focus": fallback_focus,
                "recommended_foods": recommended_default[:8],
                "forbidden_foods": forbidden_default[:8],
            }

    # ══════════════════════════════════════════════════════════════════
    # NLP — IDENTIFICACIÓN DE INTENCIONES
    # ══════════════════════════════════════════════════════════════════

    def identificar_intencion_salud(self, texto: str) -> str:
        t = texto.lower()
        if any(k in t for k in ["duele", "mal", "mareo", "dolor"]): return "ALERT"
        if any(k in t for k in ["hice", "entrené", "gym", "cardio"]): return "EXERCISE"
        if any(k in t for k in ["comí", "desayuné", "cené", "registra"]): return "LOG"
        if any(k in t for k in ["caloría", "cuánto tiene", "macro"]): return "INFO"
        if any(k in t for k in ["qué como", "recomienda", "plan", "dieta"]): return "RECIPE"
        return "GENERAL"

    def interpretar_comando_nlp(self, comando: str) -> Dict:
        """Parsea el lenguaje natural para identificar intenciones de registro."""
        intencion = self.identificar_intencion_salud(comando)
        numeros = re.findall(r'\d+\.?\d*', comando)
        return {
            "intencion": intencion,
            "texto_original": comando,
            "cantidad_detectada": float(numeros[0]) if numeros else None,
            "palabras_clave": [w for w in comando.lower().split() if len(w) > 4],
        }

    async def extraer_macros_de_texto(
        self, texto: str, peso_usuario_kg: float = 70.0,
        # Alias para compatibilidad con código anterior
        peso_usuario: float = None,
    ) -> Dict:
        """
        DB-First: CENAN → SQLite Mundial → LLM (último recurso).
        Garantiza valores consistentes en todos los contextos:
        - Solo registrar, solo consultar, recomendar+registrar, voz, texto.
        Si el LLM es necesario, guarda el resultado para reutilizarlo.
        """
        if peso_usuario is not None:
            peso_usuario_kg = peso_usuario

        texto_lower = texto.lower().strip()
        from app.services.nutricion_service import nutricion_service

        # ─────────────────────────────────────────────────────────────
        # PASO 1: Detectar ejercicio de gym (fuente: tabla MET)
        # ─────────────────────────────────────────────────────────────
        ejercicio_detectado = None
        met_detectado = None
        for ejercicio, met in sorted(METS_GYM.items(), key=lambda x: -len(x[0])):
            if ejercicio in texto_lower:
                ejercicio_detectado = ejercicio
                met_detectado = met
                break

        if ejercicio_detectado:
            # Misma duración y fórmula que en tarjetas POWER (ACSM: MET×3.5×kg/200×min).
            from app.services.asistente.asistente_ejercicio import parse_duracion_minutos
            from app.services.ejercicios_service import ejercicios_service

            duracion_min = parse_duracion_minutos(texto_lower, default=45.0)
            calorias = round(
                ejercicios_service.calcular_calorias(
                    met_detectado, peso_usuario_kg, duracion_min
                ),
                1,
            )
            return {
                "es_ejercicio": True, "es_comida": False,
                "calorias": calorias, "proteinas_g": 0,
                "carbohidratos_g": 0, "grasas_g": 0,
                "fibra_g": 0, "azucar_g": 0, "sodio_mg": 0,
                "ejercicios_detectados": [f"{ejercicio_detectado} {duracion_min} min"],
                "alimentos_detectados": [],
                "calidad_nutricional": "Alta",
                "duracion_min": duracion_min,
                "met": met_detectado,
                "origen": "Tabla MET Cientifica"
            }

        # PASO 1b: "hice X … min/series" con vocabulario de gimnasio → ejercicio (no CENAN/LLM comida)
        from app.services.asistente.asistente_ejercicio import (
            extraccion_ejercicio_fallback_fuerza,
            frase_registro_actividad_fisica,
            frase_vocabulario_gimnasio,
        )
        if frase_registro_actividad_fisica(texto) and frase_vocabulario_gimnasio(texto):
            return extraccion_ejercicio_fallback_fuerza(texto, texto_lower, peso_usuario_kg)

        # ─────────────────────────────────────────────────────────────
        # PASO 2: Detectar porción según hora peruana y texto
        # ─────────────────────────────────────────────────────────────
        try:
            from app.core.utils import get_peru_now
            hora_peru = get_peru_now().hour
        except Exception:
            hora_peru = 12

        # Porción base según hora del día
        if 5 <= hora_peru < 10:
            porcion_g = PORCIONES_ESTANDAR["desayuno"]   # 300g
        elif 10 <= hora_peru < 12:
            porcion_g = PORCIONES_ESTANDAR["snack"]      # 150g
        elif 12 <= hora_peru < 15:
            porcion_g = PORCIONES_ESTANDAR["almuerzo"]   # 400g
        elif 15 <= hora_peru < 18:
            porcion_g = PORCIONES_ESTANDAR["snack"]      # 150g
        elif 18 <= hora_peru < 21:
            porcion_g = PORCIONES_ESTANDAR["cena"]       # 300g
        else:
            porcion_g = PORCIONES_ESTANDAR["snack"]      # 150g (nocturno)

        # Sobreescribir si el usuario especifica cantidad explícita
        match_cantidad = re.search(
            r'(\d+(?:\.\d+)?)\s*(?:g\b|gr\b|gramos?|ml\b|kg\b)',
            texto_lower
        )
        if match_cantidad:
            porcion_g = float(match_cantidad.group(1))

        # Sobreescribir si el texto menciona tipo de plato/momento
        for keyword, gramos in PORCIONES_ESTANDAR.items():
            if keyword in texto_lower:
                porcion_g = gramos
                break

        # ─────────────────────────────────────────────────────────────
        # PASO 3: Buscar en CENAN/INS (fuente de verdad oficial Perú)
        # ─────────────────────────────────────────────────────────────
        _texto_busqueda = texto.split("(NOTA INTERNA", 1)[0].strip()
        info_db = nutricion_service.obtener_info_alimento(_texto_busqueda or texto)
        if info_db and float(info_db.get("calorias", 0) or 0) > 0:
            factor = porcion_g / 100.0
            print(f"[DB-First] '{texto}' -> {info_db['nombre']} ({porcion_g}g) = {round(info_db['calorias'] * factor, 1)} kcal [{info_db.get('origen','BD')}]")
            return {
                "es_comida": True,
                "es_ejercicio": False,
                "calorias": round(float(info_db["calorias"]) * factor, 1),
                "proteinas_g": round(float(info_db.get("proteinas", 0) or 0) * factor, 1),
                "carbohidratos_g": round(float(info_db.get("carbohidratos", 0) or 0) * factor, 1),
                "grasas_g": round(float(info_db.get("grasas", 0) or 0) * factor, 1),
                "fibra_g": round(float(info_db.get("fibra", 0) or 0) * factor, 1),
                "azucar_g": round(float(info_db.get("azucares", 0) or 0) * factor, 1),
                "sodio_mg": round(float(info_db.get("sodio", 0) or 0) * factor, 1),
                "alimentos_detectados": [info_db["nombre"]],
                "ejercicios_detectados": [],
                "calidad_nutricional": "Alta",
                "porcion_g": porcion_g,
                "origen": info_db.get("origen", "BD Oficial"),
            }

        # ─────────────────────────────────────────────────────────────
        # PASO 3b: FatSecret API (macros estándar) antes del LLM
        # ─────────────────────────────────────────────────────────────
        if not getattr(settings, "DISABLE_FATSECRET", False):
            try:
                from app.services.fatsecret_client import get_fatsecret_client, simplify_text_for_fatsecret_query

                fs = get_fatsecret_client()
                if fs:
                    q = simplify_text_for_fatsecret_query(texto)
                    if len(q) >= 2:
                        hit = await asyncio.to_thread(fs.lookup_macros, q, porcion_g)
                        if hit and float(hit.get("calorias", 0) or 0) > 0:
                            print(
                                f"[FatSecret] '{q}' -> {hit.get('nombre')} "
                                f"= {hit['calorias']} kcal (porción ref. {porcion_g}g)"
                            )
                            return {
                                "es_comida": True,
                                "es_ejercicio": False,
                                "calorias": float(hit["calorias"]),
                                "proteinas_g": float(hit.get("proteinas", 0) or 0),
                                "carbohidratos_g": float(hit.get("carbohidratos", 0) or 0),
                                "grasas_g": float(hit.get("grasas", 0) or 0),
                                "fibra_g": float(hit.get("fibra", 0) or 0),
                                "azucar_g": float(hit.get("azucares", 0) or 0),
                                "sodio_mg": float(hit.get("sodio", 0) or 0),
                                "alimentos_detectados": [hit.get("nombre", q)],
                                "ejercicios_detectados": [],
                                "calidad_nutricional": "Alta",
                                "porcion_g": porcion_g,
                                "origen": hit.get("origen", "FatSecret API"),
                            }
            except Exception as e:
                print(f"[FatSecret] {e}")

        # ─────────────────────────────────────────────────────────────
        # PASO 4: LLM como último recurso (platos compuestos no en BD)
        # El resultado se devuelve; el AsistenteService lo guardará
        # en PreferenciaAlimento para que la próxima vez sea consistente.
        # ─────────────────────────────────────────────────────────────
        print(f"[DB-First] '{texto}' no encontrado en BD -> usando LLM (se cacheara)")
        meal_context = "de desayuno" if 5 <= hora_peru < 10 else "tamaño de porción razonable"
        qtxt = (texto.split("(NOTA INTERNA", 1)[0] or texto).replace('"', "'").strip()
        prompt = (
            f"Eres nutricionista. Estima 1 porción típica ({meal_context}) del alimento: «{qtxt}». "
            "Responde **solo** JSON: "
            '{"alimento": "string", "calorias": 0, "proteinas_g": 0, "carbohidratos_g": 0, "grasas_g": 0, "es_comida": true}'
        )
        raw = await self._llamar_groq(prompt, max_tokens=200, temp=0.1)
        try:
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            parsed = json.loads(m.group()) if m else {}
            if parsed.get("calorias", 0):
                parsed.setdefault("es_comida", True)
                parsed.setdefault("es_ejercicio", False)
                parsed.setdefault("alimentos_detectados", [parsed.get("alimento", texto)])
                parsed.setdefault("ejercicios_detectados", [])
                parsed.setdefault("calidad_nutricional", "Media")
                parsed["porcion_g"] = float(porcion_g) if porcion_g else 250.0
                parsed["origen"] = "LLM (Llama-3)"
                return parsed
        except Exception:
            pass
        return {"es_comida": False, "es_ejercicio": False, "calorias": 0, "alimentos_detectados": [], "ejercicios_detectados": []}

    # ══════════════════════════════════════════════════════════════════
    # GENERADORES DE PLAN INICIAL
    # ══════════════════════════════════════════════════════════════════

    def generar_plan_inicial_automatico(self, datos_cliente: Dict) -> Optional[Dict]:
        """Genera el primer plan del cliente usando base clínica y personalización por objetivo."""
        try:
            genero = 1 if str(datos_cliente.get("genero", "M")).upper() == "M" else 2
            calorias = self.calcular_requerimiento(
                genero, int(datos_cliente.get("edad", 25)), float(datos_cliente.get("peso", 70)),
                float(datos_cliente.get("talla", 170)), float(datos_cliente.get("nivel_actividad", 1.2)),
                datos_cliente.get("objetivo", "mantener")
            )
            macros = self.calcular_macros_optimizados(calorias, datos_cliente.get("objetivo", "mantener"), float(datos_cliente.get("peso", 70)))
            
            dias = []
            for i in range(1, 8):
                dias.append({
                    "dia_numero": i,
                    "calorias_dia": calorias,
                    "proteinas_g": macros["proteinas_g"],
                    "carbohidratos_g": macros["carbohidratos_g"],
                    "grasas_g": macros["grasas_g"],
                    "sugerencia_entrenamiento_ia": "Entrenamiento moderado sugerido por IA.",
                    "nota_asistente_ia": f"Día {i} enfocado en {datos_cliente.get('objetivo')}."
                })
            return {"calorias_diarias": calorias, "macros": macros, "dias": dias}
        except Exception as e:
            print(f"[Plan inicial] Error: {e}")
            return None

# Instancia exportada
ia_service = IAService()
ia_engine = ia_service
