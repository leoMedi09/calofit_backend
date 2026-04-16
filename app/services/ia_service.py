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

try:
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
except ImportError:
    fuzz = None
    ctrl = None

from app.core.config import settings
from app.services.nutricion_service import nutricion_service

# Constantes de Salud
CONDICIONES_CRITICAS = [
    "diabetes", "hipertensión", "hipertension", "renal", "cardíaca",
    "cardiaca", "embarazo", "lactancia", "celiaco", "celíaco"
]

# ─────────────────────────────────────────────────────────────────────
# TABLA MET — Ejercicios de Gimnasio y Rutinas
# Fuente: Compendium of Physical Activities (Ainsworth et al., 2011)
# Fórmula: kcal = MET × peso_kg × tiempo_horas
# ─────────────────────────────────────────────────────────────────────
METS_GYM = {
    # Pesas / Fuerza
    "pesas": 5.0, "pesa": 5.0, "peso libre": 5.0, "mancuernas": 4.5,
    "barra": 5.0, "sentadilla": 5.5, "squat": 5.5, "press banca": 5.0,
    "press de banca": 5.0, "bench press": 5.0, "deadlift": 6.0,
    "peso muerto": 6.0, "dominadas": 5.0, "pull up": 5.0, "jalón": 4.5,
    "remo": 5.0, "curl biceps": 4.0, "curl de biceps": 4.0,
    "triceps": 4.0, "hombros": 4.5, "press militar": 4.5,
    "leg press": 5.0, "prensa": 5.0, "extensión": 4.0, "femoral": 4.5,
    "hip thrust": 4.5, "glúteos": 4.0, "glúte": 4.0,
    # Cardio de Gym
    "caminadora": 4.5, "trotadora": 7.0, "trotar caminadora": 7.0,
    "elíptica": 5.0, "bicicleta estática": 6.0, "spinning": 8.5,
    "remo máquina": 7.0, "rowing": 7.0, "escaladora": 8.0, "stepper": 6.0,
    # Cardio General (gym o fuera)
    "cardio": 6.0, "hiit": 10.0, "circuito": 7.0, "funcional": 6.5,
    "trotar": 8.0, "correr": 10.0, "saltar soga": 9.0, "cuerda": 9.0,
    # Core / Abs
    "abdominales": 3.8, "abs": 3.8, "plancha": 4.0, "crunch": 3.8,
    "core": 4.0, "flexiones": 4.0, "push up": 4.0, "burpees": 10.0,
    # Rutinas completas (aproximado)
    "rutina de pecho": 5.0, "rutina de espalda": 5.0, "rutina de pierna": 5.5,
    "rutina de hombros": 4.5, "rutina de brazo": 4.0, "rutina de brazos": 4.0,
    "día de pierna": 5.5, "día de pecho": 5.0, "día de espalda": 5.0,
    "entrené": 5.0, "entrene": 5.0, "entrenamiento": 5.0, "gym": 5.0,
    # Yoga / Pilates / Otros
    "yoga": 2.5, "pilates": 3.5, "stretching": 2.5, "estiramientos": 2.5,
    "natación": 7.0, "nadar": 7.0, "boxeo": 9.0, "zumba": 6.5,
}

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
        # Groq / Llama-3 (Motor de Lenguaje Natural)
        if AsyncGroq and getattr(settings, "GROQ_API_KEY", None):
            self.groq_client = AsyncGroq(api_key=settings.GROQ_API_KEY)
        else:
            self.groq_client = None
            print("⚠️ Groq no inicializado — modo offline activo.")

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
        """Retorna calorias y desglose de macros inicial."""
        calorias = self.calcular_requerimiento(genero, edad, peso, talla, nivel_actividad, objetivo)
        obj_lower = objetivo.lower()
        
        # Distribución estándar clínica 30/40/30 para pérdida, 25/50/25 para ganancia/mantenimiento
        prot_ratio = 0.30 if "perder" in obj_lower else 0.25
        carb_ratio = 0.40 if "perder" in obj_lower else 0.50
        gras_ratio = 1.0 - prot_ratio - carb_ratio
        
        return (
            calorias,
            round((calorias * prot_ratio) / 4, 1),
            round((calorias * carb_ratio) / 4, 1),
            round((calorias * gras_ratio) / 9, 1),
        )

    def calcular_macros_optimizados(
        self, calorias: float, objetivo: str, peso: float = 70.0
    ) -> Dict:
        """Distribución avanzada de macros basada en el peso del usuario."""
        obj = objetivo.lower()
        if "perder" in obj:
            prot_g, gras_ratio = round(peso * 2.1, 1), 0.25 # Alta proteína para preservar músculo
        elif "ganar" in obj:
            prot_g, gras_ratio = round(peso * 1.8, 1), 0.25
        else:
            prot_g, gras_ratio = round(peso * 1.6, 1), 0.25

        cal_gras = calorias * gras_ratio
        cal_carb = calorias - (prot_g * 4) - cal_gras
        return {
            "calorias_totales": round(calorias, 1),
            "proteinas_g":      prot_g,
            "carbohidratos_g":  round(cal_carb / 4, 1),
            "grasas_g":         round(cal_gras / 9, 1),
        }

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

    async def _llamar_groq(self, prompt: str, max_tokens: int = 800, temp: float = 0.7) -> str:
        if not self.groq_client: return "[Modo Offline]"
        try:
            r = await self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temp,
            )
            return r.choices[0].message.content.strip()
        except Exception as e: return f"[Error: {e}]"

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
            # Detectar duración: "30 minutos", "1 hora", "45 min", "1h", etc.
            duracion_min = 45  # default: sesión estándar de gym
            match_dur = re.search(
                r'(\d+)\s*(min(?:utos?)?|h(?:oras?|rs?)?)',
                texto_lower
            )
            if match_dur:
                valor = int(match_dur.group(1))
                unidad = match_dur.group(2)
                duracion_min = valor * 60 if unidad.startswith('h') else valor

            calorias = round(met_detectado * peso_usuario_kg * (duracion_min / 60), 1)
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
                "origen": "Tabla MET Científica 🏋️"
            }

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
        info_db = nutricion_service.obtener_info_alimento(texto)
        if info_db and float(info_db.get("calorias", 0) or 0) > 0:
            factor = porcion_g / 100.0
            print(f"✅ [DB-First] '{texto}' → {info_db['nombre']} ({porcion_g}g) = {round(info_db['calorias'] * factor, 1)} kcal [{info_db.get('origen','BD')}]")
            return {
                "es_comida": True, "es_ejercicio": False,
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
                "origen": info_db.get("origen", "BD Oficial 🇵🇪"),
            }

        # ─────────────────────────────────────────────────────────────
        # PASO 4: LLM como último recurso (platos compuestos no en BD)
        # El resultado se devuelve; el AsistenteService lo guardará
        # en PreferenciaAlimento para que la próxima vez sea consistente.
        # ─────────────────────────────────────────────────────────────
        print(f"⚠️ [DB-First] '{texto}' no encontrado en BD → usando LLM (se cacheará)")
        meal_context = "de desayuno" if 5 <= hora_peru < 10 else "estándar"
        prompt = (
            f'Soy un nutricionista peruano. Dame los macros de "{texto}" '
            f'para {porcion_g}g (1 porción {meal_context}). '
            f'Responde SOLO JSON sin texto extra: '
            f'{{"alimento": "nombre", "calorias": 0, "proteinas_g": 0, "carbohidratos_g": 0, '
            f'"grasas_g": 0, "es_comida": true}}'
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
                parsed["porcion_g"] = porcion_g
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
            print(f"❌ Error plan inicial: {e}")
            return None

# Instancia exportada
ia_service = IAService()
ia_engine = ia_service
