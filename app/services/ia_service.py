import re
import json
import os
import asyncio
import functools
import pandas as pd
try:
    from groq import AsyncGroq
except ImportError:
    AsyncGroq = None
    print("⚠️ Advertencia: Librería 'groq' no encontrada. Consultas a IA desactivadas.")
from app.core.config import settings
from app.services.nutricion_service import NutricionService
from datetime import datetime
import numpy as np

# Imports opcionales de ML
try:
    import joblib
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    joblib = None
    cosine_similarity = None

try:
    from tensorflow import keras
except ImportError:
    keras = None

try:
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
except ImportError:
    fuzz = None
    ctrl = None
from app.services.nutricion_service import NutricionService

# ==========================================================
# 1. DEFINICIÓN DE RUTAS (SINCRONIZADO CON DISCO LOCAL)
# ==========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models", "ai_models")

# Modelo Predictivo Principal (Basado en tu archivo caloric_regressor_final.pkl)
MODEL_PATH = os.path.join(MODELS_DIR, "caloric_regressor_final.pkl")

# Motor de Recomendación Nutricional (CBF)
CBF_MATRIX_PATH = os.path.join(MODELS_DIR, "matrix_nutricion.pkl")
CBF_SCALER_PATH = os.path.join(MODELS_DIR, "scaler_nutricion.pkl")

# Motor de Recomendación Fitness (CBF)
FIT_MATRIX_PATH = os.path.join(MODELS_DIR, "matrix_fitness.pkl")
FIT_SCALER_PATH = os.path.join(MODELS_DIR, "scaler_fitness.pkl")

# Red Neuronal Profunda (ANN)
# Nota: Usamos el archivo .keras según tu carpeta local
ANN_MODEL_PATH = os.path.join(MODELS_DIR, "ann_calories_burned_pro.keras")

# ==========================================================
# CONSTANTES DE ESTADOS DE PLANES NUTRICIONALES
# ==========================================================
ESTADOS_PLAN = {
    "provisional_ia": "Plan generado automáticamente - Pendiente de validación",
    "en_revision": "Nutricionista revisando tu plan",
    "validado": "Plan aprobado por nutricionista",
    "modificado": "Nutricionista realizó ajustes personalizados"
}

# Condiciones médicas que requieren validación obligatoria
CONDICIONES_CRITICAS = [
    "diabetes tipo 1", 
    "insuficiencia renal", 
    "enfermedad cardiovascular",
    "hipertensión severa",
    "embarazo",
    "lactancia",
    "trastorno alimentario",
    "cirugía reciente"
]

class IAService:
    # v43.0: CONSTANTES DE PROCESAMIENTO REUTILIZABLES (Pre-compiladas para velocidad)
    TRADUCTORES_REGIONALES = {
        "ají amarillo": "ají amarillo", "ají panca": "ají panca", "rocoto": "ají", "huacatay": "huacatay",
        "culantro": "cilantro", "papa amarilla": "papa amarilla", "papa blanca": "papa blanca", "chuño": "papa seca",
        "charqui": "carne de res seca", "maíz morado": "maíz morado", "choclo": "choclo", "mote": "maíz mote",
        "cancha": "maíz cancha", "olluco": "olluco", "zapallo loche": "zapallo", "cabrito": "carne de cabrito",
        "cuy": "carne de cuy", "yuyo": "algas", "tarwi": "chocho", "maca": "maca", "quinua": "quinua",
        "plátano verde": "plátano de seda", "plátano maduro": "plátano de seda",
        "tacacho": "plátano, de seda", "cecina": "cerdo, carne magra, cruda*",
        "chancho": "cerdo, carne magra, cruda*", "pechuga": "pollo, carne magra",
        "pollo": "pollo, carne", "asado": "res, carne", "bistec": "res, carne magra",
        "yucca": "yuca, raíz", "yuca": "yuca, raíz", "arroz": "arroz blanco corriente",
        "aceite de oliva": "aceite vegetal de oliva", "aceite de coco": "aceite vegetal de coco",
        "quinoa": "quinua", "jitomate": "tomate", "betabel": "beterraga", "ejote": "vainita",
        "elote": "choclo", "cacahuate": "maní", "puerco": "cerdo", "aguacate": "palta",
        "camote": "camote, raíz", "yuca": "yuca, raíz", "soja": "sillao", "soya": "sillao", "lentejas rojas": "lenteja roja", "lenteja roja": "lenteja roja",
        "lomo": "res, carne magra", "gallina": "gallina, pechuga", "pato": "pato, carne", "avena": "avena, hojuelas", "atun": "pescado atun",
        "calabacín": "zapallito italiano", "calabacin": "zapallito italiano", "salsa de tomate": "tomate, salsa", "pasta de tomate": "tomate, puré",
        "aguacate": "palta", "palta": "palta", "aceite de oliva": "aceite vegetal de oliva", "vinagre de manzana": "vinagre",
        "lenteja": "lentejas chicas cocidas", "lentejas": "lentejas chicas cocidas", "lentejas cocidas": "lentejas chicas cocidas",
        "cebollita china": "cebolla china", "cebolla": "cebolla roja", "jengibre": "kion", "pecanas": "pecana",
        "ajonjoli": "semilla de ajonjolí*", "ajonjolí": "semilla de ajonjolí*", "sillao": "sillao",
        "pepino": "pepinillo sin cáscara", "caldo vegetal": "caldo de pollo, o gallina*", "champiñones": "setas", "quinoa": "quinua",
        "poro": "cebolla blanca", "apio": "apio", "coliflor": "coliflor", "vainitas": "vainita", "pimiento": "pimiento", "zapallito": "zapallito italiano"
    }

    PALABRAS_RUIDO = [
        "picado", "trozos", "cortada", "pelada", "fresca", "fresco", "al gusto", 
        "opcional", "maduros", "verdes", "frescos", "limpio", "limpia", "frescas",
        "picados", "cortados", "en trozos", "grandes", "pequeños", "rebanadas",
        "cocida", "cocidas", "cocido", "cocidos", "crujiente", "lavada", "lavado",
        "verduras", "vegetales", "deshilachada", "deshilachado", "molida", "molido", "picadita", "picadito", "picada", "picado",
        "cocida en agua", "cocidas en agua", "cocido en agua", "en agua", "con agua", "en cubos", "en rodajas", "en juliana", "trozado", "trozada", "fresco", "fresca"
    ]

    # Pre-compilar regex para máximo rendimiento en bucles
    PATRON_INGREDIENTE = re.compile(
        r'(?:^|\r?\n)\s*(?:[-\*•]\s*)?(?:(\d+(?:[.,/]\d+)?)\s*(?:(g|gr|gramos|taza|tazas|unidad|unidades|piezas|pieza|cucharada|cucharadas|cucharadita|cucharaditas|oz|ml|l|kg)\b)?\s*(?:de\s+)?)?([^\n]+)',
        re.MULTILINE | re.IGNORECASE
    )

    def __init__(self):
        # v42.1: MODO ULTRA-FAST STARTUP. No cargar modelos pesados al inicio.
        self.model = None
        self.ann_model = None
        self.nlp = None
        self.cbf_matrix = None
        self.cbf_scaler = None
        self.fit_matrix = None
        self.fit_scaler = None
        
        # Inicializar Groq (Sigue siendo instantáneo)
        if AsyncGroq:
            self.groq_client = AsyncGroq(api_key=settings.GROQ_API_KEY)
        else:
            self.groq_client = None
            print("⚠️ Cliente Groq no inicializado")

        self.setup_fuzzy_logic()
        
        # Cargar Base de Datos de Ejercicios (Biomecánica & METs)
        self.datos_ejercicios = []
        try:
            ruta_ejercicios = os.path.join(BASE_DIR, 'data', 'ejercicios.json')
            if os.path.exists(ruta_ejercicios):
                with open(ruta_ejercicios, 'r', encoding='utf-8') as f:
                    self.datos_ejercicios = json.load(f)
                print(f"✅ Base de Ejercicios cargada: {len(self.datos_ejercicios)} items")
        except: pass
        
        # Centralizar NutricionService
        self.nutricion_service = NutricionService()

    def _ensure_main_model(self):
        """Carga perezosa del regresor calórico."""
        if self.model is None and os.path.exists(MODEL_PATH):
            try:
                print(f"🔬 Cargando Regresor Calórico (Lazy)...")
                self.model = joblib.load(MODEL_PATH)
            except: pass

    def _ensure_ann_model(self):
        """Carga perezosa de la Red Neuronal FitRec."""
        if self.ann_model is None and os.path.exists(ANN_MODEL_PATH):
            try:
                print(f"🧠 Cargando ANN FitRec (Lazy)...")
                self.ann_model = keras.models.load_model(ANN_MODEL_PATH)
            except: pass

    def _ensure_cbf_nutrition(self):
        """Carga perezosa de matrices de recomendación nutricional."""
        if self.cbf_matrix is None and os.path.exists(CBF_MATRIX_PATH):
            try:
                print(f"🍎 Cargando Matrix Nutrición (Lazy)...")
                self.cbf_matrix = joblib.load(CBF_MATRIX_PATH)
                self.cbf_scaler = joblib.load(CBF_SCALER_PATH)
            except: pass

    def _ensure_fit_model(self):
        """Carga perezosa de los modelos de Fitness (CBF)."""
        if self.fit_matrix is None and os.path.exists(FIT_MATRIX_PATH):
            try:
                print(f"🏋️ Cargando Matrix Fitness (Lazy)...")
                self.fit_matrix = joblib.load(FIT_MATRIX_PATH)
                self.fit_scaler = joblib.load(FIT_SCALER_PATH)
            except: pass

    def setup_fuzzy_logic(self):
        """
        Configura el sistema de lógica difusa para personalizar alertas según adherencia y progreso.
        """
        if not (ctrl and np):
            print("⚠️ Lógica Difusa desactivada (Librerías faltantes)")
            self.simulacion_alerta = None
            return

        try:
            # Variables de entrada
            self.adherencia = ctrl.Antecedent(np.arange(0, 101, 1), 'adherencia')
            self.progreso = ctrl.Antecedent(np.arange(0, 101, 1), 'progreso')
        except Exception as e:
            print(f"Error inicializando lógica difusa: {e}")
            self.alerta_sim = None
            return

        # Resto de la config...
        self.alerta_tipo = ctrl.Consequent(np.arange(0, 101, 1), 'alerta_tipo')  # 0=suave, 100=estricta

        # Funciones de membresía
        self.adherencia['baja'] = fuzz.trimf(self.adherencia.universe, [0, 0, 50])
        self.adherencia['media'] = fuzz.trimf(self.adherencia.universe, [25, 50, 75])
        self.adherencia['alta'] = fuzz.trimf(self.adherencia.universe, [50, 100, 100])

        self.progreso['lento'] = fuzz.trimf(self.progreso.universe, [0, 0, 50])
        self.progreso['normal'] = fuzz.trimf(self.progreso.universe, [25, 50, 75])
        self.progreso['rapido'] = fuzz.trimf(self.progreso.universe, [50, 100, 100])

        self.alerta_tipo['suave'] = fuzz.trimf(self.alerta_tipo.universe, [0, 0, 50])
        self.alerta_tipo['moderada'] = fuzz.trimf(self.alerta_tipo.universe, [25, 50, 75])
        self.alerta_tipo['estricta'] = fuzz.trimf(self.alerta_tipo.universe, [50, 100, 100])

        # Reglas difusas
        rule1 = ctrl.Rule(self.adherencia['alta'] & self.progreso['rapido'], self.alerta_tipo['suave'])
        rule2 = ctrl.Rule(self.adherencia['media'] & self.progreso['normal'], self.alerta_tipo['moderada'])
        rule3 = ctrl.Rule(self.adherencia['baja'] | self.progreso['lento'], self.alerta_tipo['estricta'])

        # Sistema de control
        self.alerta_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
        self.alerta_sim = ctrl.ControlSystemSimulation(self.alerta_ctrl)

    def interpretar_comando_nlp(self, texto):
        """
        Usa spaCy para interpretar comandos en lenguaje natural.
        Retorna intent y entities.
        """
        if not self.nlp:
            return {"intent": "desconocido", "entities": {}}

        doc = self.nlp(texto.lower())
        
        # Intents básicos (simplificado, en producción usar un modelo entrenado con SNIPS)
        intents = {
            "perder_peso": ["perder peso", "bajar de peso", "adelgazar"],
            "ganar_peso": ["ganar peso", "aumentar masa", "engordar"],
            "mantener_peso": ["mantener peso", "conservar peso"],
            "ejercicios": ["ejercicio", "rutina", "entrenar", "gimnasio"],
            "nutricion": ["comida", "dieta", "alimentacion", "calorias"]
        }
        
        intent = "general"
        for key, keywords in intents.items():
            if any(keyword in texto for keyword in keywords):
                intent = key
                break
        
        # Extraer entidades (edad, peso, etc.)
        entities = {}
        for ent in doc.ents:
            if ent.label_ == "PERCENT" or "edad" in ent.text:
                entities["edad"] = ent.text
            elif "kg" in ent.text or "peso" in ent.text:
                entities["peso"] = ent.text
        
        return {"intent": intent, "entities": entities}

    def generar_alerta_fuzzy(self, adherencia_pct, progreso_pct):
        """
        Usa lógica difusa para generar alertas personalizadas.
        """
        if not hasattr(self, 'alerta_sim'):
            return "Alerta moderada: Recuerda seguir tu plan."

        self.alerta_sim.input['adherencia'] = adherencia_pct
        self.alerta_sim.input['progreso'] = progreso_pct
        
        try:
            self.alerta_sim.compute()
            tipo_alerta = self.alerta_sim.output['alerta_tipo']
            
            if tipo_alerta < 33:
                return "¡Excelente progreso! Sigue así, campeón."
            elif tipo_alerta < 66:
                return "Vas bien, pero puedes mejorar un poco más."
            else:
                return "Necesitas más compromiso. ¡Vamos, tú puedes!"
        except:
            return "Alerta moderada: Mantén el ritmo."

    # ==========================================================
    # FUNCIONES CENTRALIZADAS - EVITAR DUPLICACIÓN
    # ==========================================================
    
    def _calcular_tmb_harris_benedict(self, genero, edad, peso, talla):
        """
        Fallback: Fórmula Harris-Benedict para TMB cuando el modelo ML falla.
        genero: 1 = Masculino, 2 = Femenino
        """
        if genero == 1:
            tmb = 88.362 + (13.397 * peso) + (4.799 * talla) - (5.677 * edad)
        else:
            tmb = 447.593 + (9.247 * peso) + (3.098 * talla) - (4.330 * edad)
        return round(tmb, 2)
    
    def calcular_macros_optimizados(self, peso, objetivo_key, calorias_diarias, condiciones_medicas=""):
        """
        📐 FUNCIÓN CENTRALIZADA: Calcula macros por g/kg de forma unificada.
        
        Esta función asegura que todos los módulos usen la misma lógica:
        - generar_plan_inicial_automatico
        - recomendar_alimentos_con_groq
        - Dashboard endpoints
        
        Args:
            peso: Peso del cliente en kg
            objetivo_key: Clave del objetivo (perder_agresivo, mantener, ganar_bulk, etc.)
            calorias_diarias: Calorías totales calculadas por el modelo
            condiciones_medicas: String con condiciones médicas del cliente
        
        Returns:
            dict: {"proteinas_g": float, "carbohidratos_g": float, "grasas_g": float, "alerta_medica": str}
        """
        print(f"📐 Calculando macros: Peso={peso}kg, Objetivo={objetivo_key}, Calorías={calorias_diarias}")
        
        # 1. Determinar g/kg según objetivo (Rango Científico: P 1.6-2.2, G 0.7-1.0)
        if "perder" in objetivo_key.lower():
            # Déficit: Maximizamos proteína para proteger músculo, Grasas al mínimo saludable
            g_proteina_kg = 2.2  # Rango alto (2.2g)
            g_grasa_kg = 0.7     # Rango bajo (0.7g) para dar espacio a carbos
        elif "ganar" in objetivo_key.lower():
            # Volumen: Proteína moderada (suficiente), Grasas altas para superávit fácil
            g_proteina_kg = 1.8  # Rango medio (1.8g) es suficiente para crecer
            g_grasa_kg = 1.0     # Rango alto (1.0g)
        else:
            # Mantenimiento: Balance perfecto
            g_proteina_kg = 2.0
            g_grasa_kg = 0.9
        
        # 2. Calcular gramos de proteína y grasa
        proteinas_g = round(peso * g_proteina_kg, 1)
        grasas_g = round(peso * g_grasa_kg, 1)
        
        # 3. Carbohidratos por diferencia (método profesional)
        calorias_p_g = (proteinas_g * 4) + (grasas_g * 9)
        calorias_restantes = max(0, calorias_diarias - calorias_p_g)
        carbohidratos_g = round(calorias_restantes / 4, 1)
        
        # 4. Ajustes por Condiciones Médicas
        alerta_medica = ""
        condiciones = condiciones_medicas.lower()
        
        if "diabetes" in condiciones or "resistencia a la insulina" in condiciones:
            # Límite de seguridad: máximo 3g/kg de carbohidratos
            limite_carbos = peso * 3
            if carbohidratos_g > limite_carbos:
                carbohidratos_g = round(limite_carbos, 1)
                # Recalcular calorías totales
                calorias_ajustadas = (proteinas_g * 4) + (grasas_g * 9) + (carbohidratos_g * 4)
                alerta_medica = f"⚠️ Ajuste por Diabetes: Carbohidratos limitados a {carbohidratos_g}g (Calorías ajustadas a {calorias_ajustadas:.0f}kcal)"
        
        if "hipertensión" in condiciones or "presión alta" in condiciones:
            alerta_medica += " 🧂 REDUCIR SODIO: Evitar procesados y sal de mesa."
        
        print(f"✅ Macros calculados: P={proteinas_g}g, C={carbohidratos_g}g, G={grasas_g}g")
        
        return {
            "proteinas_g": proteinas_g,
            "carbohidratos_g": carbohidratos_g,
            "grasas_g": grasas_g,
            "alerta_medica": alerta_medica
        }

    def calcular_requerimiento(self, genero, edad, peso, talla, nivel_actividad=1.2, objetivo="mantener"):
        """
        Calcula requerimiento calórico usando Gradient Boosting con fallback a Harris-Benedict.
        """
        print(f"🔬 Calculando requerimiento: Género={genero}, Edad={edad}, Peso={peso}, Talla={talla}, Nivel={nivel_actividad}, Objetivo={objetivo}")
        
        # 0. Cálculo Base Harris-Benedict (Baseline de Seguridad)
        basal_hb = self._calcular_tmb_harris_benedict(genero, edad, peso, talla)
        
        self._ensure_main_model()
        if not self.model:
            print("⚠️ Modelo ML no disponible, usando Harris-Benedict como baseline")
            basal = basal_hb
        else:
            try:
                # 1. Predicción con Machine Learning
                df = pd.DataFrame([[genero, edad, peso, talla]], 
                                  columns=['RIAGENDR', 'RIDAGEYR', 'BMXWT', 'BMXHT'])
                
                pred = self.model.predict(df)
                basal_ml = pred.item()
                
                # 🛡️ SANITY CHECK (v1.6): Blindaje Clínico Agresivo
                # Evitar división por cero si basal_hb es 0 (aunque poco probable para TMB)
                if basal_hb == 0:
                    error_relativo = float('inf') # Representa un error muy grande
                else:
                    error_relativo = abs(basal_ml - basal_hb) / basal_hb

                if error_relativo > 0.15: # Desviación mayor al 15%
                    print(f"⚠️ [IA-SHIELD] ML {basal_ml:.0f} vs HB {basal_hb:.0f} ({error_relativo*100:.1f}%) - Desviación excesiva.")
                    # Si el ML falla por mucho, confiamos 95% en Harris-Benedict (valor clínico seguro)
                    basal = (basal_hb * 0.95) + (basal_ml * 0.05)
                    print(f"⚖️ Ajuste clínico aplicado: {basal:.2f} kcal")
                else:
                    basal = basal_ml
                    print(f"✅ TMB calculado por ML: {basal:.2f} kcal")
            except Exception as e:
                print(f"❌ Error en predicción ML: {e}, usando Harris-Benedict")
                basal = basal_hb
        
        mantenimiento = basal * nivel_actividad
        
        # 2. Ajuste por 5 Estados Metabólicos (Granularidad para Tesis)
        ajuste_calorico = {
            "perder_agresivo": -500,     # Déficit Agresivo
            "perder_definicion": -300,   # Definición (Cut)
            "mantener": 0,               # Recomposición
            "ganar_lean_bulk": 250,      # Volumen Limpio
            "ganar_bulk": 500,           # Volumen (Bulk)
            # Mapeo de compatibilidad
            "perder": -500,
            "ganar": 500
        }
        
        offset = ajuste_calorico.get(objetivo.lower(), 0)
        resultado_final = mantenimiento + offset
        
        print(f"📊 Resultado final: TMB={basal:.0f} * {nivel_actividad} + {offset} = {resultado_final:.0f} kcal")
            
        return round(resultado_final, 2)

    def calcular_calorias_quemadas(self, tipo_ejercicio, duracion, intensidad, perfil_usuario):
        """
        Usa la ANN para estimar calorías quemadas.
        Inputs: tipo_ejercicio (int), duracion (float), intensidad (float), perfil_usuario (dict con edad, peso, genero)
        """
        self._ensure_ann_model()
        if not self.ann_model:
            return None
        
        # Preparar input para la ANN (ajusta según FitRec: probablemente [tipo, duracion, intensidad, edad, peso, genero, ...])
        # Asumir 7 features: tipo, duracion, intensidad, edad, peso, genero, intensidad*peso o algo
        edad = perfil_usuario.get('edad', 30)
        peso = perfil_usuario.get('peso', 70)  # Asumir kg
        genero_str = perfil_usuario.get('genero', 'masculino').lower()
        # Mapear género a numérico
        if genero_str in ['masculino', 'hombre', 'm', 'male']:
            genero = 1
        elif genero_str in ['femenino', 'mujer', 'f', 'female']:
            genero = 2
        else:
            genero = 1  # default masculino
        # Normalizar intensidad a 0-1 (en vez de 1-10) para más realismo
        intensidad_normalizada = intensidad / 10.0
        input_data = pd.DataFrame([[tipo_ejercicio, duracion, intensidad_normalizada, edad, peso, genero, intensidad_normalizada * peso]], 
                                  columns=['tipo', 'duracion', 'intensidad', 'edad', 'peso', 'genero', 'intensidad_peso'])
        
        try:
            prediccion = self.ann_model.predict(input_data)
            calorias_raw = float(prediccion[0][0])
            # Ajuste para hacer realista: dividir por 10 (con intensidad normalizada)
            calorias_ajustadas = calorias_raw / 10
            return round(calorias_ajustadas, 2)
        except Exception as e:
            print(f"Error en predicción ANN: {e}")
            return None

    # --- NUEVAS FUNCIONES DE ASISTENCIA (Para tu Tesis) ---

    def generar_sugerencia_entrenamiento(self, objetivo, dia_numero):
        """
        Actúa como respaldo cuando el Coach está ocupado.
        """
        # Días de descanso (6 y 7)
        if dia_numero > 5:
            return "Día de Recuperación: Realiza estiramientos activos y 20 min de caminata suave."
        
        # Días de entrenamiento (1 al 5)
        rutinas = {
            "ganar": "Fuerza e Hipertrofia: Prioriza ejercicios multiarticulares (Sentadillas/Press). 4 series de 8-10 reps.",
            "perder": "Gasto Calórico: Enfoque en circuitos o superseries con descansos cortos (30s) para maximizar la quema.",
            "mantener": "Tonificación: Entrenamiento balanceado de fuerza y cardio moderado (Zona 2)."
        }
        return rutinas.get(objetivo, "Sigue las indicaciones generales del Coach.")

    def recomendar_alimentos_con_groq(self, perfil_usuario, preferencias=None, comando_texto=None, adherencia_pct=50, progreso_pct=50):
        """
        Combina CBF con Groq para recomendaciones de alimentos, ahora con NLP y lógica difusa.
        perfil_usuario: dict con edad, genero, objetivo, etc.
        preferencias: lista de alimentos preferidos/no preferidos.
        comando_texto: texto en lenguaje natural para interpretar intent.
        adherencia_pct: porcentaje de adherencia del usuario (0-100).
        progreso_pct: porcentaje de progreso (0-100).
        """
        # Interpretar comando con NLP si se proporciona
        intent_info = None
        if comando_texto:
            intent_info = self.interpretar_comando_nlp(comando_texto)
            print(f"🔍 Intent detectado: {intent_info}")

            # Ajustar perfil basado en intent
            if intent_info['intent'] == 'perder_peso':
                perfil_usuario['objetivo'] = 'perder'
            elif intent_info['intent'] == 'ganar_peso':
                perfil_usuario['objetivo'] = 'ganar'
            elif intent_info['intent'] == 'mantener_peso':
                perfil_usuario['objetivo'] = 'mantener'

        # Generar alerta personalizada con fuzzy logic
        alerta_personalizada = self.generar_alerta_fuzzy(adherencia_pct, progreso_pct)
        
        # 1. Calcular calorías exactas usando el ML
        genero_map = {"M": 1, "F": 2}
        genero = genero_map.get(perfil_usuario.get('gender', 'M'), 1)
        
        # Obtener factor de actividad real
        nivel_map = {"Sedentario": 1.20, "Ligero": 1.375, "Moderado": 1.55, "Activo": 1.725, "Muy activo": 1.90}
        nivel = nivel_map.get(perfil_usuario.get('activity_level', 'Sedentario'), 1.20)
        
        calorias_reales = self.calcular_requerimiento(
            genero, 
            perfil_usuario.get('age', 25), 
            perfil_usuario.get('weight', 70), 
            perfil_usuario.get('height', 170), 
            nivel, 
            perfil_usuario.get('objetivo', 'mantener')
        )

        # 2. Usar función centralizada para calcular macros
        peso = perfil_usuario.get('weight', 70)
        objetivo = perfil_usuario.get('objetivo', 'mantener')
        condiciones = perfil_usuario.get('medical_conditions', '')
        
        macros_data = self.calcular_macros_optimizados(peso, objetivo, calorias_reales, condiciones)
        prot_g = macros_data['proteinas_g']
        carb_g = macros_data['carbohidratos_g']
        gras_g = macros_data['grasas_g']
        alerta_medica_macros = macros_data['alerta_medica']
        
        # Combinar alertas
        alerta_final = f"{alerta_personalizada}. {alerta_medica_macros}" if alerta_medica_macros else alerta_personalizada

        # Prompt profesional para Tesis - Lógica de Equivalentes Peruanos
        prompt = f"""
        Eres un Nutricionista Colegiado en Perú experto en IA. 
        REQUERIMIENTO: {calorias_reales} kcal | P: {prot_g}g, C: {carb_g}g, G: {gras_g}g.
        
        REGLA DE ORO (MANDATORIA): 
        1. REVISA LA SECCIÓN DE 'PLATOS DISPONIBLES (MUESTRA)' QUE SE TE ENTREGÓ ARRIBA.
        2. SI HAY PLATOS DE LA REGIÓN DEL USUARIO (Selva/Sierra/Costa), ELIGE UNO DE ESOS OBLIGATORIAMENTE.
        3. ¡NO INVENTES PLATOS EXTRANJEROS NI MEZCLAS RARAS! (Chifrijo es de COSTA RICA, NO PERÚ).
        4. Si no hay nada específico, usa tu conocimiento general PERO ADAPTADO (Pollo, Pescado, Huevos, Camote).

        ADAPTACIONES OBLIGATORIAS:
        - Ej: Arándanos -> Aguaymanto/Fresa nacional.
        - Ej: Salmón -> Trucha andina/Bonito/Jurel.
        - Ej: Kale/Greens -> Espinaca/Acelga/Hojas de quinua.
        - Ej: Aceite de Canola -> Aceite de Oliva/Sacha Inchi.
        
        MENÚ PERUANO (5 COMIDAS):
        - Desayuno, Media Mañana, Almuerzo (principal), Media Tarde, Cena.
        - Usa términos locales: palta, camote, papa, choclo, menestras.
        - Indica porciones claras y el aporte calórico por comida.
        - CRÍTICO: Para el Almuerzo y Cena, desglosa la Preparación en MÍNIMO 7 PASOS TÉCNICOS DETALLADOS (ej: maceración, temperatura de sellado, orden de sofrito). Prohibido resumir.
        
        Responde en Markdown y agrega: "{alerta_final}".
        """

        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.7
            )
            recomendacion_nutricion = response.choices[0].message.content.strip()

            # Agregar recomendaciones de ejercicios
            recomendacion_ejercicios = self.recomendar_ejercicios_con_groq(perfil_usuario, preferencias)
            recomendacion_completa = f"{recomendacion_nutricion}\n\n**Recomendaciones de Ejercicios:**\n{recomendacion_ejercicios}\n\n**Alerta Personalizada:**\n{alerta_personalizada}"
            return recomendacion_completa
        except Exception as e:
            print(f"Error con Groq: {e}")
            return "Recomendación básica: Incluye proteínas magras, vegetales y carbohidratos complejos."

    def recomendar_ejercicios_con_groq(self, perfil_usuario, preferencias=None):
        """
        Recomienda ejercicios usando CBF de fitness + Groq.
        """
        ejercicios_base = []
        self._ensure_fit_model()
        if self.fit_matrix is not None and self.fit_scaler is not None:
            try:
                # Vector de usuario basado en perfil (ajusta features según tu scaler)
                objetivo = perfil_usuario.get('objetivo', 'mantener')
                if objetivo == 'perder':
                    user_values = [30, 1.5]  # Ej. edad, intensidad (ajusta a 2 features)
                elif objetivo == 'ganar':
                    user_values = [30, 2.0]
                else:
                    user_values = [30, 1.7]
                user_vector = pd.DataFrame([user_values])
                user_scaled = self.fit_scaler.transform(user_vector)
                similarities = cosine_similarity(user_scaled, self.fit_matrix)[0]
                top_indices = similarities.argsort()[-5:][::-1]
                # Usar nombres genéricos ya que no tenemos dataset
                ejercicios_base = [f"Ejercicio #{i+1}" for i in top_indices]
                print(f"✅ CBF Fitness recomendó: {ejercicios_base}")
            except Exception as e:
                print(f"❌ Error en CBF Fitness: {e}")
                ejercicios_base = ["Caminata 30 min", "Flexiones", "Sentadillas"]
        else:
            ejercicios_base = ["Caminata 30 min", "Flexiones", "Sentadillas", "Plancha", "Saltos"]

        # Mapear ejercicios a IDs para ANN (ajusta según tu dataset FitRec)
        ejercicio_id_map = {
            "Single-cone sprint drill": 1, "Carrera de velocidad": 1,
            "In-out jump squat": 2, "Saltos de piernas": 2,
            "Gorilla squat": 3, "Sentadillas": 3,
            "Burpee tuck jump": 4, "Burpees": 4,
            "Linear 3-Part Start Technique": 5, "Ejercicio de velocidad lineal": 5,
            "Caminata 30 min": 6, "Flexiones": 7, "Plancha": 8, "Saltos": 9
        }

        # Calcular calorías para cada ejercicio usando ANN
        ejercicios_con_calorias = []
        for ej in ejercicios_base:
            ej_id = ejercicio_id_map.get(ej, 1)  # Default a 1 si no mapeado
            calorias = self.calcular_calorias_quemadas(ej_id, 30, 5, perfil_usuario)  # 30 min, intensidad media, perfil
            ejercicios_con_calorias.append(f"{ej} (~{calorias} calorías)" if calorias else f"{ej}")

        # Prompt para Groq Fitness - Adaptación al contexto nacional
        prompt = f"""
        Eres un Entrenador Personal experto. Perfil: {perfil_usuario}.
        Genera una rutina diaria para el objetivo: '{objetivo}'.
        
        REGLA DE CONTEXTO:
        Adapta los ejercicios a lo que un usuario en Perú suele hacer. 
        Usa nombres estándar pero considera el entorno:
        - Gimnasio (pesas, máquinas).
        - Espacios públicos (trote en parque, calistenia).
        - Deportes comunes (Fútbol, Vóley, Natación, Baile).
        
        Estructura la rutina con Calentamiento, Parte Principal y Estiramiento. 
        Usa lenguaje profesional en español.
        """
        try:
            response = self.groq_client.chat.completions.create(
                model="gemma2-9b-it",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.7
            )
            recomendacion = response.choices[0].message.content.strip()
            return recomendacion
        except Exception as e:
            print(f"Error con Groq en ejercicios: {e}")
            return "Rutina básica: Caminata 30 min, flexiones 3x10, sentadillas 3x15."

    def generar_nota_contingencia(self, es_contingencia):
        """
        Mensaje para cuando el Nutricionista no pudo validar o el cliente faltó a la cita.
        """
        if es_contingencia:
            return "Asistente IA: Tu nutricionista no ha podido validar esta semana aún. He generado este plan de mantenimiento para que no pierdas tu ritmo."
        return "Plan validado. Sigue las recomendaciones para alcanzar tu meta semanal."

    def generar_insight_diario(self, perfil_usuario, consumo_actual):
        """
        Genera una frase corta de insight basada en el consumo real vs la meta.
        consumo_actual: dict {'calorias': 1500, 'proteinas': 50, 'carbos': 200, 'grasas': 40}
        """
        # 1. Calcular la meta usando tu modelo de Gradient Boosting ya cargado
        meta_calorias = self.calcular_requerimiento(
            genero=1 if perfil_usuario['gender'] == 'M' else 2,
            edad=perfil_usuario['age'],
            peso=perfil_usuario['weight'],
            talla=perfil_usuario['height'],
            objetivo=perfil_usuario['goal']
        )

        # 2. Lógica de comparación
        # Evitar división por cero si meta_calorias es 0
        if meta_calorias == 0:
            pct_calorias = 0
        else:
            pct_calorias = (consumo_actual['calorias'] / meta_calorias) * 100
        
        # v2.0: "GPS Financiero" - Prioridad Proteína
        peso_usuario = perfil_usuario.get('weight', 70)
        meta_proteina = peso_usuario * 2.0 # Meta base estándar
        consumo_proteina = consumo_actual.get('proteinas', 0)
        # Evitar división por cero si meta_proteina es 0
        if meta_proteina == 0:
            pct_proteina = 0
        else:
            pct_proteina = (consumo_proteina / meta_proteina) * 100
        
        contexto_extra = ""
        if pct_proteina > 80:
            contexto_extra = "IMPORTANTE: El usuario cumplió su meta de PROTEÍNA. Si se pasó de calorías, SÉ PERMISIVO. Felicítalo por la proteína."
        else:
            contexto_extra = "ALERTA: Proteína baja. Sugiere alimentos ricos en proteína para la cena."

        # 3. Construir el prompt para Groq (enfocado en una frase corta)
        prompt = f"""
        Eres un coach de salud experto en Nutrición Moderna (No dieta vieja escuela).
        Usuario: {perfil_usuario['first_name']}. 
        Meta Calórica: {meta_calorias:.0f}. Consumo: {consumo_actual['calorias']:.0f} ({pct_calorias:.1f}%).
        Meta Proteína: {meta_proteina:.0f}g. Consumo: {consumo_proteina:.0f}g ({pct_proteina:.1f}%).
        
        CONTEXTO CLAVE: {contexto_extra}
        TIMING SUGERIDO: Si es medio día, sugiere carbos pre-entreno. Si es noche, proteína post-entreno.
        
        Genera un 'insight' de UNA SOLA FRASE (máximo 15 palabras). Haz que cada palabra cuente.
        
        Genera un 'insight' de UNA SOLA FRASE (máximo 15 palabras). 
        Si el % es > 90, advierte sobre el límite. Si es < 50, motiva a comer más proteína.
        Sé muy específico y usa un tono profesional pero amigable.
        """

        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=60,
                temperature=0.5
            )
            return response.choices[0].message.content.strip()
        except:
            return f"¡Vas por buen camino, {perfil_usuario['first_name']}! Sigue hidratándote y cumpliendo tus metas."

# Singleton
    async def extraer_macros_de_texto(self, texto: str, peso_usuario_kg: float = 70.0):
        """
        Usa Groq para extraer información nutricional con RIGOR CIENTÍFICO.
        """
        # Prompt Ingeniería de Datos para Nutrición de Precisión
        prompt = f"""
        Actúa como un Nutricionista Deportivo y Científico de Datos experto.
        Analiza el siguiente texto: "{texto}" y extrae la información con MÁXIMA PRECISIÓN CIENTÍFICA.
        
        Tus fuentes de verdad son:
        1. USDA FoodData Central (para nutrición).
        2. Compendium of Physical Activities 2011 (Ainsworth et al.) para ejercicios (METs).
        
        PESO DEL USUARIO: {peso_usuario_kg} kg (Úsalo para calcular calorías quemadas).
        
        Debes responder ÚNICAMENTE en JSON con esta estructura exacta:
        {{
            "alimentos_detectados": ["Nombre alimento 1"],
            "ejercicios_detectados": ["Nombre ejercicio 1"],
            "calorias": 0,
            "proteinas_g": 0.0,
            "carbohidratos_g": 0.0,
            "grasas_g": 0.0,
            "fibra_g": 0.0,
            "azucar_g": 0.0,
            "es_comida": true,
            "es_ejercicio": false,
            "calidad_nutricional": "Alta/Media/Baja",
            "metodologia": "Breve explicación científica"
        }}
        
        REGLAS CIENTÍFICAS OBLIGATORIAS:
        
        1. 🥘 PLATOS COMPLEJOS Y PERUANOS (Ingeniería Inversa):
           - Desglosa ingredientes. Ejemplo "Arroz con Pato": Pato (Grasa/Prot) + Arroz (Carb) + Cerveza (Azúcar).
           
        2. 🔢 MATEMÁTICA DE CALORÍAS (CONSISTENCIA):
           - Las calorías NO pueden ser inventadas. Deben cuadrar matemáticamente:
           - Fórmula: (Proteína * 4) + (Carbos * 4) + (Grasas * 9).
           - Si hay Fibra: Restar 2 kcal por cada gramo de fibra (aprox).
           - Si hay Alcohol: Sumar 7 kcal por gramo.
           
        3. 🍎 PARA COMIDA (Base de datos USDA):
           - NO INVENTES VALORES. Usa promedios estándar de USDA.
           - Ejemplo: Pechuga de pollo cocida ≈ 31g proteína/100g. Arroz cocido ≈ 28g carbs/100g.
           - Manzana mediana (182g) ≈ 95 kcal, 0.5g P, 25g C, 0.3g G.
           - Si no se especifica cantidad, asume: "1 porción" o "unidad mediana" o "taza (150-200g)".
           - Incluye SIEMPRE Fibra y Azúcar si el alimento lo tiene (Frutas, granos, procesados).
           - CALIDAD NUTRICIONAL:
             * Alta: Comida real (Vegetales, Carnes magras, Frutas, Huevos, Avena).
             * Media: Procesados simples (Pan, Jamón, Queso, Batidos).
             * Baja: Ultraprocesados (Gaseosa, Galletas, Frituras, Dulces).
        
        4. 🏃 PARA EJERCICIO (Fórmula METs):
           - Fórmula: Calorías = METs * Peso({peso_usuario_kg}kg) * Tiempo(horas).
           - Usa estos METs de referencia (Compendium 2011):
             * Caminar moderado: 3.5 METs
             * Correr (8 km/h): 8.3 METs
             * Pesas/Gimnasio intenso: 6.0 METs
             * Yoga: 2.5 METs
             * Fútbol/Basket: 7.0 - 8.0 METs
           - SIEMPRE calcula: MET * {peso_usuario_kg} * (minutos/60).
           - Ejemplo: "Corrí 30 min" (8.3 METs) -> 8.3 * {peso_usuario_kg} * 0.5 = 290.5 kcal.
           - NO DEVUELVAS 0. Si dice "Corrí", "Caminé", "Entrené", CALCULA.
        
        3. 🚫 SI NO HAY DATOS CLAROS:
           - Si el texto es ambiguo ("hola", "tengo hambre"), devuelve todo en 0.
        """
        try:
            response = await self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",  # v44.1: Corregido (gemma2-9b-it fue deprecado)
                messages=[{"role": "user", "content": prompt}],
                max_tokens=250,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            response_json = json.loads(response.choices[0].message.content.strip())
            
            # v69.3: Limpieza profunda de nombres (eliminar ** que la IA a veces inyecta)
            if "alimentos_detectados" in response_json:
                response_json["alimentos_detectados"] = [a.replace("*", "").strip() for a in response_json["alimentos_detectados"]]
            if "ejercicios_detectados" in response_json:
                response_json["ejercicios_detectados"] = [e.replace("*", "").strip() for e in response_json["ejercicios_detectados"]]
            
            # 🚀 ENHANCEMENT v2.0: CORRECCIÓN CON DATOS REALES (SQLite/JSON)
            # La IA es buena detectando nombres, pero mala con números exactos de marcas.
            # Usamos el NutricionService para corregir los valores.
            
            if self.nutricion_service and response_json.get("alimentos_detectados"):
                print("🔍 IA: Corrigiendo macros con datos reales de la Base de Datos...")
                
                # Reiniciar contadores para sumar con precisión
                total_cal = 0
                total_prot = 0
                total_carb = 0
                total_gras = 0
                total_azucar = 0
                total_fibra = 0
                total_sodio = 0
                
                alimentos_corregidos = []
                
                for alimento_nombre in response_json["alimentos_detectados"]:
                    # v70.0: Usar el motor único para coherencia absoluta
                    res = self._calcular_exacto_de_alimento(alimento_nombre, "1", "unidad")
                    
                    if res["encontrado"]:
                        print(f"✅ DATO REAL ENCONTRADO (v70.0): {alimento_nombre} -> {res['cals']} kcal")
                        total_cal += res["cals"]
                        total_prot += res["prot"]
                        total_carb += res["carb"]
                        total_gras += res["gras"]
                        alimentos_corregidos.append(f"{alimento_nombre} (Verificado)")
                    else:
                        print(f"⚠️ No encontrado en BD: {alimento_nombre}. Manteniendo estimación IA.")
                        pass

                # LÓGICA FINAL DE REEMPLAZO SEGURA
                # Solo reemplazamos si hemos encontrado TODOS los alimentos detectados en nuestra BD.
                # Si falta alguno, es más seguro confiar en la estimación total de Groq.
                if len(alimentos_corregidos) == len(response_json["alimentos_detectados"]) and total_cal > 0:
                    print(f"📊 ACTUALIZANDO con precisión (100% Match): Antes {response_json['calorias']} -> Ahora {total_cal}")
                    response_json["calorias"] = float(round(total_cal, 1))
                    response_json["proteinas_g"] = float(round(total_prot, 1))
                    response_json["carbohidratos_g"] = float(round(total_carb, 1))
                    response_json["grasas_g"] = float(round(total_gras, 1))
                    response_json["azucar_g"] = float(round(total_azucar, 1))
                    response_json["fibra_g"] = float(round(total_fibra, 1))
                    response_json["sodio_mg"] = float(round(total_sodio, 1))
                    # v69.0: NO inyectar textos largos aquí, rompen la UI del móvil.
                    # El campo 'calidad_nutricional' debe ser corto: "Alta", "Media" o "Baja".
                    # response_json["calidad_nutricional"] += " (Verificado...)" <- ELIMINADO
                else:
                     print(f"⚠️ Cobertura parcial ({len(alimentos_corregidos)}/{len(response_json['alimentos_detectados'])}). Usando estimación IA para evitar subestimación.")

            return response_json
        except Exception as e:
            print(f"Error CRÍTICO extrayendo macros con Groq o BD: {type(e).__name__}: {e}")
            # Retornar estructura vacía para no romper el front
            return {
                "calorias": 0, "proteinas_g": 0, "carbohidratos_g": 0, "grasas_g": 0,
                "alimentos_detectados": [], "tipo_detectado": "error", "calidad_nutricional": "Desconocida"
            }
            return None

    async def identificar_intencion_salud(self, mensaje: str) -> dict:
        """
        Nivel 0: Detector Rápido de Riesgos (Async).
        """
        prompt = f"""ANALIZA SI EL SIGUIENTE MENSAJE DESCRIBE UN PROBLEMA DE SALUD FÍSICO O MENTAL:
        Mensaje: "{mensaje}"
        
        RESPONDE EN FORMATO JSON VÁLIDO:
        {{
            "tiene_alerta": true,
            "tipo": "lesion" | "fatiga" | "malestar" | "desanimo" | "enfermedad" | "otro",
            "descripcion_resumida": "Breve descripción",
            "severidad": "bajo" | "medio" | "alto",
            "recomendacion_contingencia": "Sugerencia profesional"
        }}
        Si no hay alerta, responde con {{"tiene_alerta": false}}.
        """
        try:
            response = await self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.1
            )
            respuesta_texto = response.choices[0].message.content.strip()
            
            import json
            import re
            json_match = re.search(r'\{.*\}', respuesta_texto, re.DOTALL)
            if json_match:
                resultado = json.loads(json_match.group())
                return resultado
            return {"tiene_alerta": False}
        except Exception as e:
            print(f"⚠️ [IA-HEALTH-ERROR]: {e}")
            return {"tiene_alerta": False}

    async def asistir_cliente(self, contexto: str, mensaje_usuario: str, historial: list = None, tono_aplicado: str = "") -> str:
        """
        Consulta a Groq con un contexto adaptativo (Async).
        """
        import time
        import random
        import json
        import re
        t_inicio = time.time()
        print(f"� [IA-START] Procesando petición: {mensaje_usuario[:40]}...")

        # 0. Preparar Contexto Dinámico (Reducido para velocidad)
        texto_extra = ""
        msg_low = mensaje_usuario.lower()
        
        ejercicios_local = list(self.datos_ejercicios)

        # 🚀 NUEVO: DETECCIÓN DE LESIONES PARA FILTRADO DE EJERCICIOS
        lesiones_detectadas = []
        keywords_rodilla = ["rodilla", "menisco", "ligamento cruzado", "patela"]
        keywords_espalda = ["espalda", "lumbar", "hernia", "columna", "ciática"]
        keywords_hombro = ["hombro", "manguito", "clavícula"]
        
        full_text_input = (mensaje_usuario + " " + contexto).lower()
        
        if any(k in full_text_input for k in keywords_rodilla): lesiones_detectadas.append("RODILLA")
        if any(k in full_text_input for k in keywords_espalda): lesiones_detectadas.append("ESPALDA/LUMBAR")
        if any(k in full_text_input for k in keywords_hombro): lesiones_detectadas.append("HOMBRO")

        try:
            # --- BASE DE DATOS DE EJERCICIOS (Con filtrado por lesión) ---
            if ejercicios_local:
                # Si hay lesiones, filtramos ejercicios que impacten esas zonas
                if lesiones_detectadas:
                    texto_extra += f"\n\n⚠️ PRECAUCIÓN FÍSICA: Se ha detectado molestia/lesión en: {', '.join(lesiones_detectadas)}."
                    texto_extra += "\n- PRIORIZA: Movilidad, estiramientos y ejercicios de bajo impacto."
                    texto_extra += "\n- EVITA: Saltos, cargas axiales pesadas o movimientos con dolor."
                    
                    # Filtrado simple: si es rodilla, quitamos sentadillas pesadas o pichanga
                    if "RODILLA" in lesiones_detectadas:
                        ejercicios_local = [e for e in ejercicios_local if e.get('id') not in ['sentadilla_barra', 'pichanga_futbol', 'burpees', 'estocadas']]
                    if "ESPALDA/LUMBAR" in lesiones_detectadas:
                        ejercicios_local = [e for e in ejercicios_local if e.get('id') not in ['peso_muerto_convencional', 'sentadilla_barra', 'press_militar_parado']]
                
                es_gym = any(k in msg_low for k in ["gym", "gimnasio", "pesas", "musculo", "fuerza", "hipertrofia"])
                
                gold_standard = [e for e in ejercicios_local if e.get('origen') == 'gold_standard']
                peruanos = [e for e in ejercicios_local if e.get('origen') == 'peru_lifestyle']
                otros = [e for e in ejercicios_local if e.get('origen') == 'dataset_importado']
                
                # Muestra estratégica
                es_casa = any(k in msg_low for k in ["casa", "hogar", "home", "departamento", "cuarto", "habitación", "pequeño"])
                
                if es_gym:
                    muestra_ej = gold_standard[:4] + otros[:2] # Reducido
                    texto_extra += "\n### CONTEXTO GIMNASIO ACTIVO: Sugiere máquinas, pesas o cardio indoor. PROHIBIDO: Fútbol/Pichanga."
                elif es_casa:
                    muestra_ej = gold_standard[:4] + peruanos[:1] # Reducido
                    texto_extra += "\n### CONTEXTO CASA ACTIVO: Sugiere ejercicios con peso corporal o espacio reducido."
                else:
                    muestra_ej = peruanos[:2] + gold_standard[:3] # Reducido
                
                # ESCAPAR LLAVES PARA EVITAR ERROR EN F-STRING
                ejercicios_str = json.dumps(muestra_ej, ensure_ascii=False)
                texto_extra += "\n### BASE DE DATOS DE EJERCICIOS (MUESTRA):\n"
                texto_extra += ejercicios_str.replace("{", "{{").replace("}", "}}")
                texto_extra += "\n(IMPORTANTE: Prioriza Desayunos ligeros si es mañana. EVITA Platos de almuerzo pesados)."

            # 🛡️ NUEVO: CAPA DE SEGURIDAD MÉDICA PROACTIVA
            context_low = contexto.lower()
            condiciones_detectadas = [c for c in CONDICIONES_CRITICAS if c in context_low]
            if condiciones_detectadas:
                condiciones_str = ", ".join(condiciones_detectadas).upper()
                texto_extra += f"\n\n🚨 ALERTA MÉDICA CRÍTICA: El usuario tiene las siguientes condiciones: {condiciones_str}."
                texto_extra += "\n- ESTÁ PROHIBIDO dar consejos médicos diagnósticos o cambios drásticos sin supervisión médica."
                texto_extra += "\n- INCLUYE SIEMPRE un disclaimer breve al inicio: 'Leonardo, dado que tienes [Condición], recuerda validar esto con tu médico...'"
                texto_extra += "\n- SE CONSERVADOR con el sodio (hipertensión) y azúcar/carbohidratos (diabetes)."

        except Exception as e:
            print(f"Error preparando contexto cultural/médico: {e}")

        # --- LÓGICA DE EMERGENCIA VEGANA (FUERA DEL TRY) ---
        if "vegano" in contexto.lower() or "vegetariano" in contexto.lower():
             texto_extra += "\n\n⛔ ALERTA VEGANA CRÍTICA: El usuario es VEGANO/VEGETARIANO. PROHIBIDO: Carne, Pollo, Pescado, Huevos, Leche, Queso, Miel. ¡NI UNA SOLA TRAZA! Usa: Tofu, Soya, Quinua, Menestras, Seitán."

        # (v11.5 - Prompt Dinámico de Intención)
        es_consulta_info = False
        es_saludo = False
        
        keywords_info = ["cuantas calorias", "qué es", "que es", "beneficios", "propiedades", "engorda", "adelgaza", "información", "tengo", "puedo"]
        keywords_accion = ["receta", "preparar", "cocinar", "plato", "menú", "menu", "desayuno", "almuerzo", "cena", "rutina", "entrenamiento", "ejercicios", "plan", "dieta", "sugerencia", "opcion", "dame"]
        keywords_saludo = ["hola", "buen", "hey", "salu", "que tal", "qué tal", "gracias", "chau", "adiós", "adios"]
        keywords_chat_corto = ["si", "sí", "no", "ok", "vale", "dale", "bueno", "perfecto", "entendido", "vaya", "genial", "claro", "por supuesto"]
        keywords_opciones = ["opciones", "opcion", "alternativas", "alternativa", "variedades", "variedad", "sugerencias", "algunas"]
        
        msg_low = mensaje_usuario.lower().strip()
        
        # 1. Detectar si es un saludo o respuesta corta
        if len(msg_low) < 5 or (len(msg_low) < 15 and any(s in msg_low for s in keywords_saludo + keywords_chat_corto)):
            es_saludo = True
        
        # 2. Detectar si es consulta de info
        if any(ki in msg_low for ki in keywords_info) and not any(ka in msg_low for ka in keywords_accion):
            es_consulta_info = True
        
        # v44.0: DETECCIÓN DE TIPO PARA max_tokens DINÁMICO
        es_multiopcion = any(k in msg_low for k in keywords_opciones)
        es_accion = any(k in msg_low for k in keywords_accion)
        
        # Calcular max_tokens según complejidad de la solicitud (↓ tokens = ↓ tiempo)
        if es_saludo:
            max_tokens_ia = 300
        elif es_consulta_info:
            max_tokens_ia = 600
        elif es_multiopcion:
            max_tokens_ia = 1400  # 2 opciones en lugar de 3 (más rápido)
        elif es_accion:
            max_tokens_ia = 1000  # Una sola receta/rutina detallada
        else:
            max_tokens_ia = 700   # Chat general

        if es_consulta_info or es_saludo:
            system_content = """ERES UN ASISTENTE DE NUTRICIÓN EXPERTO.
### CONTEXTO DEL USUARIO Y STATUS DIARIO:
{contexto}
{texto_extra}
TU META: Responder al usuario de forma amable, breve y empática.
USA EL TAG [CALOFIT_INTENT: CHAT] AL INICIO DE TU RESPUESTA.

### 🛡️ SEGURIDAD MÉDICA (CRÍTICO):
- Si el usuario tiene condiciones críticas listadas, SE CAUTELOSO. 
- NO recetes medicamentos ni sugieras ayunos extremos.
- Si es un saludo, responde cálidamente.
- Si es una duda, responde de forma DIRECTA Y CIENTÍFICA.
- NO uses etiquetas como 'Plato:', 'Rutina:', [CALOFIT_HEADER] o [CALOFIT_STATS].
Respuestas en texto plano sin formatos complejos.""".replace("{contexto}", contexto).replace("{texto_extra}", texto_extra)
        else:
            system_content = """OPERANDO BAJO EL PROTOCOLO 'CALOFIT UNIFIED V3.1' (PROTOCOLO INQUEBRANTABLE).
### REGLA DE ORO #0:
TODA receta o rutina DEBE usar obligatoriamente los bloques [CALOFIT_INTENT], [CALOFIT_HEADER], [CALOFIT_LIST] y [CALOFIT_ACTION].
PROHIBIDO responder en texto plano sin estas etiquetas bajo ninguna circunstancia (incluso si pides dieta o bajo en calorías).

### ESTATUS DEL USUARIO:
{contexto}
{texto_extra}

### 📏 REGLA DE UNIDADES (OBLIGATORIO):
USA ÚNICAMENTE UNIDADES MÉTRICAS:
- SÓLIDOS: **g** (gramos). NO USES TAZAS.
- LÍQUIDOS/PURÉS: **ml** (mililitros). NO USES TAZAS.
- PEQUEÑAS CANTIDADES: **cda** (cucharada) o **cdta** (cucharadita).
- UNIDADES: **unidad** (ej: 1 unidad de huevo).

### 🏷️ REGLA MAESTRA DE CATEGORIZACIÓN (OBLIGATORIO):
Toda respuesta DEBE comenzar con una etiqueta de intención exacta. NO USES OTRAS ETIQUETAS.
**DETECCIÓN AUTOMÁTICA DE INTENCIÓN:**
- SI el usuario menciona: "rutina", "ejercicio", "entrenamiento", "workout", "gimnasio", "minutos" (en contexto fitness) -> USA [CALOFIT_INTENT: ITEM_WORKOUT]
- SI el usuario menciona: "receta", "plato", "comida", "almuerzo", "cena", "desayuno" -> USA [CALOFIT_INTENT: ITEM_RECIPE]

### 🚨 PROTOCOLO DE MÚLTIPLES CARTAS (CRÍTICO):
1. **SI EL USUARIO PIDE "OPCIONES" (Plural):**
   - GENERA EXACTAMENTE 2 opciones distintas (no más, para ser rápido y preciso).
   - ⚠️ REGLA DE ORO: CADA OPCIÓN DEBE SER UN BLOQUE INDEPENDIENTE.
   - PRIMERO: Saluda y presenta las opciones con un bloque [CALOFIT_INTENT: CHAT].
     Ejemplo: [CALOFIT_INTENT: CHAT] ¡Claro Leonardo! Aquí tienes 2 excelentes opciones para tu cena: [/CALOFIT_INTENT: CHAT]
   
   - LUEGO: Genera las tarjetas (Exactamente 2):
     [CALOFIT_INTENT: ITEM_RECIPE]
     [CALOFIT_HEADER] Nombre del Plato 1 [/CALOFIT_HEADER]
     [CALOFIT_LIST] Ingredientes... [/CALOFIT_LIST]
     [CALOFIT_ACTION] Pasos... [/CALOFIT_ACTION]

   - ⛔ PROHIBIDO: Escribir "Opción 1:", "Opción 2:" o "Opción 3:" al final del bloque CHAT o dentro del [CALOFIT_HEADER]. SOLO el nombre del plato.
   - ⚠️ CONSISTENCIA (CRÍTICO): Genera el [CALOFIT_LIST] con ingredientes REALES. El sistema calculará las calorías exactas basándose en tu lista. Si incluyes Lomo Saltado, asegúrate de listar carne, papa, arroz y vegetales en cantidades precisas (ej: 150g de carne, 1 taza de arroz) para que el registro posterior coincida con lo mostrado.
   - ⛔ PROHIBIDO: Escribir "Ingredientes:" o "Preparación:" DENTRO de los bloques.
   - ⚠️ CONTROL CALÓRICO: Si el usuario pidió "ligero" o "baja caloría", CADA OPCIÓN debe tener raciones pequeñas y macros precisos.

**ETIQUETAS VÁLIDAS:**
[CALOFIT_INTENT: CHAT] -> Para saludos o consejos cortos.
[CALOFIT_INTENT: ITEM_RECIPE] -> Para una receta detallada (usar etiquetas blindadas).
[CALOFIT_INTENT: ITEM_WORKOUT] -> Para una rutina detallada (usar etiquetas blindadas).

### 🛡️ SEGURIDAD Y REALIDAD:
1. LESIONES: Si menciona dolor de rodilla/espalda, PROHIBIDO impacto (saltos, correr). Sugiere movilidad y fuerza isométrica.
2. ALERGIAS E INTOLERANCIAS (CRÍTICO): Revisa el perfil del usuario. SI es alérgico a un alimento (ej. Maní, Mariscos), PROHIBIDO incluirlo en la lista de ingredientes o preparación.
3. CONDICIONES CRÍTICAS: Si tiene Diabetes/Hipertensión, prioritiza dietas bajas en azúcar/sodio y añade disclaimer médico.
4. CONOCIMIENTO PERUANO Y CASERO (CRÍTICO): 
   - SIEMPRE enfatiza productos del día a día: **Avena**, **Atún en lata**, **Huevos**, **Arroz**, **Pan**.
   - PLATOS FUERTES PERUANOS: Recomienda **Lomo Saltado**, **Ají de Gallina**, **Arroz con Pato**, **Seco de Pollo**, **Lentejas con Arroz**.
   - ⚠️ ADAPTACIÓN VEGANA: Si el perfil es Vegano, transforma los platos: "Saltado de Seitán" (en lugar de Lomo), "Ají de Setas" (en lugar de Gallina), "Seco de Tofu".
   - 💡 CONSULTA DE PLATOS: Si el usuario pregunta por un plato específico, NO respondas solo con texto genérico. Trátalo como un [CALOFIT_INTENT: ITEM_RECIPE], ofrece una versión balanceada del plato.

### 🥗 ESTRUCTURA OBLIGATORIA PARA RECETAS:
    - ⚠️ PORCIONES (CRÍTICO): Todas las recetas y porciones DEBEN ser calculadas para **1 PERSONA** (un solo comensal). Si el usuario pide para más, multiplica las cantidades, pero por defecto siempre para 1.
    - Lista exacta con cantidades lógicas para 1 persona:
      * Sólidos/Proteínas: SIEMPRE en **gramos (g)**.
      * Líquidos Grandes (agua, caldo, leche): **ml**. (NO USAR TAZAS).
      * Aliños/Grasas (aceite, limón, vinagre): **cucharadas (cda)** o **cucharaditas (cdta)**.
    - 🥣 SOPAS Y CALDOS (OBLIGATORIO): Si sugieres una sopa o caldo, DEBES incluir "agua" o "caldo" como ingrediente base en la lista y mencionar el proceso de hervido en la preparación.
    - ❌ PROHIBIDO: Escribir la palabra "Ingredientes:".
2. [CALOFIT_ACTION] = **PREPARACIÓN** (OBLIGATORIO):
   - Pasos breves y lógicos. Para sopas, incluye siempre el paso de "hervir".
   - ❌ PROHIBIDO: Escribir la palabra "Preparación:".
   - OBLIGATORIO: Pasos numerados (1., 2., 3.) de la receta del Header.
   - ❌ PROHIBIDO: Escribir la palabra "Preparación:" o "Pasos:" al inicio.

### 🏋️ ESTRUCTURA OBLIGATORIA PARA RUTINAS (FALLO = PENALIZACIÓN):
⚠️ CADA SECCIÓN DEBE TENER CONTENIDO REAL. SI UNA SECCIÓN ESTÁ VACÍA, LA RESPUESTA SERÁ RECHAZADA.

1. [CALOFIT_LIST] = **CIRCUITO** (QUÉ ejercicios hacer):
   - OBLIGATORIO: Lista de 4-6 ejercicios ESPECÍFICOS con series y repeticiones
   - Ejemplo CORRECTO:
     - 3 series x 15 Sentadillas
     - 3 series x 12 Flexiones de pecho
     - 3 series x 20 Mountain Climbers
     - 2 series x 30seg Plancha frontal
   - ❌ PROHIBIDO dejar vacío o poner solo "Realiza los ejercicios"

2. [CALOFIT_ACTION] = **INSTRUCCIONES TÉCNICAS** (CÓMO hacer cada ejercicio):
   - OBLIGATORIO: Explicación técnica paso a paso de LA EJECUCIÓN de los ejercicios del circuito
   - Ejemplo: "1. Sentadilla: Baja la cadera hasta 90 grados manteniendo el peso en los talones..."
   - ❌ PROHIBIDO: Copiar y pegar el mismo texto para todos los ejercicios.

### 🏷️ ESTRUCTURA DE ETIQUETAS (FALLORES CERO):
[CALOFIT_HEADER] Nombre del Entrenamiento [/CALOFIT_HEADER]
[CALOFIT_STATS] Cal: XXXkcal | Tiempo: XX min [/CALOFIT_STATS]
[CALOFIT_LIST] ... [/CALOFIT_LIST]
[CALOFIT_ACTION] ... [/CALOFIT_ACTION]
[CALOFIT_FOOTER] ... [/CALOFIT_FOOTER]
""".replace("{contexto}", contexto).replace("{texto_extra}", texto_extra)

        # Buscamos en todo el contexto disponible (System Content + Mensaje Usuario)
        contexto_total = (str(system_content) + " " + str(mensaje_usuario)).lower()
        
        # 🥣 DETECCIÓN DE PREFERENCIAS DE VOLUMEN/CALORÍAS (v5.6)
        preferencias_ia = []
        msg_low = mensaje_usuario.lower()
        
        usuario_goal = "mantener"
        if "perder" in contexto.lower(): usuario_goal = "perder"
        elif "ganar" in contexto.lower() or "volumen" in contexto.lower(): usuario_goal = "ganar"
        
        match_cal = re.search(r'Restante:\s*(\d+)', contexto)
        cal_objetivo = int(match_cal.group(1)) if match_cal else 2000
        if cal_objetivo > 4000 or cal_objetivo < 1200: cal_objetivo = 2000
        target_por_plato = cal_objetivo / 3 

        kw_baja_cal = ["baja", "bajo", "caloría", "caloria", "lite", "light", "ligera", "definicion", "cut", "pocas", "dieta"]
        if any(k in msg_low for k in kw_baja_cal):
            preferencias_ia.append("PRIORIDAD MÁXIMA - MODO DIETA: El usuario pidió explícitamente BAJAR CALORÍAS. CADA plato debe tener MÁXIMO 400 kcal. PROHIBIDO SUGERIR > 500 kcal.")
        else:
            preferencias_ia.append(f"MODO PERFIL ACTIVO (Objetivo: {usuario_goal.upper()}): Target aprox por comida: {target_por_plato:.0f} kcal.")

        restricciones_activas = []
        if "CONDICIONES MÉDICAS:" in contexto:
            match_condiciones = re.search(r'CONDICIONES MÉDICAS:\s*([^\n]+)', contexto)
            if match_condiciones and match_condiciones.group(1).lower() != "ninguna":
                cond = match_condiciones.group(1)
                restricciones_activas.append(f"⚠️ REGLA MÉDICA: El usuario tiene {cond}. Ajusta intensidad y nutrición.")

        bloque_restricciones = "\n".join([f"- REGLA DE ORO ACTUAL: {r}" for r in restricciones_activas + preferencias_ia])
        if bloque_restricciones:
            system_content += f"\n\n### ⚠️ REGLAS CRÍTICAS BASADAS EN PERFIL DEL USUARIO:\n{bloque_restricciones}"

        system_content = system_content.replace("**", "")
        mensajes_ia = [{"role": "system", "content": system_content}]
        if historial:
            mensajes_ia.extend(historial[-2:])
        mensajes_ia.append({"role": "user", "content": mensaje_usuario})

        try:
            intentos = 0
            while intentos < 2:
                response = await self.groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=mensajes_ia,
                    max_tokens=max_tokens_ia,  # v44.0: Dinámico según tipo de solicitud
                    temperature=0.7
                )
                respuesta_ia = response.choices[0].message.content.strip()
                
                # 🛡️ PROTECCIÓN CONTRA RECORTES (Si la IA se queda a medias)
                if "[CALOFIT_STATS]" in respuesta_ia and "[/CALOFIT_STATS]" not in respuesta_ia:
                    respuesta_ia += " [/CALOFIT_STATS]"
                if "[CALOFIT_HEADER]" in respuesta_ia and "[/CALOFIT_HEADER]" not in respuesta_ia:
                    respuesta_ia += " [/CALOFIT_HEADER]"
                
                # --- v32.0: EL MARTILLO DE ETIQUETAS DEFINITIVO ---
                # 1. Limpiar basura de Markdown y alucinaciones de formato
                respuesta_ia = re.sub(r'###\s*\[', '[', respuesta_ia)
                respuesta_ia = re.sub(r'\*\*\s*\[', '[', respuesta_ia)
                
                # 2. Corregir etiquetas huérfanas o creativas sin cierre
                etiquetas_oficiales = "CALOFIT_INTENT|CALOFIT_HEADER|CALOFIT_STATS|CALOFIT_LIST|CALOFIT_ACTION|CALOFIT_FOOTER"
                respuesta_ia = re.sub(fr'^\[(?!(?:{etiquetas_oficiales}))[A-Z0-9_ ]+\]$', '[CALOFIT_HEADER] \g<0> [/CALOFIT_HEADER]', respuesta_ia, flags=re.MULTILINE)
                
                # 2. Corregir etiquetas creativas con sufijo (ej: [NOM_STATS] -> [CALOFIT_STATS])
                for tag in ["HEADER", "STATS", "LIST", "ACTION", "FOOTER"]:
                    respuesta_ia = re.sub(fr'\[[A-Z0-9_]+_{tag}\]', f'[CALOFIT_{tag}]', respuesta_ia)
                    respuesta_ia = re.sub(fr'\[/[A-Z0-9_]+_{tag}\]', f'[/CALOFIT_{tag}]', respuesta_ia)

                # 3. Normalizar estadísticas multilínea (IA suele poner saltos)
                respuesta_ia = re.sub(r'\[CALOFIT_STATS\]\s*(.*?)\s*\[/CALOFIT_STATS\]', r'[CALOFIT_STATS] \1 [/CALOFIT_STATS]', respuesta_ia, flags=re.DOTALL)
                
                # 4. Corregir alucinaciones tipo [Contenido](TAG)
                patrones_error = [
                    (r'\[(.*?)\]\(CALOFIT_(HEADER|STATS|LIST|ACTION|FOOTER)\)', r'[CALOFIT_\2] \1 [/CALOFIT_\2]'),
                    (r'\[CALOFIT_(HEADER|STATS|LIST|ACTION|FOOTER):\s*(.*?)\]', r'[CALOFIT_\1] \2 [/CALOFIT_\1]'),
                ]
                for p_err, p_fix in patrones_error:
                    respuesta_ia = re.sub(p_err, p_fix, respuesta_ia, flags=re.IGNORECASE)
                
                # v20.0: Limpieza de Markdown
                respuesta_ia = respuesta_ia.replace("***", "").replace("**", "")
                # Limpiar espacios antes de dos puntos en etiquetas de fallback
                respuesta_ia = re.sub(r'\s+:', ':', respuesta_ia)
                print(f"🤖 [IA RESPONSE]: {respuesta_ia[:200]}...")
                
                # --- NIVEL 2: AUDITORÍA DE CALIDAD (ML-CRITIC) ---
                respuesta_auditada = self.auditar_calidad_respuesta(respuesta_ia, mensaje_usuario)
                
                # --- NIVEL 3: VALIDACIÓN MATEMÁTICA INTELIGENTE (v20.0) ---
                es_rutina = False
                # --- NIVEL 3: ORQUESTADOR MULTI-SECCIÓN (v33.0) ---
                patron_split = r'(\[CALOFIT_INTENT:.*?\]|\[CALOFIT_HEADER\])'
                bloques = re.split(patron_split, respuesta_auditada)
                respuesta_procesada = ""
                
                peso_usuario = 70.0
                match_peso = re.search(r'Perfil:\s*(\d+(?:\.\d+)?)\s*kg', contexto)
                if match_peso:
                    try: peso_usuario = float(match_peso.group(1))
                    except: pass

                i = 0
                while i < len(bloques):
                    fragmento = bloques[i].strip()
                    if not fragmento: 
                        i += 1
                        continue
                    
                    if fragmento.startswith("[CALOFIT_INTENT:") or fragmento == "[CALOFIT_HEADER]":
                        etiqueta = fragmento
                        # El cuerpo es el siguiente fragmento
                        cuerpo = bloques[i+1] if (i+1) < len(bloques) else ""
                        
                        # Si el cuerpo empieza con la misma etiqueta (alucinación IA), limpiarlo
                        if cuerpo.strip().startswith(etiqueta):
                             cuerpo = re.sub(re.escape(etiqueta), "", cuerpo, 1, flags=re.IGNORECASE)

                        if etiqueta == "[CALOFIT_HEADER]":
                           cuerpo = "[CALOFIT_HEADER]" + cuerpo
                           int_deducida = "ITEM_WORKOUT" if "ejercicio" in cuerpo.lower() or "repeticiones" in cuerpo.lower() else "ITEM_RECIPE"
                           intencion = f"[CALOFIT_INTENT: {int_deducida}]"
                        else:
                           intencion = etiqueta

                        if any(k in intencion for k in ["WORKOUT", "EJERCICIO"]):
                            # v44.1: Ejecutar en thread pool (es síncrona/bloqueante)
                            try:
                                loop = asyncio.get_event_loop()
                                cuerpo_validado = await asyncio.wait_for(
                                    loop.run_in_executor(None, self.validar_y_corregir_ejercicio, cuerpo, peso_usuario),
                                    timeout=15.0
                                )
                            except asyncio.TimeoutError:
                                print("⚠️ Timeout en validar_ejercicio, usando respuesta IA directa")
                                cuerpo_validado = cuerpo
                        elif any(k in intencion for k in ["RECIPE", "DIET", "COMIDA"]):
                            # v44.1: Solo validar nutrición si es receta real (hay ingredientes)
                            # Ejecutar en thread pool para no bloquear el event loop
                            if "[CALOFIT_LIST]" in cuerpo or "[CALOFIT_HEADER]" in cuerpo:
                                try:
                                    loop = asyncio.get_event_loop()
                                    cuerpo_validado = await asyncio.wait_for(
                                        loop.run_in_executor(
                                            None,
                                            functools.partial(self.validar_y_corregir_nutricion, cuerpo, mensaje_usuario)
                                        ),
                                        timeout=15.0
                                    )
                                except asyncio.TimeoutError:
                                    print("⚠️ Timeout en validar_nutricion, usando respuesta IA directa")
                                    cuerpo_validado = cuerpo
                            else:
                                cuerpo_validado = cuerpo
                        else:
                            cuerpo_validado = cuerpo
                            
                        # Inyectar Intención si no está (v33.0)
                        c_final = cuerpo_validado.strip()
                        if not c_final.startswith("[CALOFIT_INTENT:"):
                             respuesta_procesada += "\n\n" + intencion + "\n" + c_final
                        else:
                             respuesta_procesada += "\n\n" + c_final
                        i += 2
                    else:
                        respuesta_procesada += fragmento
                        i += 1
                
                # --- v34.0: POST-PROCESADOR DE LIMPIEZA TOTAL ---
                # 1. Limpiar dobles etiquetas de intención (AI suele repetirlas)
                regex_intent = r'(\[CALOFIT_INTENT:.*?\])\s*(\[CALOFIT_INTENT:.*?\])'
                respuesta_procesada = re.sub(regex_intent, r'\1', respuesta_procesada)
                
                # 2. Limpiar corchetes accidentales en Header
                respuesta_procesada = re.sub(r'\[CALOFIT_HEADER\]\s*\[(.*?)\]\s*\[/CALOFIT_HEADER\]', r'[CALOFIT_HEADER] \1 [/CALOFIT_HEADER]', respuesta_procesada)
                
                # 3. Eliminar bloques vacíos
                respuesta_procesada = re.sub(r'\[CALOFIT_(HEADER|STATS|LIST|ACTION|FOOTER)\]\s*\[/CALOFIT_\1\]', '', respuesta_procesada)
                
                # 4. (Eliminado) Ya no borramos intents duplicados, permitimos múltiples tarjetas
                # respuesta_procesada = re.sub(r'(\[CALOFIT_INTENT:.*?\][\s\S]*?)\[CALOFIT_INTENT:.*?\]', r'\1', respuesta_procesada)
                respuesta_final = respuesta_procesada.strip()
                
                return respuesta_final
            
            # Si agota los intentos, devuelve la última respuesta generada
            return respuesta_final
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Error en chat de Groq: {error_msg}")
            if "rate_limit" in error_msg.lower():
                return "Lo siento, el servicio de IA está un poco saturado (Rate Limit). Por favor, intenta de nuevo en unos segundos."
            return f"Lo siento, hubo un error técnico al procesar tu solicitud: {error_msg}. ¿Podrías intentar de nuevo?"

    # ✅ Función de Auditoría de Calidad (Versión Mejorada)
    def auditar_calidad_respuesta(self, respuesta: str, input_usuario: str) -> str:
        """
        Nivel 2 de Robustez: Scanner de coherencia física y regional.
        """
        import re
        
        # ⚠️ DETECTOR DE ALUCINACIONES BIOMECÁNICAS (Ultra-Agresivo)
        if "dominada" in respuesta.lower():
            # Errores críticos: confundir con ejercicios de pesas externas o empuje
            errores_comunes = [
                "pies en la barra", "levanta la barra", "codos rectos", 
                "muslos paralelos", "sentar", "pies en el suelo", 
                "sin levantar los pies", "posterior de la cabeza", "detrás de la nuca",
                "baja la barra", "carga el peso", "carga la barra", "mueve la barra"
            ]
            if any(error in respuesta.lower() for error in errores_comunes):
                print("🚨 ALERTA: Física imposible detectada en Dominadas. Aplicando técnica fija...")
                # Regex potente para atrapar el bloque sin importar el formato inicial
                respuesta = re.sub(
                    r"(\d+\.\s*|\*\s*)?\*\*Dominada[^*]*\*\*:?([\s\S]+?)(?=\n\d+\.|\n\n|\n\s*(\d+\.\s*|\*\s*)?\*\*|$)", 
                    f"**Dominada con Autocarga**: 3 series de 8-12 reps. {self.CONOCIMIENTO_TECNICO.get('dominada', '')} El cuerpo sube a la barra fija. No muevas la barra hacia ti.",
                    respuesta, flags=re.IGNORECASE
                )

        if "remo" in respuesta.lower() or "romo" in respuesta.lower():
            if any(x in respuesta.lower() for x in ["codos rectos", "pies en la barra", "sentadilla", "posterior de la cabeza"]):
                print("🚨 ALERTA: Alucinación en Remo detectada.")
                respuesta = re.sub(
                    r"(\d+\.\s*|\*\s*)?\*\*Re?mo[^*]*\*\*:?([\s\S]+?)(?=\n\d+\.|\n\n|\n\s*(\d+\.\s*|\*\s*)?\*\*|$)",
                    f"**Remo con Barra/Mancuerna**: 3 series de 8-12 reps. {self.CONOCIMIENTO_TECNICO.get('remo', '')} Tracción fluida al abdomen.",
                    respuesta, flags=re.IGNORECASE
                )
        
        # 🌴 REFUERZO REGIONAL (Corrección de formato)
        if "selva" in input_usuario.lower() and not any(i in respuesta.lower() for i in ["paiche", "cecina", "cocona", "bijao", "yuca"]):
             if "**Ingredientes clave:**" in respuesta or "Ingredientes clave:" in respuesta:
                 respuesta = respuesta.replace("Ingredientes clave:", "Ingredientes clave:\n* **Sabor Amazónico**: Ají de Cocona o Patacones")

        return respuesta
    
    # ✅ Función Matemática (Revertido nombre original para evitar crash)
    def validar_y_corregir_nutricion(self, respuesta_ia: str, mensaje_usuario: str = None) -> str:
        """
        v68.2: MOTOR DE VALIDACIÓN CON RE-ETIQUETADO AUTOMÁTICO.
        Sincroniza el HEADER con la suma exacta y repara formatos rotos.
        """
        import re

        # --- v68.2: SANITIZADOR DE ETIQUETAS (REPARAR FORMATO ROTO) ---
        # Si la IA respondió sin etiquetas pero parece una receta (tiene lista con viñetas)
        if "[CALOFIT_INTENT" not in respuesta_ia and ("•" in respuesta_ia or "*" in respuesta_ia or "Ingredientes:" in respuesta_ia):
            # Intentar envolver el contenido en un formato válido
            match_header = re.search(r'^(.*?)(?:\n|\r|$)', respuesta_ia)
            header_sug = match_header.group(1).strip() if match_header else "Sugerencia CaloFit"
            
            # Extraer ingredientes (buscando líneas con * o •)
            ing_lines = re.findall(r'[•\*]\s*(.*?)(?:\n|\r|$)', respuesta_ia)
            prep_match = re.search(r'(?:Preparación|Pasos|Instrucciones):?\s*(.*)', respuesta_ia, re.DOTALL | re.IGNORECASE)
            prep_sug = prep_match.group(1).strip() if prep_match else "Sigue las instrucciones generales."
            
            if ing_lines:
                lista_fake = "\n".join([f"* {line}" for line in ing_lines])
                respuesta_ia = f"""[CALOFIT_INTENT: ITEM_RECIPE]
[CALOFIT_HEADER] {header_sug[:40]} [/CALOFIT_HEADER]
[CALOFIT_LIST]
{lista_fake}
[/CALOFIT_LIST]
[CALOFIT_ACTION]
{prep_sug}
[/CALOFIT_ACTION]
[/CALOFIT_INTENT: ITEM_RECIPE]"""

        # Regex para capturar tarjetas de receta completas
        patron_card = r'(\[CALOFIT_INTENT:\s*ITEM_RECIPE\].*?\[/CALOFIT_INTENT:\s*ITEM_RECIPE\])'
        cards = re.findall(patron_card, respuesta_ia, re.DOTALL)
        
        if cards:
            for card in cards:
                nueva_card = self._procesar_card_nutricional_v67(card, mensaje_usuario)
                respuesta_ia = respuesta_ia.replace(card, nueva_card, 1)
            
            # Limpiar el Header de basura numérica global si quedó algo
            respuesta_ia = re.sub(r'(\[CALOFIT_HEADER\].*?) (?:P|C|G|Cal):.*?(?=\[/CALOFIT_HEADER\])', r'\1', respuesta_ia, flags=re.IGNORECASE)
            return respuesta_ia
        else:
            # Fallback para respuestas sin INTENT tags pero con LISTA
            if "[CALOFIT_LIST]" in respuesta_ia:
                # v70.0: DETECTOR DE CONSULTA DIRECTA
                # Si el usuario hace una pregunta informativa y no un registro explícito,
                # preferimos responder con texto directo en lugar de Card.
                msg_low = mensaje_usuario.lower()
                es_consulta_directa = any(k in msg_low for k in ["cuántas", "cuantas", "qué tiene", "que tiene", "cuánto de", "cuanto de", "macros de"])
                if es_consulta_directa and "registra" not in msg_low and "anota" not in msg_low:
                    print("🔍 v70.0: Consulta directa detectada, convirtiendo Card a texto...")
                    # Extraer info de la card y convertirla a texto amigable
                    respuesta_ia = self._convertir_card_a_texto_directo(respuesta_ia, mensaje_usuario)
                    return respuesta_ia
                
                return self._procesar_card_nutricional_v67(respuesta_ia, mensaje_usuario)
            return respuesta_ia

    def _calcular_exacto_de_alimento(self, nombre_raw: str, cant_raw: str, unidad_raw: str):
        """
        MOTOR ÚNICO v70.0: Centraliza la lógica de búsqueda y cálculo nutricional.
        Garantiza coherencia total entre Chat y Registro.
        """
        n_limpio = nombre_raw.lower()
        for ruido in self.PALABRAS_RUIDO:
            n_limpio = n_limpio.replace(ruido, "").strip()
        
        # Traducción Regional
        n_query = n_limpio
        for original, trad in self.TRADUCTORES_REGIONALES.items():
            if original in n_limpio:
                n_query = trad
                break
        
        info = self.nutricion_service.obtener_info_alimento_fast(n_query)
        if not info and n_query != n_limpio:
            info = self.nutricion_service.obtener_info_alimento_fast(n_limpio)
        if not info:
            info = self.nutricion_service.obtener_info_alimento_fast(nombre_raw.lower().strip())
        
        cantidad = self._parse_cantidad(cant_raw)
        if cantidad == 0.0:
            # v70.0: Mejora de estimación de piezas si no hay cantidad
            if any(x in n_limpio for x in ["manzana", "pera", "plátano", "huevo", "pan", "unidad", "pieza"]):
                cantidad = 1.0
                if not unidad_raw: unidad_raw = "unidad"
            else:
                cantidad = 15.0 if any(pg in n_limpio for pg in ["aceite", "mantequilla", "aliño"]) else 100.0
        
        unidad = (unidad_raw or ("g" if not cant_raw else "")).strip().lower()
        
        # Clasificación
        es_sal_o_especia = any(x == n_limpio or (x in n_limpio and len(n_limpio) < 6) for x in ["sal", "pimienta", "comino", "laurel", "clavo", "orégano", "especia", "canela"])
        es_saborizante = any(x == n_limpio for x in ["agua", "hielo"])
        
        cant_visual = cantidad
        cals_item, p_item, c_item, g_item = 0.0, 0.0, 0.0, 0.0
        
        if info and not (es_sal_o_especia or es_saborizante):
            equiv_g = cant_visual
            if not unidad_raw or unidad == "":
                if any(x in n_limpio for x in ["manzana", "pera", "plátano", "huevo"]): equiv_g = cant_visual * 150
                elif "diente" in n_limpio: equiv_g = cant_visual * 5
                
            if unidad in ['taza', 'tazas']: equiv_g = cant_visual * 200
            elif unidad in ['cucharada', 'cda']: equiv_g = cant_visual * 15
            elif unidad in ['cucharadita', 'cdta']: equiv_g = cant_visual * 5
            elif unidad in ['unidad', 'unidades', 'pieza', 'piezas']: equiv_g = cant_visual * 150
            
            f = equiv_g / 100.0
            cals_item = (info.get("calorias") or 0) * f
            p_item = (info.get("proteinas") or 0) * f
            c_item = (info.get("carbohidratos") or 0) * f
            g_item = (info.get("grasas") or 0) * f
            
            if "frito" in nombre_raw.lower() and "frito" not in info.get("alimento",""):
                cals_item += 72.0
                g_item += 8.0
        else:
            if not (es_sal_o_especia or es_saborizante):
                equiv_g = cant_visual
                if not unidad_raw or unidad == "":
                    if any(x in n_limpio for x in ["manzana", "pera", "plátano", "huevo"]): equiv_g = cant_visual * 150
                
                if unidad in ['taza', 'tazas']: equiv_g = cant_visual * 200
                elif unidad in ['cucharada', 'cda']: equiv_g = cant_visual * 15
                elif unidad in ['cucharadita', 'cdta']: equiv_g = cant_visual * 5
                elif unidad in ['unidad', 'unidades', 'pieza', 'piezas']: equiv_g = cant_visual * 150
                
                f_est = equiv_g / 100.0
                cals_item = 45.0 * f_est
                p_item = 1.2 * f_est
                c_item = 9.0 * f_est
                g_item = 0.4 * f_est
        
        if not (es_sal_o_especia or es_saborizante) and cals_item < 1.0:
            cals_item = 1.0
            
        return {
            "cals": cals_item, "prot": p_item, "carb": c_item, "gras": g_item,
            "cant_visual": cant_visual, "unidad": unidad, "es_0": (es_sal_o_especia or es_saborizante),
            "encontrado": bool(info)
        }

    def _convertir_card_a_texto_directo(self, card_text: str, mensaje_usuario: str) -> str:
        """v70.1: Convierte la Card interactiva en una respuesta de texto amigable y coherente."""
        import re
        patron_stats = r'\[CALOFIT_STATS\](.*?)\[/CALOFIT_STATS\]'
        match_stats = re.search(patron_stats, card_text)
        stats = match_stats.group(1).replace("Cal:", "🔥 Calorías:").replace("P:", "🥚 P:").replace("C:", "🍞 C:").replace("G:", "🥑 G:").strip() if match_stats else ""
        
        # Limpiar etiquetas
        texto = re.sub(r'\[CALOFIT_.*?\]', '', card_text)
        texto = re.sub(r'\[/CALOFIT_.*?\]', '', texto)
        
        # Extraer nombre del header para el título
        match_h = re.search(r'\[CALOFIT_HEADER\](.*?)\[/CALOFIT_HEADER\]', card_text)
        titulo = match_h.group(1).strip() if match_h else "Información Nutricional"
        
        # Extraer ingredientes de la lista procesada (ya corregidos por el motor)
        match_l = re.search(r'\[CALOFIT_LIST\](.*?)\[/CALOFIT_LIST\]', card_text, re.DOTALL)
        lista = match_l.group(1).strip() if match_l else ""
        
        res = f"🍏 **{titulo}**\n\n"
        if lista:
            res += f"{lista}\n\n"
        if stats:
            res += f"📊 **Macros calculados:** {stats}\n"
        
        res += "\n💡 *Nota: Estos valores coinciden exactamente con lo que se registrará si confirmas la comida.*"
        return res

    def _procesar_card_nutricional_v67(self, card_text: str, mensaje_usuario: str) -> str:
        """Helper v67.7 para procesar una tarjeta individual y cuadrar sumas."""
        import re
        
        # 1. Extraer ingredientes del bloque
        patron_bloque_interno = r'\[CALOFIT_LIST\](.*?)\[/CALOFIT_LIST\]'
        match_lista = re.search(patron_bloque_interno, card_text, re.DOTALL)
        if not match_lista: return card_text
        
        bloque_original = match_lista.group(1)
        # Limpieza de pasos numerados en la lista
        bloque_limpio = re.sub(r'^\d+\.\s+.*$', '', bloque_original, flags=re.MULTILINE)
        
        matches = self.PATRON_INGREDIENTE.findall(bloque_limpio)
        if not matches: return card_text
        
        ingredientes_calculados = []
        ingredientes_no_encontrados = []
        
        for cant_raw, unidad_raw, nombre_raw in matches:
            res = self._calcular_exacto_de_alimento(nombre_raw, cant_raw, unidad_raw)
            if not res["encontrado"] and not res["es_0"]:
                ingredientes_no_encontrados.append(nombre_raw.lower())

            ingredientes_calculados.append({
                "nombre_raw": nombre_raw,
                "cant_visual": res["cant_visual"],
                "unidad": res["unidad"],
                "cals": res["cals"],
                "prot": res["prot"],
                "carb": res["carb"],
                "gras": res["gras"],
                "es_0": res["es_0"],
                "n_low": nombre_raw.lower()
            })

        # Escalar
        cals_original = sum(it["cals"] for it in ingredientes_calculados)
        h_match = re.search(r'\[CALOFIT_HEADER\](.*?)\[/CALOFIT_HEADER\]', card_text, re.IGNORECASE)
        h_text = h_match.group(1).lower() if h_match else ""
        msg_low = mensaje_usuario.lower() if mensaje_usuario else ""
        es_ligero = any(k in msg_low or k in h_text for k in ["ligero", "baja", "light", "diet", "poco"])
        
        limite = 750.0
        if "desayuno" in h_text or "desayuno" in msg_low: limite = 300.0 if es_ligero else 500.0
        elif "snack" in h_text or "snack" in msg_low: limite = 150.0 if es_ligero else 250.0
        elif "almuerzo" in h_text or "almuerzo" in msg_low: limite = 550.0 if es_ligero else 950.0
        elif "cena" in h_text or "cena" in msg_low: limite = 400.0 if es_ligero else 700.0
        elif es_ligero: limite = 450.0

        factor_scale = 1.0
        tag_ajuste = ""
        if cals_original > limite:
            factor_scale = limite / max(cals_original, 1.0)
            tag_ajuste = " (Ajustado)"

        # Reconstrucción de Líneas y Suma Sincronizada
        nuevas_lineas = []
        suma_cals, suma_p, suma_c, suma_g = 0, 0.0, 0.0, 0.0
        
        for it in ingredientes_calculados:
            c_scaled = it["cals"] * factor_scale
            c_round = int(round(c_scaled))
            
            # v67.7: USAR EL ROUND PARA EL HEADER
            suma_cals += c_round
            suma_p += it["prot"] * factor_scale
            suma_c += it["carb"] * factor_scale
            suma_g += it["gras"] * factor_scale
            
            # Label Visual con Normalización (Simplificado v66.7 - v67.7)
            # v67.7: Garantizar que CADA ingrediente tenga su label (X kcal)
            # v67.9: ELIMINACIÓN DE 'TAZA' (Filtro de Salida Forzoso)
            u_final = it["unidad"]
            c_final = it["cant_visual"]
            n_low = it["n_low"]
            
            if u_final in ['taza', 'tazas', 'tazita', 'tacita']:
                # Decidir si es líquido o sólido para convertir a ml o g
                es_liq = any(lk in n_low for lk in ["leche", "agua", "caldo", "jugo", "sopa", "infusión", "café", "té", "bebida", "yogur", "salsa", "puré", "pure"])
                if es_liq:
                    u_final = "ml"
                    c_final = c_final * 200
                else:
                    u_final = "g"
                    # Densidad variable: Hojas vs Granos
                    es_hoja = any(hk in n_low for hk in ["espinaca", "lechuga", "albahaca", "col", "acelga"])
                    factor_est = 35 if es_hoja else 180
                    c_final = c_final * factor_est

            linea_limpia = it["nombre_raw"].split('(')[0].strip()
            label_cals = f"({c_round} kcal)" if not it["es_0"] else "(0 kcal)"
            
            # Formatear números (ej: 150.0 -> 150)
            c_str = f"{int(c_final)}" if c_final == int(c_final) else f"{round(c_final, 1)}".replace('.', ',')
            nuevas_lineas.append(f"* {c_str} {u_final} de {linea_limpia} {label_cals}")

        res_stats = f"P: {suma_p:.1f}g | C: {suma_c:.1f}g | G: {suma_g:.1f}g | Cal: {suma_cals}kcal{tag_ajuste}"
        if "frito" in card_text.lower(): res_stats += " (Aceite incluido) 🍳"
        macros_tag = f"[CALOFIT_STATS] {res_stats} [/CALOFIT_STATS]"
        
        updated = re.sub(r'\[CALOFIT_STATS\].*?\[/CALOFIT_STATS\]', macros_tag, card_text, flags=re.DOTALL)
        if macros_tag not in updated: 
            updated = updated.replace("[/CALOFIT_HEADER]", f"[/CALOFIT_HEADER]\n{macros_tag}", 1)
        
        lista_tag = "[CALOFIT_LIST]\n" + "\n".join(nuevas_lineas) + "\n[/CALOFIT_LIST]"
        updated = re.sub(r'\[CALOFIT_LIST\].*?\[/CALOFIT_LIST\]', lista_tag, updated, flags=re.DOTALL)
        
        if ingredientes_no_encontrados:
            msg_i = f"\n⚠️ Info: {', '.join(list(set(ingredientes_no_encontrados))[:2])} estimado."
            if "[/CALOFIT_FOOTER]" in updated: updated = updated.replace("[/CALOFIT_FOOTER]", msg_i + "\n[/CALOFIT_FOOTER]")
            else: updated += msg_i
            
        return updated

    def validar_y_corregir_ejercicio(self, respuesta_ia: str, peso_usuario: float = 70.0) -> str:
        """
        Calcula las calorías quemadas reales usando METs de ejercicios.json.
        """
        from app.services.ejercicios_service import ejercicios_service
        import re

        # 1. Duración ESTIMADA
        minutos_totales = 30.0
        match_duracion = re.search(r'(\d+)\s*(?:min|minutos)', respuesta_ia.lower())
        if match_duracion:
            try: minutos_totales = float(match_duracion.group(1))
            except: pass

        # 2. Escáner de Ejercicios
        cals_quemadas = 0.0
        ejercicios_detectados = []
        
        # Primero buscar dentro de [CALOFIT_LIST]
        patron_bloque = r'\[CALOFIT_LIST\](.*?)\[/CALOFIT_LIST\]'
        bloque_ej = re.search(patron_bloque, respuesta_ia, re.DOTALL)
        texto_busqueda = bloque_ej.group(1) if bloque_ej else respuesta_ia

        lineas = texto_busqueda.split('\n')
        for linea in lineas:
            l = linea.strip().lower()
            if l.startswith('-') or l.startswith('•') or re.match(r'^\d+\.', l):
                nombre_ej = re.sub(r'^[-\*•\d\.\s]+', '', l).split('(')[0].split(':')[0].strip()
                if len(nombre_ej) < 3: continue
                
                info_ej = ejercicios_service.obtener_info_ejercicio(nombre_ej)
                if info_ej and info_ej.get("met"):
                    ejercicios_detectados.append(info_ej)

        if ejercicios_detectados:
            for ej in ejercicios_detectados:
                met = float(ej["met"])
                # Dividir tiempo entre ejercicios encontrados
                min_por_ej = minutos_totales / len(ejercicios_detectados)
                cals_quemadas += (met * 3.5 * peso_usuario / 200.0) * min_por_ej
            
            self.ultimas_calorias_calculadas = cals_quemadas
            
            # Inyección de Stats con formato de EJERCICIO (sin macros falsos)
            macros_inyectados = f"[CALOFIT_STATS] Quemado: {cals_quemadas:.0f}kcal aprox [/CALOFIT_STATS]"
            regex_stats = r'\[CALOFIT_STATS\].*?\[/CALOFIT_STATS\]'
            
            if re.search(regex_stats, respuesta_ia, re.DOTALL):
                respuesta_ia = re.sub(regex_stats, macros_inyectados, respuesta_ia, flags=re.DOTALL)
            else:
                if "[/CALOFIT_HEADER]" in respuesta_ia:
                    respuesta_ia = respuesta_ia.replace("[/CALOFIT_HEADER]", "[/CALOFIT_HEADER]\n" + macros_inyectados)
                else:
                    respuesta_ia = macros_inyectados + "\n" + respuesta_ia
        
        return respuesta_ia

    def generar_plan_inicial_automatico(self, cliente_data: dict):
        """
        Genera un plan nutricional inicial refinado con lógica de 5 estados y g/kg.
        """
        print(f"🤖 Generando plan inicial refinado para: {cliente_data.get('email')}")
        
        # 1. Mapear datos base
        genero_map = {"M": 1, "F": 2}
        genero = genero_map.get(cliente_data.get("genero", "M"), 1)
        # Manejar edad si viene como objeto datetime o int
        nacimiento = cliente_data.get("fecha_nacimiento")
        if hasattr(nacimiento, 'year'):
            edad = datetime.now().year - nacimiento.year
        else:
            edad = cliente_data.get("edad", 25)

        peso = cliente_data.get("peso", 70.0)
        talla = cliente_data.get("talla", 170.0)
        
        # 2. Obtener objetivo granular
        objetivo_raw = cliente_data.get("objetivo", "Mantener peso")
        objetivo_map = {
            "Perder peso (Agresivo)": "perder_agresivo",
            "Perder peso (Definición)": "perder_definicion",
            "Mantener peso": "mantener",
            "Ganar masa (Limpio)": "ganar_lean_bulk",
            "Ganar masa (Volumen)": "ganar_bulk",
            # Fallbacks
            "Perder peso": "perder_agresivo",
            "Ganar masa": "ganar_bulk"
        }
        objetivo_key = objetivo_map.get(objetivo_raw, "mantener")
        
        # 3. Calcular calorías con Gradient Boosting
        nivel_actividad_map = {
            "Sedentario": 1.20, "Ligero": 1.375, "Moderado": 1.55, 
            "Activo": 1.725, "Muy activo": 1.90
        }
        nivel = nivel_actividad_map.get(cliente_data.get("nivel_actividad", "Sedentario"), 1.20)
        
        calorias_diarias = self.calcular_requerimiento(genero, edad, peso, talla, nivel, objetivo_key)
        
        # 4. Usar función centralizada para calcular macros
        condiciones_medicas = cliente_data.get("condiciones_medicas", "")
        macros_data = self.calcular_macros_optimizados(peso, objetivo_key, calorias_diarias, condiciones_medicas)
        
        proteinas_g = macros_data['proteinas_g']
        carbohidratos_g = macros_data['carbohidratos_g']
        grasas_g = macros_data['grasas_g']
        alerta_medica = macros_data['alerta_medica']
        
        # 5. Sistema de Validación Médica Mejorado
        validacion_requerida = False
        es_condicion_critica = False
        estado_plan = "provisional_ia"
        
        # Detectar condiciones críticas que requieren validación obligatoria
        for condicion in CONDICIONES_CRITICAS:
            if condicion in condiciones_medicas.lower():
                es_condicion_critica = True
                validacion_requerida = True
                alerta_medica += f" ⚠️ IMPORTANTE: Detectada '{condicion}'. Este plan es PROVISIONAL y requiere aprobación del nutricionista antes de su uso completo."
                estado_plan = "en_revision"
                break
        
        # Detectar otras condiciones que ameritan revisión
        if any(c in condiciones_medicas.lower() for c in ["lesion", "dolor", "hernia"]):
            validacion_requerida = True
            alerta_medica += " 🏥 REVISIÓN MÉDICA REQUERIDA antes de iniciar rutina fuerte."
        
        # Si hay condición crítica, aplicar plan ultra-conservador
        if es_condicion_critica:
            print(f"⚠️ Condición crítica detectada. Aplicando plan conservador.")
            # Forzar nivel sedentario y mantenimiento
            calorias_diarias = self._calcular_tmb_harris_benedict(genero, edad, peso, talla) * 1.2
            # Recalcular macros con las calorías conservadoras
            macros_data = self.calcular_macros_optimizados(peso, "mantener", calorias_diarias, condiciones_medicas)
            proteinas_g = macros_data['proteinas_g']
            carbohidratos_g = macros_data['carbohidratos_g']
            grasas_g = macros_data['grasas_g']

        macros = {"P": proteinas_g, "C": carbohidratos_g, "G": grasas_g}
        
        # 6. Generar Plan de 7 días con metadata completa
        dias_plan = []
        mensaje_estado = ESTADOS_PLAN.get(estado_plan, "Plan en proceso")
        
        for dia in range(1, 8):
            # Nota para cada día según el estado
            if es_condicion_critica:
                nota_dia = f"🤖 Plan provisional conservador. {alerta_medica}"
            elif alerta_medica:
                nota_dia = f"🤖 IA: {alerta_medica}"
            else:
                nota_dia = f"🤖 Plan {objetivo_key.replace('_', ' ')} calculado exitosamente."
            
            dias_plan.append({
                "dia_numero": dia,
                "calorias_dia": round(calorias_diarias, 2),
                "proteinas_g": proteinas_g,
                "carbohidratos_g": carbohidratos_g,
                "grasas_g": grasas_g,
                "sugerencia_entrenamiento_ia": self.generar_sugerencia_entrenamiento(objetivo_key.split('_')[0], dia),
                "nota_asistente_ia": nota_dia,
                "validado_nutri": False,
                "estado": estado_plan,
                "requiere_validacion": validacion_requerida
            })
        
        # 7. Mensaje personalizado para el cliente
        if es_condicion_critica:
            mensaje_cliente = "🏥 Hemos detectado una condición médica importante. Este plan es ultra-conservador y PROVISIONAL. Tu nutricionista debe revisarlo antes de que lo sigas completamente. Mientras tanto, puedes usarlo como guía general."
        elif validacion_requerida:
            mensaje_cliente = "🤖 Este es un plan provisional diseñado para que empieces de inmediato. Tu nutricionista lo revisará y ajustará según tus necesidades específicas."
        else:
            mensaje_cliente = "🤖 Este plan fue generado automáticamente basándose en tus datos. Tu nutricionista lo revisará pronto para optimizarlo aún más."
        
        return {
            "calorias_diarias": round(calorias_diarias, 2),
            "macros": macros,
            "dias": dias_plan,
            "estado_plan": estado_plan,
            "requiere_validacion": validacion_requerida,
            "es_condicion_critica": es_condicion_critica,
            "alerta_seguridad": alerta_medica,
            "generado_automaticamente": True,
            "fecha_generacion": datetime.now().isoformat(),
            "valido_hasta_validacion": True,
            "mensaje_cliente": mensaje_cliente,
            "descripcion_estado": mensaje_estado
        }

    def _parse_cantidad(self, c_raw: str) -> float:
        """v67.5: Convierte de forma segura cantidades tipo '1/2', '0.5' o '1,5' a float."""
        if not c_raw:
            return 1.0
        try:
            c_clean = str(c_raw).replace(',', '.')
            if '/' in c_clean:
                num, den = c_clean.split('/', 1)
                return float(num) / float(den)
            return float(c_clean)
        except:
            return 1.0 # Fallback seguro ante basura o formato no reconocido


ia_engine = IAService()
