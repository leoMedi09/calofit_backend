import joblib
import pandas as pd
import os
from groq import Groq
from sklearn.metrics.pairwise import cosine_similarity
from app.core.config import settings
from tensorflow import keras
import spacy
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import numpy as np
from datetime import datetime

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
    def __init__(self):
        print(f"🔍 Buscando modelo en: {MODEL_PATH}")
        if not os.path.exists(MODEL_PATH):
            print(f"❌ ERROR: No se encontró el archivo .pkl")
            self.model = None
            return

        try:
            self.model = joblib.load(MODEL_PATH)
            print("✅ IA Service: Modelo cargado exitosamente")
        except Exception as e:
            print(f"❌ IA Service: Error al cargar el modelo: {e}")
            self.model = None

        # Inicializar Groq
        self.groq_client = Groq(api_key=settings.GROQ_API_KEY)

        # Cargar modelo CBF (matrix, scaler)
        self.cbf_matrix = None
        self.cbf_scaler = None
        if os.path.exists(CBF_MATRIX_PATH):
            try:
                self.cbf_matrix = joblib.load(CBF_MATRIX_PATH)
                print("✅ Matrix CBF cargada")
            except Exception as e:
                print(f"❌ Error al cargar matrix CBF: {e}")
        if os.path.exists(CBF_SCALER_PATH):
            try:
                self.cbf_scaler = joblib.load(CBF_SCALER_PATH)
                print("✅ Scaler CBF cargado")
            except Exception as e:
                print(f"❌ Error al cargar scaler CBF: {e}")

        # Cargar modelos de fitness
        self.fit_matrix = None
        self.fit_scaler = None
        if os.path.exists(FIT_MATRIX_PATH):
            try:
                self.fit_matrix = joblib.load(FIT_MATRIX_PATH)
                print("✅ Matrix Fitness cargada")
            except Exception as e:
                print(f"❌ Error al cargar matrix Fitness: {e}")
        if os.path.exists(FIT_SCALER_PATH):
            try:
                self.fit_scaler = joblib.load(FIT_SCALER_PATH)
                print("✅ Scaler Fitness cargado")
            except Exception as e:
                print(f"❌ Error al cargar scaler Fitness: {e}")

        # Cargar modelo ANN para calorías quemadas
        self.ann_model = None
        if os.path.exists(ANN_MODEL_PATH):
            try:
                self.ann_model = keras.models.load_model(ANN_MODEL_PATH)
                print("✅ Modelo ANN cargado")
            except Exception as e:
                print(f"❌ Error al cargar modelo ANN: {e}")

        # Cargar modelo spaCy para NLP
        try:
            self.nlp = spacy.load('es_core_news_sm')
            print("✅ Modelo spaCy cargado")
        except Exception as e:
            print(f"❌ Error al cargar spaCy: {e}")
            self.nlp = None

        # Configurar lógica difusa para alertas
        self.setup_fuzzy_logic()

    def setup_fuzzy_logic(self):
        """
        Configura el sistema de lógica difusa para personalizar alertas según adherencia y progreso.
        """
        # Variables de entrada
        self.adherencia = ctrl.Antecedent(np.arange(0, 101, 1), 'adherencia')  # 0-100%
        self.progreso = ctrl.Antecedent(np.arange(0, 101, 1), 'progreso')     # 0-100%

        # Variable de salida
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
        
        # 1. Determinar g/kg según objetivo
        if "perder" in objetivo_key.lower():
            g_proteina_kg = 2.2  # Máxima protección muscular en déficit
            g_grasa_kg = 0.8     # Grasas base
        elif "ganar" in objetivo_key.lower():
            g_proteina_kg = 2.0  # Construcción muscular
            g_grasa_kg = 1.0     # Balance hormonal para anabolismo
        else:
            g_proteina_kg = 1.8  # Mantenimiento
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
        
        if not self.model:
            print("⚠️ Modelo ML no disponible, usando Harris-Benedict como fallback")
            basal = self._calcular_tmb_harris_benedict(genero, edad, peso, talla)
        else:
            try:
                # 1. Predicción con Gradient Boosting (Basado en NHANES)
                df = pd.DataFrame([[genero, edad, peso, talla]], 
                                  columns=['RIAGENDR', 'RIDAGEYR', 'BMXWT', 'BMXHT'])
                
                pred = self.model.predict(df)
                basal = pred.item()
                print(f"✅ TMB calculado por ML: {basal:.2f} kcal")
            except Exception as e:
                print(f"❌ Error en predicción ML: {e}, usando Harris-Benedict")
                basal = self._calcular_tmb_harris_benedict(genero, edad, peso, talla)
        
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
        
        REGLA DE ORO: 
        Cualquier sugerencia de alimentos debe ser adaptada a la biodiversidad y mercado peruano. 
        Si el sistema sugiere un ingrediente genérico, sustitúyelo por su equivalente peruano:
        - Ej: Arándanos -> Aguaymanto/Fresa nacional.
        - Ej: Salmón -> Trucha andina/Bonito/Jurel.
        - Ej: Kale/Greens -> Espinaca/Acelerga/Hojas de quinua.
        - Ej: Aceite de Canola -> Aceite de Oliva/Sacha Inchi.
        
        MENÚ PERUANO (5 COMIDAS):
        - Desayuno, Media Mañana, Almuerzo (principal), Media Tarde, Cena.
        - Usa términos locales: palta, camote, papa, choclo, menestras.
        - Indica porciones claras y el aporte calórico por comida.
        
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
                model="llama-3.1-8b-instant",
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
        pct_calorias = (consumo_actual['calorias'] / meta_calorias) * 100
        
        # 3. Construir el prompt para Groq (enfocado en una frase corta)
        prompt = f"""
        Eres un coach de salud. Usuario: {perfil_usuario['first_name']}. 
        Meta: {meta_calorias} kcal. Consumo hoy: {consumo_actual['calorias']} kcal ({pct_calorias:.1f}%).
        Condiciones médicas: {perfil_usuario['medical_conditions']}.
        
        Genera un 'insight' de UNA SOLA FRASE (máximo 15 palabras). 
        Si el % es > 90, advierte sobre el límite. Si es < 50, motiva a comer más proteína.
        Sé muy específico y usa un tono profesional pero amigable.
        """

        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.5
            )
            return response.choices[0].message.content.strip()
        except:
            return f"¡Vas por buen camino, {perfil_usuario['first_name']}! Sigue hidratándote y cumpliendo tus metas."

# Singleton
    def extraer_macros_de_texto(self, texto: str):
        """
        Usa Groq para extraer información nutricional de un texto libre.
        Ejemplo: "Hoy comí arroz con pollo y una manzana"
        """
        prompt = f"""
        Analiza el siguiente texto y extrae la información nutricional estimada: "{texto}"
        
        Debes responder ÚNICAMENTE en formato JSON plano con la siguiente estructura:
        {{
            "alimentos_detectados": ["alimento1", "alimento2"],
            "calorias": 0,
            "proteinas_g": 0,
            "carbohidratos_g": 0,
            "grasas_g": 0,
            "es_comida": true,
            "es_ejercicio": false
        }}
        
        Si el texto describe un ejercicio, pon "es_comida": false y "es_ejercicio": true, 
        y estima las calorías quemadas (en positivo).
        Si no puedes identificar nada, devuelve ceros.
        """
        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.1, # Muy bajo para mantener formato estricto
                response_format={"type": "json_object"}
            )
            import json
            return json.loads(response.choices[0].message.content.strip())
        except Exception as e:
            print(f"Error extrayendo macros con Groq: {e}")
            return None

    def identificar_intencion_salud(self, texto: str):
        """
        Detecta si el mensaje del usuario contiene alguna alerta de salud
        (lesiones, dolores, fatiga, malestar, etc.) usando Groq.
        
        Returns:
            dict con: {
                "tiene_alerta": bool,
                "tipo": str (lesion/fatiga/desanimo/malestar/otro),
                "descripcion_resumida": str,
                "severidad": str (bajo/medio/alto),
                "recomendacion_contingencia": str
            }
        """
        prompt = f"""
        Analiza el siguiente mensaje de un cliente de gimnasio y determina si reporta algún problema de salud.
        
        MENSAJE DEL CLIENTE: "{texto}"
        
        INSTRUCCIÓN: Detecta si el mensaje menciona:
        - Lesiones (dolor, golpe, torcedura, esguince, fractura)
        - Fatiga extrema (muy cansado, agotado, sin energía)
        - Malestar general (mareos, náuseas, debilidad)
        - Problemas emocionales (desmotivado, deprimido, ansioso)
        - Enfermedades (gripe, fiebre, resfriado)
        
        RESPONDE EN FORMATO JSON VÁLIDO:
        {{
            "tiene_alerta": true/false,
            "tipo": "lesion" | "fatiga" | "malestar" | "desanimo" | "enfermedad" | "otro",
            "descripcion_resumida": "Breve descripción del problema (máximo 100 caracteres)",
            "severidad": "bajo" | "medio" | "alto",
            "recomendacion_contingencia": "Sugerencia profesional (reposo, consultar médico, hidratación, etc.)"
        }}
        
        CRITERIOS DE SEVERIDAD:
        - BAJO: Molestias leves, cansancio normal
        - MEDIO: Dolor moderado, fatiga significativa que limita actividad
        - ALTO: Dolor intenso, lesión grave, mareos fuertes, síntomas de emergencia
        
        Si el mensaje NO menciona ningún problema de salud, responde:
        {{
            "tiene_alerta": false
        }}
        
        SOLO responde con JSON válido, sin texto adicional.
        """
        
        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.3  # Baja temperatura para respuestas más determinísticas
            )
            
            respuesta_texto = response.choices[0].message.content.strip()
            
            # Extraer JSON de la respuesta (a veces Groq agrega texto extra)
            import json
            import re
            
            # Buscar el JSON en la respuesta
            json_match = re.search(r'\{.*\}', respuesta_texto, re.DOTALL)
            if json_match:
                resultado = json.loads(json_match.group())
                return resultado
            else:
                print(f"No se pudo parsear JSON de Groq: {respuesta_texto}")
                return {"tiene_alerta": False}
                
        except Exception as e:
            print(f"Error en identificar_intencion_salud: {e}")
            return {"tiene_alerta": False}

    def asistir_cliente(self, contexto: str, mensaje_usuario: str, historial: list = None):
        """
        Maneja la conversación con memoria. 
        historial: lista de dicts [{"role": "user/assistant", "content": "..."}]
        """
        # 1. Preparar el Sistema de Mensajes (System Prompt)
        mensajes_ia = [
            {
                "role": "system", 
                "content": f"""Eres CaloFit IA, el asistente experto de una plataforma de nutrición peruana.
                CONTEXTO ACTUAL DEL USUARIO: {contexto}
                
                REGLAS:
                - Recomendaciones: Solo platos peruanos reales (ej: Pollo a la plancha, Lomo Saltado, Ají de Gallina, etc).
                - NO inventes nombres de platos extraños ni procesos inexistentes.
                - Sé breve (máximo 40 palabras).
                - Estructura: Mensaje corto + 1 a 3 viñetas claras.
                - Precisión: No inventes números. Usa los del CONTEXTO. """
            }
        ]

        # 2. Agregar historial previo si existe (Memoria)
        if historial:
            mensajes_ia.extend(historial[-6:]) # Enviamos los últimos 6 mensajes para no saturar tokens

        # 3. Agregar el mensaje actual del usuario
        mensajes_ia.append({"role": "user", "content": mensaje_usuario})

        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=mensajes_ia,
                max_tokens=500,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"❌ Error en chat de Groq: {e}")
            return "Lo siento, perdí el hilo de la conversación por un momento. ¿En qué nos quedamos?"

    def generar_plan_inicial_automatico(self, cliente_data: dict):
        """
        🆕 Genera un plan nutricional inicial refinado con lógica de 5 estados y g/kg.
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


ia_engine = IAService()