import re
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

        # Cargar Base de Datos de Ejercicios (Biomecánica & METs)
        self.datos_ejercicios = []
        try:
            ruta_ejercicios = os.path.join(BASE_DIR, 'data', 'ejercicios.json')
            if os.path.exists(ruta_ejercicios):
                import json
                with open(ruta_ejercicios, 'r', encoding='utf-8') as f:
                    self.datos_ejercicios = json.load(f)
                print(f"✅ Base de Ejercicios cargada: {len(self.datos_ejercicios)} items")
            else:
                print(f"⚠️ No se encontró: {ruta_ejercicios}")
        except Exception as e:
            print(f"❌ Error al cargar ejercicios.json: {e}")

        # Cargar Base de Datos de Alimentos (INS Perú & Platos Típicos)
        self.datos_nutricionales = []
        try:
            ruta_alimentos = os.path.join(BASE_DIR, 'data', 'alimentos_peru_ins.json')
            if os.path.exists(ruta_alimentos):
                import json
                with open(ruta_alimentos, 'r', encoding='utf-8') as f:
                    self.datos_nutricionales = json.load(f)
                print(f"✅ Base de Alimentos cargada: {len(self.datos_nutricionales)} items")
            else:
                print(f"⚠️ No se encontró: {ruta_alimentos}")
        except Exception as e:
            print(f"❌ Error al cargar alimentos_peru_ins.json: {e}")

        # Base de Conocimiento Técnico (Hardcoded para validación de alucinaciones)
        self.CONOCIMIENTO_TECNICO = {
            "dominada": "Evita balanceos. Sube hasta pasar la barbilla. Baja controlado.",
            "remo": "Mantén la espalda neutra. Tira con los codos hacia atrás, no con los bíceps."
        }

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
        
        # 0. Cálculo Base Harris-Benedict (Baseline de Seguridad)
        basal_hb = self._calcular_tmb_harris_benedict(genero, edad, peso, talla)
        
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
        
        REGLAS CRÍTICAS (FALLO CERO):
        1. COMIDA: Calcula calorias y macros sumarizados siempre.
        2. EJERCICIO: Si detectas ejercicio, "es_comida": false, "es_ejercicio": true.
           ⚠️ OBLIGATORIO: DEBES CALCULAR CALORÍAS QUEMADAS (aprox 10 cal/min para intenso, 5 cal/min moderado).
           Ejemplo: "Corrí 30 min" -> 30 * 10 = 300 kcal.
           NUNCA DEVUELVAS 0 si hay mención de tiempo o esfuerzo.
        3. SI NO HAY INFORMACIÓN: Devuelve todo en 0.
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

    def asistir_cliente(self, contexto: str, mensaje_usuario: str, historial: list = None, tono_aplicado: str = "") -> str:
        """
        Consulta a Groq con un contexto adaptativo.
        """
        print(f"📡 Enviando a Groq - Tono: {tono_aplicado[:20]}...")
        
        # 0. Preparar Contexto Dinámico de Ejercicios y Cultura Peruana
        texto_extra = ""
        platos_prioritarios = [] # Definir para evitar NameError
        try:
            import random
            import json

            # --- BASE DE DATOS DE EJERCICIOS ---
            if hasattr(self, 'datos_ejercicios') and self.datos_ejercicios:
                msg_low = mensaje_usuario.lower()
                es_gym = any(k in msg_low for k in ["gym", "gimnasio", "pesas", "musculo", "fuerza", "hipertrofia"])
                
                # 1. Gold Standard (Estándar mundial)
                gold_standard = [e for e in self.datos_ejercicios if e.get('origen') == 'gold_standard']
                # 2. Peruanos (Lifestyle/Pichangas)
                peruanos = [e for e in self.datos_ejercicios if e.get('origen') == 'peru_lifestyle']
                # 3. Importados (Gym)
                otros = [e for e in self.datos_ejercicios if e.get('origen') == 'dataset_importado']
                
                # Muestra estratégica: Si pide GYM, EXCLUIR peruanos (Pichanga) para evitar confusiones
                if es_gym:
                    muestra_ej = gold_standard[:15] + otros[:10]
                    texto_extra += "\n### CONTEXTO GIMNASIO ACTIVO: Sugiere máquinas, pesas o cardio indoor. PROHIBIDO: Fútbol/Pichanga."
                else:
                    muestra_ej = peruanos[:4] + gold_standard[:4]
                
                texto_extra += "\n### BASE DE DATOS DE EJERCICIOS (MUESTRA):\n"
                texto_extra += json.dumps(muestra_ej, ensure_ascii=False)
                # --- FILTRO VEGANO HARDCORE ---
                es_vegano = "vegano" in contexto.lower() or "vegetariano" in contexto.lower()
                if es_vegano:
                    prohibidos = ["pollo", "carne", "res", "pescado", "huevo", "leche", "queso", "cecina", "paiche", "trucha", "cuy", "chancho", "puerco", "jamon", "chorizo", "salame", "atun", "pachamanca", "lomo saltado"]
                    self.datos_nutricionales = [
                        p for p in self.datos_nutricionales 
                        if not any(pro in p.get('nombre', '').lower() for pro in prohibidos)
                    ]
                    print(f"🌱 Filtro Vegano Activo: {len(self.datos_nutricionales)} platos aptos restantes.")

                # 1. Platos Fuertes (Almuerzos/Cenas)
                todos_fuertes = [a for a in self.datos_nutricionales if a.get('categoria') in ['Comida Típica', 'Sopa', 'Postre']]
                
                # Mezclar prioritarios con random para rellenar
                if platos_prioritarios:
                    fuertes_region = [p for p in platos_prioritarios if p in todos_fuertes]
                    resto_fuertes = [p for p in todos_fuertes if p not in fuertes_region]
                    muestra_comida = fuertes_region[:2] + random.sample(resto_fuertes, min(2, len(resto_fuertes)))
                else:
                    muestra_comida = random.sample(todos_fuertes, min(3, len(todos_fuertes))) if todos_fuertes else []

                # 2. Desayunos Potentes (BD)
                todos_desayunos = [
                    d for d in self.datos_nutricionales 
                    if d.get('categoria') == 'Desayuno' or d.get('origen') == 'plato_compuesto'
                ]
                
                if platos_prioritarios:
                    desayunos_region = [d for d in platos_prioritarios if d in todos_desayunos]
                    resto_desayunos = [d for d in todos_desayunos if d not in desayunos_region]
                    # AUMENTAR MUESTRA: ¡Mostrar todos los regionales posibles! (antes era 3)
                    muestra_desayunos = desayunos_region[:2] + random.sample(resto_desayunos, min(2, len(resto_desayunos)))
                else:
                    muestra_desayunos = random.sample(todos_desayunos, min(3, len(todos_desayunos))) if todos_desayunos else []

                if muestra_comida or muestra_desayunos:
                    # Combinar priorizando desayunos si es de mañana
                    todo_junto = muestra_desayunos + muestra_comida
                    # FALLBACK SEGURO: Si no tiene 'id', usa 'nombre' para evitar crash
                    todo_junto_unico = list({v.get('id', v.get('nombre')): v for v in todo_junto}.values())

                    for v in todo_junto_unico:
                        if 'proteina_100g' in v: # Es un alimento/plato
                            texto_extra += f"- {v.get('nombre')} (P: {v.get('proteina_100g')}g, C: {v.get('carbohindratos_100g')}g, G: {v.get('grasas_100g')}g, Cal: {v.get('calorias_100g')}kcal por 100g)\n"
                        else: # Es un ejercicio
                            # v18.8: Filtrar deportes si es una petición de RUTINA
                            es_deporte_social = any(k in v.get('nombre', '').lower() for k in ["pichanga", "fútbol", "futbol", "vóley", "voley", "fulbito"])
                            if "rutina" in mensaje_usuario.lower() and es_deporte_social:
                                continue
                            texto_extra += f"- {v.get('nombre')} (MET: {v.get('met', 5.0)})\n"
                    texto_extra += "\n(IMPORTANTE: Prioriza Desayunos ligeros si es mañana. EVITA Platos de almuerzo pesados)."

        except Exception as e:
            print(f"Error preparando contexto cultural: {e}")

        # --- LÓGICA DE EMERGENCIA VEGANA (FUERA DEL TRY) ---
        if "vegano" in contexto.lower() or "vegetariano" in contexto.lower():
             texto_extra += "\n\n⛔ ALERTA VEGANA CRÍTICA: El usuario es VEGANO/VEGETARIANO. PROHIBIDO: Carne, Pollo, Pescado, Huevos, Leche, Queso, Miel. ¡NI UNA SOLA TRAZA! Usa: Tofu, Soya, Quinua, Menestras, Seitán."

        # (v11.5 - Prompt Dinámico de Intención)
        es_consulta_info = False
        keywords_info = ["cuantas calorias", "qué es", "que es", "beneficios", "propiedades", "engorda", "adelgaza", "información", "tengo", "puedo"]
        keywords_accion = ["receta", "preparar", "cocinar", "plato", "menú", "menu", "desayuno", "almuerzo", "cena", "rutina", "entrenamiento", "ejercicios", "plan", "dieta", "sugerencia", "opcion", "dame"]

        msg_low = mensaje_usuario.lower()
        if any(ki in msg_low for ki in keywords_info) and not any(ka in msg_low for ka in keywords_accion):
            es_consulta_info = True

        if es_consulta_info:
            system_content = f"""ERES UN ASISTENTE DE NUTRICIÓN EXPERTO.
            
            ### CONTEXTO DEL USUARIO Y STATUS DIARIO:
            {contexto}
            
            {texto_extra}
            
            TU META: Responder la duda del usuario de forma EXTREMADAMENTE BREVE, DIRECTA Y CIENTÍFICA.
            - Usa la sección 'STATUS DEL DÍA' del contexto para responder sobre sus números.
            - NO inventes recetas, platos ni rutinas.
            - NO uses etiquetas como 'Plato:', 'Rutina:', 'Ingredientes:', 'Preparación:' o 'Técnica:'.
            - Si el usuario pregunta qué lleva consumido o qué le falta, dale los números exactos y un consejo corto.
            - Respuestas en texto plano sin formatos complejos."""
        else:
            system_content = f"""OPERANDO BAJO EL PROTOCOLO 'CALOFIT UNIFIED V3.0' (FALLO CERO).

            ### ESTATUS DEL USUARIO:
            {contexto}
            {texto_extra}

            ### 🚨 REGLA MAESTRA DE CATEGORIZACIÓN (OBLIGATORIO):
            Toda respuesta DEBE comenzar con una etiqueta de intención exacta. NO USES OTRAS ETIQUETAS.
            - [CALOFIT_INTENT: CHAT] -> Para saludos, dudas generales o consejos cortos.
            - [CALOFIT_INTENT: ITEM_RECIPE] -> Para una receta detallada (usar etiquetas blindadas).
            - [CALOFIT_INTENT: ITEM_WORKOUT] -> Para un ejercicio o rutina detallada (usar etiquetas blindadas).
            - [CALOFIT_INTENT: PLAN_DIET] -> Para planes de alimentación (formato tabla).
            - [CALOFIT_INTENT: PLAN_WORKOUT] -> Para planes de entrenamiento (formato tabla).

            ### 🛡️ REALITY CHECK (SEGURIDAD Y ÉTICA):
            SI EL USUARIO PIDE METAS IMPOSIBLES O PELIGROSAS (ej: "bajar 5kg en 2 días", "rutina de 4 horas", "no comer nada", "esteroides"):
            1. RECHAZA LA SOLICITUD AMABLEMENTE. No generes la rutina ni dieta solicitada.
            2. EDUCA AL USUARIO: Explica por qué es peligroso o imposible (pérdida de masa muscular, deshidratación, riesgo cardíaco).
            3. PROPÓN UNA ALTERNATIVA SEGURA Y REALISTA (ej: "Lo saludable es 0.5kg/semana", "Rutina de 45-60 min").
            4. USA EL TAG: [CALOFIT_INTENT: CHAT]

            ### 🤫 CONOCIMIENTO CULTURAL PERUANO (FALLO CERO):
            1. TACACHO: Se hace con Plátano Verde machacado y manteca/aceite. NO lleva pan. NO lleva yuca.
            2. CECINA: Es carne de cerdo ahumada. Se sirve con el Tacacho.
            3. RUTINAS: Si pides rutina, no inventes ejercicios de 'música' o 'baile' a menos que te lo pidan.

            ### 🏷️ ESTRUCTURA DE ETIQUETAS BLINDADAS (ITEM_RECIPE / ITEM_WORKOUT):
            Prohibido usar negritas. Prohibido usar paréntesis. Usa exactamente este formato:
            [CALOFIT_HEADER] Nombre del Item [/CALOFIT_HEADER]
            [CALOFIT_STATS] P: [X]g | C: [X]g | G: [X]g | Cal: [X]kcal [/CALOFIT_STATS]
            [CALOFIT_LIST]
            - Cantidad Elemento 1
            - Cantidad Elemento 2
            [/CALOFIT_LIST]
            [CALOFIT_ACTION]
            1. Paso...
            [/CALOFIT_ACTION]
            [CALOFIT_FOOTER] Nota del Coach [/CALOFIT_FOOTER]

            ❌ PROHIBIDO: Usar '###' o '**' dentro o cerca de las etiquetas. No escribas '### [CALOFIT_STATS]'. Escribe solo '[CALOFIT_STATS]'.
            ✅ FORMATO CORRECTO: [CALOFIT_HEADER] Nombre [/CALOFIT_HEADER]

            ### 🗓️ ESTRUCTURA DE PLANES (PLAN_DIET / PLAN_WORKOUT):
            Usa tablas simples:
            | Día | Mañana | Tarde | Noche |
            |---|---|---|---|
            | Lunes | ... | ... | ... |

            ### 🤫 REGLAS DE NEGOCIO Y CONOCIMIENTO CULTURAL:
            1. Saluda cordialmente al inicio (¡Hola [Nombre]!).
            2. CONOCIMIENTO PERUANO: El Tacacho SIEMPRE se hace con **Plátano Verde**, jamás con Yuca. Si sugieres Tacacho, usa Plátano.
            3. Toda respuesta DEBE comenzar estrictamente con el tag de intención.
            4. Jamás sugieras 'pichanga' o 'fútbol' en una rutina de entrenamiento estructurada.
            5. Si preguntan por calorías quemadas, usa el peso del usuario: {contexto}.
            """


        # Buscamos en todo el contexto disponible (System Content + Mensaje Usuario)
        contexto_total = (str(system_content) + " " + str(mensaje_usuario)).lower()
        
        # DEBUG MODE: Imprimir el mensaje del usuario y el contexto
        print(f"\n💬 [USER MESSAGE]: {mensaje_usuario}")
        print(f"🔍 [IA-DEBUG] Contexto Total Recibido (Primeros 500 chars): {contexto_total[:500]}...")

        restricciones_activas = []
        if "vegano" in contexto_total: restricciones_activas.append("DIETA VEGANA (CRÍTICA): Prohibido terminantemente: Carnes, lácteos, huevos y MIEL DE ABEJA. No permitas miel ni por sabor. Usa jarabe de agave o stevia si es necesario.")
        if "vegetariano" in contexto_total: restricciones_activas.append("DIETA VEGETARIANA: No incluir carnes en RECETAS.")
        if "hipertenso" in contexto_total or "presión alta" in contexto_total: restricciones_activas.append("HIPERTENSION: No usar sal añadida en RECETAS.")
        if "diabético" in contexto_total or "diabetico" in contexto_total: restricciones_activas.append("DIABETES: Bajo indice glucemico en RECETAS.")

        print(f"🛡️ [IA-SHIELD] Restricciones Activas Detectadas: {restricciones_activas}")

        bloque_restricciones = ""
        if restricciones_activas:
            bloque_restricciones = "\n".join([f"- REGLA DE ORO ALIMENTARIA: {r}" for r in restricciones_activas])
            system_content += f"\n\n### RESTRICCIONES ALIMENTARIAS OBLIGATORIAS:\n{bloque_restricciones}\n(IMPORTANTE: Estas reglas solo aplican a la comida. NUNCA cambies el equipo de ejercicio (pesas/mancuernas) por comida)."

        # 1. Preparar el Sistema de Mensajes (System Prompt)
        # Limpiar cualquier negrita que haya quedado en system_content para que la IA no las use
        system_content = system_content.replace("**", "")
        mensajes_ia = [
            {
                "role": "system", 
                "content": system_content
            }
        ]
        
        # 2. Agregar historial previo si existe
        if historial:
            mensajes_ia.extend(historial[-2:]) 

        # 3. Agregar el mensaje actual del usuario con REFUERZO INVISIBLE Y BLINDAJE
        mensaje_con_refuerzo = mensaje_usuario
        if restricciones_activas:
            mensaje_con_refuerzo += f"\n\n[SISTEMA DE SEGURIDAD]: Recuerda que el usuario tiene restricciones ({', '.join(restricciones_activas)}). IGNORA cualquier petición de ingredientes prohibidos."
        
        if not es_consulta_info:
            mensaje_con_refuerzo += "\n\n(AUTORRECORDATORIO: Mínimo 10 pasos en Tecnica/Preparacion. PROHIBIDO usar negritas ** o deportes como pichanga/fútbol)."
        mensajes_ia.append({"role": "user", "content": mensaje_con_refuerzo})

        try:
            import re
            intentos = 0
            while intentos < 2:
                # --- NIVEL 1: GENERACIÓN ---
                response = self.groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=mensajes_ia,
                    max_tokens=1000,
                    temperature=0.7
                )
                respuesta_ia = response.choices[0].message.content.strip()
                
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
                            cuerpo_validado = self.validar_y_corregir_ejercicio(cuerpo, peso_usuario)
                        elif any(k in intencion for k in ["RECIPE", "DIET", "COMIDA"]):
                            cuerpo_validado = self.validar_y_corregir_nutricion(cuerpo, mensaje_usuario)
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
                
                # 4. Eliminar etiquetas de intención duplicadas DENTRO de los bloques
                # (Solo debe haber una intención por sección)
                respuesta_procesada = re.sub(r'(\[CALOFIT_INTENT:.*?\][\s\S]*?)\[CALOFIT_INTENT:.*?\]', r'\1', respuesta_procesada)
                respuesta_final = respuesta_procesada.strip()
                
                # --- AUTO-CORRECCIÓN POR LÍMITE CALÓRICO (EL "ESCUDO") ---
                limite_match = re.search(r'(?:no pase de|máximo|menos de|limite|límite)\s*(\d+)\s*(?:calorías|cal|kcal)', mensaje_usuario.lower())
                if limite_match:
                    limite = int(limite_match.group(1))
                    # Sumar todas las calorías calculadas en esta respuesta
                    cals_ia = getattr(self, 'ultimas_calorias_calculadas', 0)
                    if cals_ia > limite + 30: # Margen de tolerancia
                        print(f"⚠️ [IA-SHIELD] Calorie Overflow detectado: {cals_ia} > {limite}. Reajustando porciones...")
                        mensajes_ia.append({"role": "assistant", "content": respuesta_ia})
                        mensajes_ia.append({"role": "user", "content": f"Esa opción tiene {cals_ia} kcal, pero te pedí máximo {limite} kcal. Por favor, AJUSTA LAS PORCIONES para que el total sea menor a {limite} kcal estrictamente."})
                        intentos += 1
                        continue
                
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
        NIVEL 3: CALCULADORA MATEMÁTICA REAL.
        Escanea la respuesta en busca de ingredientes y los valida contra la BD oficial.
        """
        from app.services.nutricion_service import nutricion_service
        import re

        # 1. ESCÁNER DE BLOQUE (LISTA)
        patron_bloque = r'\[CALOFIT_LIST\](.*?)\[/CALOFIT_LIST\]'
        bloques = re.findall(patron_bloque, respuesta_ia, re.DOTALL)
        texto_busqueda = bloques[-1] if bloques else respuesta_ia

        # v13.5: Traductores Regionales para Mapeo Directo (v29.0: Mapeos Lean)
        # IMPORTANTE: Orden específico (más específico primero)
        traductores = {
            "plátano verde": "plátano de seda",
            "plátano maduro": "plátano de seda",
            "tacacho": "plátano, de seda",
            "cecina": "cerdo, carne magra, cruda*",
            "chancho": "cerdo, carne magra, cruda*",
            "pechuga": "pollo, carne magra", # Pechuga antes que pollo general
            "pollo": "pollo, carne",
            "asado": "res, carne",
            "bistec": "res, carne magra", 
            "yucca": "yuca, raíz",
            "yuca": "yuca, raíz",
            "arroz": "arroz blanco corriente",
            "aceite de oliva": "aceite vegetal de oliva",
            "aceite de coco": "aceite vegetal de coco"
        }
        
        palabras_ruido = [
            "picado", "trozos", "cortada", "pelada", "fresca", "fresco", "al gusto", 
            "opcional", "maduros", "verdes", "frescos", "limpio", "limpia",
            "picados", "cortados", "en trozos", "grandes", "pequeños", "rebanadas"
        ]
        
        cals_total, prot_total, carb_total, gras_total = 0.0, 0.0, 0.0, 0.0
        ingredientes_no_encontrados = []
        encontrados_count = 0
        
        # Limpieza de pasos numerados en la lista (evita confusión con cantidades)
        texto_busqueda = re.sub(r'^\d+\.\s+.*$', '', texto_busqueda, flags=re.MULTILINE)
        
        # Regex de Ingredientes robusto (v37.0: Fix Bullet + Qty)
        # Estructura: (Inicio/Bullet) + (Cantidad y Unidad Opcionales) + (Resto/Nombre)
        patron_ingrediente = r'(?:^|\r?\n)\s*(?:[-\*•]\s*)?(?:(\d+(?:[.,/]\d+)?)\s*(?:(g|gr|gramos|taza|tazas|unidad|unidades|piezas|pieza|cucharada|cucharadas|cucharadita|cucharaditas|oz|ml|l|kg)\b)?\s*(?:de\s+)?)?([^\n]+)'
        matches = re.findall(patron_ingrediente, texto_busqueda, re.MULTILINE | re.IGNORECASE)
        
        for cant_raw, unidad_raw, nombre_raw in matches:
            try:
                # v31.0: Heurística de Cantidad Inteligente (Filtro Anti-Grasa)
                nombre_base = nombre_raw.lower().strip()
                
                # Parsing de cantidad (Soporte decimal y fracción)
                cantidad = 0.0
                if cant_raw:
                    cant_clean = cant_raw.replace(',', '.')
                    if '/' in cant_clean:
                        try:
                            num, den = cant_clean.split('/')
                            cantidad = float(num) / float(den)
                        except:
                            cantidad = 1.0 # Fallback
                    else:
                        cantidad = float(cant_clean)
                
                # Si no hay cantidad (y no se pudo parsear), decidir fallback
                if cantidad == 0.0:
                    # Si es aceite, grasa, mantequilla, aliño -> 15g (1 cucharada)
                    palabras_grasa = ["aceite", "mantequilla", "manteca", "aliño", "crema", "mayonesa", "margarina"]
                    if any(pg in nombre_base for pg in palabras_grasa):
                        cantidad = 15.0
                    else:
                        cantidad = 150.0 # Ingrediente base

                unidad = (unidad_raw or ("g" if not cant_raw else "")).strip().lower()
                
                # 1. Limpieza de nombre
                nombre_base = re.split(r'[,;\(\)]', nombre_base)[0].strip().rstrip('.')
                for ruido in palabras_ruido:
                    nombre_base = nombre_base.replace(ruido, "").strip()
                
                if len(nombre_base) < 3: continue

                # 2. Aplicar Traductores Regionales
                for t_orig, t_dest in traductores.items():
                    if t_orig in nombre_base:
                        nombre_base = t_dest
                        break

                # 3. Heurística de Unidades Mejorada
                if not unidad:
                    if any(u in nombre_base for u in ["huevo", "manzana", "plátano", "pan", "tostada", "fruta"]):
                        unidad = "unidad"
                    elif any(u in nombre_base for u in ["rebanada", "tajada", "loncha"]):
                        unidad = "rebanada"

                # v3.1: Parsing secundario de cantidad en paréntesis (ej: "pechuga (120g)")
                match_parens = re.search(r'\((\d+(?:[.,]\d+)?)\s*(g|gr|ml)\)', nombre_base)
                if match_parens:
                    cantidad = float(match_parens.group(1).replace(',', '.'))
                    unidad = match_parens.group(2)
                    # print(f"🎯 Parsing Exacto desde Paréntesis: {cantidad}{unidad}")

                # 4. Normalización de peso
                if unidad in ['g', 'gr', 'gramos', 'ml']: pass
                elif unidad in ['taza', 'tazas']: cantidad *= 200 
                elif unidad in ['unidad', 'unidades', 'pieza', 'piezas']: 
                    # Pechuga promedio = 150g, Huevo = 60g
                    cantidad *= 150 if any(x in nombre_base for x in ["pechuga", "carne", "bistec"]) else (60 if "huevo" in nombre_base else 100)
                elif unidad in ['rebanada', 'rebanadas', 'tajada', 'tajadas']:
                    cantidad *= 30 # Tajada estándar (bajado de 40 a 30)
                elif unidad in ['cucharada', 'cucharadas']: cantidad *= 15
                elif unidad in ['cucharadita', 'cucharaditas']: cantidad *= 5
                elif unidad == 'kg': cantidad *= 1000

                # 5. Búsqueda en BD
                info = nutricion_service.obtener_info_alimento(nombre_base)
                if info:
                    nombre_encontrado = info.get("alimento", "").lower()
                    
                    # VALIDACIÓN DE CATEGORÍA: Evitar que 'Pechuga' mapee a 'Bazo'
                    # Si buscamos pollo/pechuga y encontramos visceras (bazo, higado, riñon), forzar re-búsqueda o ignorar
                    if "pechuga" in nombre_base and any(v in nombre_encontrado for v in ["bazo", "higado", "riñon", "sangrecita"]):
                        # Fallback manual seguro
                        info = {"alimento": "pollo, carne magra", "calorias_100g": 110, "proteina_100g": 23.0, "carbohindratos_100g": 0.0, "grasas_100g": 1.2}
                        nombre_encontrado = "pollo, carne magra (fallback)"
                        # print(f"⚠️ Corrección de Mapeo: '{nombre_base}' redirigido a Pollo Magro.")

                    f = cantidad / 100.0
                    
                    cal_base = (info.get("calorias_100g") or 0)
                    prot_base = (info.get("proteina_100g") or 0)
                    carb_base = (info.get("carbohindratos_100g") or 0)
                    gras_base = (info.get("grasas_100g") or 0)
                    
                    # v35.0: MODIFICADORES DE COCCIÓN (Lógica de Fritura)
                    if any(x in nombre_raw.lower() for x in ["frito", "frita", "fritos", "fritas"]):
                        if "frito" not in nombre_encontrado:
                            gras_base += 8.0
                            cal_base += 72.0
                            print(f"🍳 Lógica de Cocción: Penalizando fritura para '{nombre_raw}'")

                    cals_item = cal_base * f
                    cals_total += cals_item
                    prot_total += prot_base * f
                    carb_total += carb_base * f
                    gras_total += gras_base * f
                    
                    # print(f"📊 Nutri-Debug: '{nombre_raw}' ({cant_raw or '?'}) -> '{nombre_encontrado}' | Qty: {cantidad}{unidad} | F: {f:.2f} | Cals: {cals_item:.2f}")

                    encontrados_count += 1
                else:
                    if nombre_base not in ["sal", "pimienta", "agua", "hielo", "vinagre", "limón", "jugo de limón"]:
                        ingredientes_no_encontrados.append(nombre_base)
            except: continue

        # Inyección de Stats Blindadas
        if encontrados_count > 0:
            regex_stats = r'\[CALOFIT_STATS\].*?\[/CALOFIT_STATS\]'
            res_final = f"P: {prot_total:.1f}g | C: {carb_total:.1f}g | G: {gras_total:.1f}g | Cal: {cals_total:.0f}kcal"
            if any(x in respuesta_ia.lower() for x in ["frito", "frita", "fritos", "fritas"]):
                res_final += " (Aceite incluido) 🍳"
            
            macros_inyectados = f"[CALOFIT_STATS] {res_final} [/CALOFIT_STATS]"
            
            # Limpiar el Header de basura numérica (IA suele meter macros ahí)
            # e.j. [CALOFIT_HEADER] Pollo P: 20g [/CALOFIT_HEADER] -> [CALOFIT_HEADER] Pollo [/CALOFIT_HEADER]
            respuesta_ia = re.sub(r'(\[CALOFIT_HEADER\].*?) (?:P|C|G|Cal):.*?(?=\[/CALOFIT_HEADER\])', r'\1', respuesta_ia, flags=re.IGNORECASE)

            # Reemplazar TODAS las etiquetas de stats (v26.1)
            if re.search(regex_stats, respuesta_ia, re.DOTALL):
                respuesta_ia = re.sub(regex_stats, macros_inyectados, respuesta_ia, flags=re.DOTALL)
            else:
                if "[/CALOFIT_HEADER]" in respuesta_ia:
                    respuesta_ia = respuesta_ia.replace("[/CALOFIT_HEADER]", "[/CALOFIT_HEADER]\n" + macros_inyectados)
            
            self.ultimas_calorias_calculadas = cals_total
            
            if ingredientes_no_encontrados:
                msg = f"\n⚠️ Info: {', '.join(list(set(ingredientes_no_encontrados))[:2])} no se mapearon del todo."
                if "[/CALOFIT_FOOTER]" in respuesta_ia:
                    respuesta_ia = respuesta_ia.replace("[/CALOFIT_FOOTER]", msg + "\n[/CALOFIT_FOOTER]")
                else:
                    respuesta_ia += msg
        
        return respuesta_ia

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
            
            # Inyección de Stats
            macros_inyectados = f"[CALOFIT_STATS] P: 0g | C: 0g | G: 0g | Cal: {cals_quemadas:.0f}kcal [/CALOFIT_STATS]"
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


ia_engine = IAService()
