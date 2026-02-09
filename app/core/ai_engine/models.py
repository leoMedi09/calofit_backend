import joblib
import spacy
from tensorflow.keras.models import load_model
import os

# Rutas a tus archivos descargados
MODEL_DIR = os.path.dirname(__file__)

class ModelLoader:
    def __init__(self):
        
        self.caloric_model = joblib.load(os.path.join(MODEL_DIR, "caloric_regressor.pkl"))
        
        
        self.exercise_model = load_model(os.path.join(MODEL_DIR, "fitrec_ann.h5"))
        
        
        self.nlp = spacy.load("es_core_news_md")

# Instancia global para evitar recargas constantes
loaded_models = ModelLoader()