from .models import loaded_models
import numpy as np

def predict_daily_requirement(age, weight, height, gender):
    
    input_data = np.array([[age, weight, height, gender]])
    prediction = loaded_models.caloric_model.predict(input_data)
    return float(prediction[0])

def extract_entities_from_chat(text):
    
    doc = loaded_models.nlp(text)
    entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
    return entities