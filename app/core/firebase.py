import firebase_admin
from firebase_admin import credentials, auth
import os
import json

def initialize_firebase():
    # 1. Intentamos obtener el JSON desde la variable de entorno
    firebase_info = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
    
    try:
        if firebase_info:
            # Si la variable existe, la cargamos (limpiamos posibles comillas extra)
            cred_dict = json.loads(firebase_info.strip("'"))
            cred = credentials.Certificate(cred_dict)
            print("🔥 Firebase: Inicializado mediante VARIABLE DE ENTORNO")
        else:
            # 2. Si no hay variable, usamos tu lógica actual del archivo local
            current_dir = os.path.dirname(__file__)
            path_to_json = os.path.join(current_dir, "calofit-c8c24-firebase-adminsdk-fbsvc-ae08774a9b.json")
            
            if not os.path.exists(path_to_json):
                raise FileNotFoundError(f"No se encontró el archivo JSON en: {path_to_json}")
                
            cred = credentials.Certificate(path_to_json)
            print("🔥 Firebase: Inicializado mediante ARCHIVO LOCAL")

        # Evitar inicializar la app más de una vez
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)
            
    except Exception as e:
        print(f"❌ Error crítico en Firebase: {e}")
        # En producción, esto debería detener la app
        raise e

# Ejecutamos la inicialización al importar el módulo
initialize_firebase()

def verify_firebase_token(id_token: str):
    """
    Verifica el token que enviará la App de Flutter
    """
    try:
        decoded_token = auth.verify_id_token(id_token)
        return decoded_token
    except Exception as e:
        print(f"Error al verificar token: {e}")
        return None