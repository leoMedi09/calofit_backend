import json
import os
from typing import Dict, List, Optional

class EjerciciosService:
    _instance = None
    _ejercicios_db: Dict[str, dict] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EjerciciosService, cls).__new__(cls)
            cls._instance.cargar_datos()
        return cls._instance

    def cargar_datos(self):
        """Carga los datos del archivo ejercicios.json."""
        try:
            actual_dir = os.path.dirname(os.path.abspath(__file__))
            json_path = os.path.join(actual_dir, "..", "data", "ejercicios.json")
            
            if not os.path.exists(json_path):
                 print(f"❌ EjerciciosService: No se encontró {json_path}")
                 return

            with open(json_path, 'r', encoding='utf-8') as f:
                lista_ejercicios = json.load(f)
                for item in lista_ejercicios:
                    nombre = item.get("nombre")
                    if nombre:
                        # Indexar por nombre y por ID
                        self._ejercicios_db[nombre.lower()] = item
                        self._ejercicios_db[item["id"].lower()] = item
                        # También por alias si existen
                        for alias in item.get("alias", []):
                            self._ejercicios_db[alias.lower()] = item
            
            print(f"✅ EjerciciosService: Cargados {len(lista_ejercicios)} ejercicios oficiales.")
        except Exception as e:
            print(f"❌ EjerciciosService Error: {e}")

    def obtener_info_ejercicio(self, consulta: str) -> Optional[dict]:
        """Busca un ejercicio por nombre, id o alias."""
        consulta = consulta.lower().strip()
        
        # 1. Búsqueda exacta
        if consulta in self._ejercicios_db:
            return self._ejercicios_db[consulta]
        
        # 2. Búsqueda parcial
        for key, data in self._ejercicios_db.items():
            if key in consulta or consulta in key:
                return data
        
        return None

    def calcular_calorias(self, met: float, peso_kg: float, minutos: float) -> float:
        """
        Fórmula: Calorías = (MET * 3.5 * peso_kg / 200) * minutos
        """
        return (met * 3.5 * peso_kg / 200) * minutos

# Instancia global
ejercicios_service = EjerciciosService()
