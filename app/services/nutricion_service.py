import json
import os
from typing import Dict, Optional

class NutricionService:
    _instance = None
    _datos_nutricionales: Dict[str, dict] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(NutricionService, cls).__new__(cls)
            cls._instance.cargar_datos()
        return cls._instance

    def cargar_datos(self):
        """Carga los datos del archivo JSON oficial al diccionario en memoria."""
        try:
            # Ruta absoluta robusta basada en la ubicación del script
            actual_dir = os.path.dirname(os.path.abspath(__file__))
            json_path = os.path.join(actual_dir, "..", "data", "alimentos_peru_ins.json")
            
            # Debug para ver qué ruta está intentando usar
            print(f"🔍 Intentando cargar JSON desde: {json_path}")

            if not os.path.exists(json_path):
                 raise FileNotFoundError(f"No se encontró el archivo en {json_path}")

            with open(json_path, 'r', encoding='utf-8') as f:
                lista_alimentos = json.load(f)
                # Crear diccionario indexado por nombre para búsqueda rápida O(1)
                for item in lista_alimentos:
                    clave_raw = item.get("alimento") or item.get("nombre")
                    if clave_raw:
                        clave = clave_raw.lower()
                        self._datos_nutricionales[clave] = item
            print(f"✅ NutricionService: Cargados {len(self._datos_nutricionales)} alimentos oficiales del CENAN/USDA.")
        except Exception as e:
            print(f"❌ NutricionService Error: No se pudo cargar {json_path}. {e}")

    def obtener_info_alimento(self, nombre: str) -> Optional[dict]:
        """Busca un alimento con lógica de prioridad: Exacto > Empieza con > Intersección de Palabras."""
        nombre = nombre.lower().strip()
        if not nombre: return None

        # 0. MAPEO DE SINÓNIMOS REGIONALES (v6.5 - Traduciendo al formato INS/Perú)
        sinonimos = {
            "aguacate": "palta", 
            "jitomate": "tomate",
            "ejote": "vainita",
            "vainitas": "vainita",
            "cacahuate": "maní",
            "puerco": "cerdo", 
            "chancho": "cerdo",
            "vaca": "res",
            "jengibre": "kion",
            "soja": "soya",
            "calabaza": "zapallo",
            "betabel": "beterraga",
            "elote": "choclo",
            "chicharo": "arveja",
            "verduras mixtas": "lechuga",
            "vegetales variados": "lechuga",
            "ensalada": "lechuga",
            "mix de verduras": "lechuga",
            "vegetales": "lechuga",
            "lomito": "lomo",
            "frances": "francés",
            "carne": "res",
            "bisteck": "bistec",
            "limon": "limón",
            "platano": "plátano",
            "tofu": "soya"
        }
        for s, r in sinonimos.items():
            if s in nombre:
                nombre = nombre.replace(s, r)
        
        # 1. Búsqueda exacta (Máxima prioridad)
        if nombre in self._datos_nutricionales:
            return self._datos_nutricionales[nombre]
        
        # 2. Búsqueda por "Empieza con"
        for key, data in self._datos_nutricionales.items():
            if key.startswith(nombre):
                return data


        # 3. Búsqueda por Intersección con Puntuación (MÁXIMA FLEXIBILIDAD)
        palabras_busqueda = [p for p in nombre.split() if len(p) > 2] # ['lomo', 'pollo']
        if not palabras_busqueda: return None

        mejor_match = None
        max_puntos = 0

        for key, data in self._datos_nutricionales.items():
            # Puntuación base
            puntos = sum(1 for p in palabras_busqueda if p in key)
            
            # Bono de precisión: Si el match está al inicio del nombre (Ej: "Pollo...")
            if any(key.startswith(p) for p in palabras_busqueda):
                puntos += 1.5
            
            if puntos > max_puntos:
                max_puntos = puntos
                mejor_match = data
            elif puntos > 0 and puntos == max_puntos:
                # Preferimos el nombre más corto para evitar "caldo de pollo" vs "pollo"
                if len(key) < len(list(self._datos_nutricionales.keys())[list(self._datos_nutricionales.values()).index(mejor_match)]):
                    mejor_match = data

        # (v7.7 - Umbral Relajado): Si encontramos algo con al menos 1 coincidencia fuerte, lo devolvemos.
        if mejor_match and max_puntos >= 1:
            return mejor_match
        
        # 4. Fallback: Búsqueda parcial simple
        for key, data in self._datos_nutricionales.items():
            if nombre in key:
                return data
        
        # 5. Si todo falla, devolver el que tenga al menos un match si es significativo
        if mejor_match and max_puntos >= 1:
            return mejor_match
        
        return None

    def obtener_proteina_100g(self, nombre: str) -> float:
        """Devuelve la proteína por 100g o 0.0 si no existe."""
        info = self.obtener_info_alimento(nombre)
        return info["proteina_100g"] if info else 0.0

# Instancia global única
nutricion_service = NutricionService()
