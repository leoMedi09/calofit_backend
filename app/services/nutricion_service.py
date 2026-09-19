import json
import os
import sqlite3
from typing import Dict, Optional

from app.core.alimentos_ux_filters import es_alimento_bloqueado_ia

class NutricionService:
    _instance = None
    _datos_nutricionales: Dict[str, dict] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(NutricionService, cls).__new__(cls)
            cls._instance.cargar_datos()
        return cls._instance

    def cargar_datos(self):
        """Carga los datos del archivo JSON oficial y comercial al diccionario en memoria."""
        try:
            actual_dir = os.path.dirname(os.path.abspath(__file__))
            
            json_path_ins = os.path.join(actual_dir, "..", "data", "alimentos_peru_ins.json")
            if os.path.exists(json_path_ins):
                with open(json_path_ins, 'r', encoding='utf-8') as f:
                    lista_alimentos = json.load(f)
                    for item in lista_alimentos:
                        clave_raw = item.get("alimento") or item.get("nombre")
                        if clave_raw and not es_alimento_bloqueado_ia(clave_raw):
                            self._datos_nutricionales[clave_raw.lower()] = item
                print(f"[*] NutricionService: Cargados {len(lista_alimentos)} alimentos oficiales del CENAN/INS.")

            json_path_off = os.path.join(actual_dir, "..", "data", "alimentos_peru_off.json")
            if os.path.exists(json_path_off):
                with open(json_path_off, 'r', encoding='utf-8') as f:
                    lista_off = json.load(f)
                    for item in lista_off:
                        clave_raw = item.get("alimento")
                        if clave_raw and not es_alimento_bloqueado_ia(clave_raw):
                            self._datos_nutricionales[clave_raw.lower()] = item
                print(f"[*] NutricionService: Cargados {len(lista_off)} productos comerciales de OpenFoodFacts.")
            else:
                 print(f"[!] NutricionService: No se encontró {json_path_off}, usando solo base oficial.")

        except Exception as e:
            print(f"[ERR] NutricionService Error: {e}")

    def _buscar_en_sqlite(self, nombre_busqueda: str) -> Optional[dict]:
        """Busca en la base de datos masiva SQLite con caché y optimización de índices."""
        if not hasattr(self, '_sqlite_cache'):
            self._sqlite_cache = {}
        
        if nombre_busqueda in self._sqlite_cache:
            return self._sqlite_cache[nombre_busqueda]

        try:
            db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "alimentos_mundo.db")
            if not os.path.exists(db_path):
                return None
                
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                
                query_prefix = "SELECT * FROM alimentos WHERE nombre LIKE ? ORDER BY calorias DESC LIMIT 1"
                cursor.execute(query_prefix, (f"{nombre_busqueda}%",))
                row = cursor.fetchone()
                
                if not row:
                    cursor.execute("SELECT * FROM alimentos WHERE nombre LIKE ? ORDER BY calorias DESC LIMIT 1", (f"%{nombre_busqueda}%",))
                    row = cursor.fetchone()
                
                if row:
                    if es_alimento_bloqueado_ia(row[1]):
                        return None
                    
                    res = {
                        "nombre": row[1],
                        "alimento": row[1],
                        "marca": row[2] or "Genérico",
                        "origen": "BBDD Mundial 🌎",
                        "calorias": row[3],
                        "proteinas": row[4],
                        "carbohidratos": row[5],
                        "grasas": row[7],
                        "azucares": row[6],
                        "grasas_saturadas": row[8],
                        "grasas_trans": row[9],
                        "fibra": row[12],
                        "sodio": (row[13] or 0) * 1000,
                        "calcio": (row[14] or 0) * 1000,
                        "hierro": (row[15] or 0) * 1000,
                        "vitamina_a": (row[16] or 0) * 1000000,
                        "vitamina_c": (row[17] or 0) * 1000
                    }
                    self._sqlite_cache[nombre_busqueda] = res
                    return res
        except Exception as e:
            print(f"⚠️ Error SQLite search: {e}")
            return None

    def obtener_info_alimento(self, nombre: str) -> Optional[dict]:
        """Busca un alimento con lógica de prioridad: Exacto > Intersección RAM > SQLite."""
        nombre_clean = nombre.lower().strip()
        if not nombre_clean: return None

        if not hasattr(self, '_fallos_cache'):
            self._fallos_cache = set()
        if nombre_clean in self._fallos_cache:
            return None

        sinonimos = {
            "aguacate": "palta", "jitomate": "tomate", "ejote": "vainita",
            "cacahuate": "maní", "puerco": "cerdo", "chancho": "cerdo",
            "vaca": "res", "jengibre": "kion", "soja": "sillao", "soya": "sillao", "calabaza": "zapallo", "lentejas rojas": "lenteja roja", "lenteja roja": "lenteja roja",
            "betabel": "beterraga", "elote": "choclo", "chicharo": "arveja",
            "frances": "francés", "platano": "plátano", "brócoli": "brocoli"
        }
        for s, r in sinonimos.items():
            if s in nombre_clean:
                nombre_clean = nombre_clean.replace(s, r)

        if nombre_clean in self._datos_nutricionales:
            return self._normalizar_ram(self._datos_nutricionales[nombre_clean])

        best = None
        best_score = -1
        for k, v in self._datos_nutricionales.items():
            pos = nombre_clean.find(k)
            if pos == -1:
                if k in nombre_clean: pos = 0
                elif nombre_clean in k: pos = k.find(nombre_clean)
                else: continue
            
            if len(k) < 3: continue
            if es_alimento_bloqueado_ia(k) or es_alimento_bloqueado_ia(v.get("alimento") or v.get("nombre")):
                continue
            current_score = (1000 / (pos + 1)) + len(k)
            if current_score > best_score:
                best = v
                best_score = current_score
        
        if best and not es_alimento_bloqueado_ia(best.get("alimento") or best.get("nombre")):
            return self._normalizar_ram(best)

        if len(nombre_clean) > 3:
            resultado_sql = self._buscar_en_sqlite(nombre_clean)
            if resultado_sql:
                return resultado_sql

        self._fallos_cache.add(nombre_clean)
        return None


    def _normalizar_ram(self, item_raw: dict) -> dict:
        """Normaliza los datos crudos del JSON de Perú/INS para coincidir con el esquema SQLite."""
        
        nombre = item_raw.get("alimento") or item_raw.get("nombre") or "Desconocido"
        
        cal = item_raw.get("calorias") or item_raw.get("calorias_100g") or item_raw.get("Energía (kcal)") or item_raw.get("Energía \n(kcal)") or 0
        prot = item_raw.get("proteinas") or item_raw.get("proteina_100g") or item_raw.get("Proteína \n(g)") or item_raw.get("Proteína (g)") or 0
        carb = item_raw.get("carbohidratos") or item_raw.get("carbohindratos_100g") or item_raw.get("Carbohidratos \n(g)") or item_raw.get("Carbohidratos totales (g)") or 0
        gras = item_raw.get("grasas") or item_raw.get("grasas_100g") or item_raw.get("Grasa \n(g)") or item_raw.get("Grasa total (g)") or 0

        azu = item_raw.get("azucares") or 0
        fib = item_raw.get("fibra") or item_raw.get("Fibra \n(g)") or 0
        sod = item_raw.get("sodio") or 0

        try:
            return {
                "nombre": nombre,
                "alimento": nombre,
                "marca": item_raw.get("marca", "Genérico / Perú"),
                "origen": "Base Perú 🇵🇪",
                
                "calorias": float(str(cal).replace(',','.')),
                "proteinas": float(str(prot).replace(',','.')),
                "carbohidratos": float(str(carb).replace(',','.')),
                "grasas": float(str(gras).replace(',','.')),
                
                "azucares": float(str(azu).replace(',','.')),
                "fibra": float(str(fib).replace(',','.')),
                "sodio": float(str(sod).replace(',','.')),
                
                "grasas_saturadas": 0.0,
                "calcio": 0.0,
                "hierro": 0.0,
                "vitamina_a": 0.0,
                "vitamina_c": 0.0
            }
        except Exception:
            return {
                "nombre": nombre, "calorias": 0, "proteinas": 0, 
                "carbohidratos": 0, "grasas": 0, "origen": "Error Parseo"
            }


nutricion_service = NutricionService()
