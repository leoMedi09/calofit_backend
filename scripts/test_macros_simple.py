import asyncio
import os
import sys
import json

# Setup path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from app.services.nutricion_service import NutricionService

# MOCK DE GROQ RESPONSE (Simulamos lo que diría la IA)
GROQ_MOCK_RESPONSES = {
    "Me comí 100g de Nutella": {
        "alimentos_detectados": ["Nutella"],
        "calorias": 530,  # Estimación IA
        "proteinas_g": 6,
        "carbohidratos_g": 56,
        "grasas_g": 30,
        "azucar_g": 50, # La IA a veces acierta, a veces no
        "fibra_g": 2,
        "calidad_nutricional": "Baja"
    },
    "Una Manzana y un Panetón": {
        "alimentos_detectados": ["Manzana", "Panetón"],
        "calorias": 450, # 95 + 355 aprox
        "proteinas_g": 6,
        "carbohidratos_g": 80,
        "grasas_g": 15,
        "azucar_g": 40,
        "fibra_g": 5,
        "calidad_nutricional": "Media"
    },
    "Comí asdfoijwefoij": {
        "alimentos_detectados": ["asdfoijwefoij"],
        "calorias": 0,
        "calidad_nutricional": "Desconocida"
    }
}

async def simular_extraer_macros(texto, service):
    print(f"\n👤 Usuario: '{texto}'")
    
    # 1. Simular respuesta de Groq
    response_json = GROQ_MOCK_RESPONSES.get(texto, {"alimentos_detectados": [], "calorias": 0})
    print(f"🤖 IA (Groq Inicial): {json.dumps(response_json, indent=2)}")

    # 2. LOGICA DE CORRECCIÓN (Copiada de ia_service.py)
    if service and response_json.get("alimentos_detectados"):
        print("🔍 IA: Corrigiendo macros con datos reales de la Base de Datos...")
        
        total_cal = 0
        total_prot = 0
        total_carb = 0
        total_gras = 0
        total_azucar = 0
        total_fibra = 0
        total_sodio = 0
        
        alimentos_corregidos = []
        
        for alimento_nombre in response_json["alimentos_detectados"]:
            info_real = service.obtener_info_alimento(alimento_nombre)
            
            if info_real:
                print(f"✅ DATO REAL ENCONTRADO: {alimento_nombre} -> {info_real['nombre']} ({info_real['calorias']} kcal)")
                
                total_cal += info_real.get('calorias', 0)
                total_prot += info_real.get('proteinas', 0)
                total_carb += info_real.get('carbohidratos', 0)
                total_gras += info_real.get('grasas', 0)
                
                total_azucar += info_real.get('azucares', 0) or 0
                total_fibra += info_real.get('fibra', 0) or 0
                total_sodio += info_real.get('sodio', 0) or 0
                
                alimentos_corregidos.append(f"{info_real['nombre']} (Verificado)")
            else:
                print(f"⚠️ No encontrado en BD: {alimento_nombre}. Manteniendo estimación IA.")
                pass

        if len(alimentos_corregidos) == len(response_json["alimentos_detectados"]) and total_cal > 0:
            print(f"📊 ACTUALIZANDO con precisión (100% Match): Antes {response_json['calorias']} -> Ahora {total_cal}")
            response_json["calorias"] = float(round(total_cal, 1))
            response_json["proteinas_g"] = float(round(total_prot, 1))
            response_json["carbohidratos_g"] = float(round(total_carb, 1))
            response_json["grasas_g"] = float(round(total_gras, 1))
            response_json["azucar_g"] = float(round(total_azucar, 1))
            response_json["fibra_g"] = float(round(total_fibra, 1))
            if "calidad_nutricional" not in response_json: response_json["calidad_nutricional"] = ""
            response_json["calidad_nutricional"] += " (Verificado con Base de Datos Oficial)"
        else:
             print(f"⚠️ Cobertura parcial ({len(alimentos_corregidos)}/{len(response_json['alimentos_detectados'])}). Usando estimación IA.")

    print(f"🏁 RESULTADO FINAL: {json.dumps(response_json, indent=2)}")

async def main():
    print("🚀 Iniciando NutricionService...")
    service = NutricionService()
    
    await simular_extraer_macros("Me comí 100g de Nutella", service)
    await simular_extraer_macros("Una Manzana y un Panetón", service) # Panetón estimo que lo encontrará si está en JSON/SQLite
    await simular_extraer_macros("Comí asdfoijwefoij", service)

if __name__ == "__main__":
    asyncio.run(main())
