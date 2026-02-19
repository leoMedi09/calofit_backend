import asyncio
import os
import sys
import json

# Añadir el directorio raíz al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Mock de Groq para evitar error de librería faltante en el entorno de pruebas
class MockGroqResponse:
    def __init__(self, content):
        self.choices = [type('Choice', (), {'message': type('Message', (), {'content': content})})]

async def mock_groq_call(*args, **kwargs):
    # Simulamos lo que Groq devolvería normalmente basándose en la última frase enviada
    prompt = args[0] if args else kwargs.get('messages', [{}])[-1].get('content', '')
    
    if "Nutella" in prompt:
        return MockGroqResponse(json.dumps({
            "alimentos_detectados": ["Nutella", "Plátano"],
            "calorias": 600, "proteinas_g": 8, "carbohidratos_g": 70, "grasas_g": 32,
            "azucar_g": 55, "fibra_g": 4, "calidad_nutricional": "Baja"
        }))
    elif "Lomo Saltado" in prompt:
        return MockGroqResponse(json.dumps({
            "alimentos_detectados": ["Lomo Saltado"],
            "calorias": 750, "proteinas_g": 45, "carbohidratos_g": 60, "grasas_g": 35,
            "azucar_g": 2, "fibra_g": 5, "calidad_nutricional": "Media-Alta"
        }))
    else:
        return MockGroqResponse(json.dumps({
            "alimentos_detectados": ["Leche Gloria"],
            "calorias": 150, "proteinas_g": 8, "carbohidratos_g": 10, "grasas_g": 8,
            "azucar_g": 9, "fibra_g": 0, "calidad_nutricional": "Media"
        }))

# Inyectamos el mock antes de importar ia_engine si fuera necesario, 
# pero es más fácil mockear el método directamente en la instancia
from app.services.ia_service import ia_engine
import unittest.mock as mock

# Reemplazamos la llamada a Groq real por nuestra simulación
ia_engine.groq_client = mock.AsyncMock()
ia_engine.groq_client.chat.completions.create = mock_groq_call

async def test_flujo_completo():
    print("\n" + "🚀" * 15)
    print(" INICIANDO TEST FINAL DE ASISTENTE")
    print("🚀" * 15 + "\n")

    # ---------------------------------------------------------
    # PARTE 1: PREGUNTAS Y RESPUESTAS (CHAT)
    # ---------------------------------------------------------
    print("🟢 TEST 1: CONSULTAS DE NUTRICIÓN")
    print("-" * 30)
    
    preguntas = [
        "¿Cuáles son los mejores superfoods peruanos para ganar músculo?",
        "Tengo diabetes, ¿puedo comer mango peruano? ¿cuánto?",
        "Dime una cena rápida y barata con ingredientes de mercado en Lima"
    ]
    
    perfil_contexto = "Nombre: Leonardo, Objetivo: Ganar masa muscular, Pais: Perú"

    for p in preguntas:
        print(f"\n👤 PREGUNTA: {p}")
        try:
            respuesta = await ia_engine.asistir_cliente(perfil_contexto, p)
            # Limpiar un poco la respuesta para el log
            print(f"🤖 CALOFIT: {respuesta[:400]}...") 
        except Exception as e:
            print(f"❌ Error en chat: {e}")

    # ---------------------------------------------------------
    # PARTE 2: REGISTRO POR TEXTO (CON CORRECCIÓN SQLITE)
    # ---------------------------------------------------------
    print("\n\n🟢 TEST 2: REGISTRO DE COMIDAS (CORRECCIÓN CON DB)")
    print("-" * 30)
    
    frases_registro = [
        "Comí 100g de Nutella y un plátano", # Nutella (SQLite) + Plátano (JSON/IA)
        "Me zampé un Lomo Saltado grande",    # Plato peruano (JSON)
        "Registra un tarro de leche Gloria",   # Marca peruana (JSON OFF)
    ]

    for frase in frases_registro:
        print(f"\n📝 REGISTRO: '{frase}'")
        try:
            # Simulamos peso usuario 75kg
            res = await ia_engine.extraer_macros_de_texto(frase, 75.0)
            
            print(f"📦 Detectado: {res.get('alimentos_detectados')}")
            print(f"🔥 Calorías: {res.get('calorias')} kcal")
            print(f"🍭 Azúcar:   {res.get('azucar_g')}g")
            print(f"🧂 Sodio:    {res.get('sodio_mg', 0)}mg")
            print(f"🛡️ Calidad:  {res.get('calidad_nutricional')}")
        except Exception as e:
            print(f"❌ Error en registro: {e}")

if __name__ == "__main__":
    asyncio.run(test_flujo_completo())
