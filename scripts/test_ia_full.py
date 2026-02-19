import asyncio
import os
import sys

# Añadir el directorio raíz al path para importar módulos de la app
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.services.ia_service import IAService

async def test_preguntas():
    print("\n" + "="*50)
    print("🤖 TEST 1: CONSULTAS INTELIGENTES (CHAT)")
    print("="*50)
    
    ia = IAService()
    
    # Contexto simulado de un usuario
    perfil = {
        "first_name": "Leonardo",
        "age": 25,
        "goal": "ganar_masa",
        "gender": "M"
    }
    
    contexto = f"Usuario: {perfil['first_name']}, Objetivo: {perfil['goal']}"
    
    preguntas = [
        "¿Qué puedo desayunar rico y barato en Perú?",
        "¿La quinua engorda?",
        "Estoy estresado y quiero comer dulce, ¿qué hago?"
    ]
    
    for p in preguntas:
        print(f"\n👤 Usuario: {p}")
        print("⏳ Pensando...")
        try:
            respuesta = await ia.asistir_cliente(contexto, p)
            print(f"🤖 IA: {respuesta[:300]}...") # Mostrar solo los primeros 300 chars
        except Exception as e:
            print(f"❌ Error: {e}")

async def test_registro():
    print("\n" + "="*50)
    print("📝 TEST 2: REGISTRO DE ALIMENTOS (MACROS REALES)")
    print("="*50)
    
    ia = IAService()
    
    # Casos de prueba
    frases = [
        "Me comí 100g de Nutella",  # Caso GLOBAL (SQLite) -> Debe tener mucho AZÚCAR
        "Un plato de Lentejas",     # Caso PERÚ (JSON Ram) -> Debe ser preciso
        "Una manzana mediana",      # Caso GENÉRICO
        "Un asdfoijwefoij"          # Caso BASURA -> ¿Qué hace?
    ]
    
    for frase in frases:
        print(f"\n👤 Usuario: '{frase}'")
        print("⏳ Extrayendo macros...")
        try:
            # Usamos peso default 70kg
            resultado = await ia.extraer_macros_de_texto(frase, 70.0)
            
            print(f"📦 Detectado: {resultado.get('alimentos_detectados')}")
            print(f"🔥 Calorías: {resultado.get('calorias')} kcal")
            print(f"🥩 Proteína: {resultado.get('proteinas_g')}g")
            print(f"🍞 Carbos:   {resultado.get('carbohidratos_g')}g")
            print(f"🍭 Azúcar:   {resultado.get('azucar_g')}g (IMPORTANTE)")
            print(f"🧂 Sodio:    {resultado.get('sodio_mg', 0)}mg") # Nota: ia_service a veces guarda como sodio_g o mg, revisar
            
            if "Verificado" in resultado.get("calidad_nutricional", ""):
                print("✅ CALIDAD: Verificado con Base de Datos 🏆")
            else:
                print("⚠️ CALIDAD: Estimación IA (No encontrado en BD)")
                
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    # Ejecutar ambos tests
    asyncio.run(test_registro()) # Primero registro que es lo que acabamos de cambiar
    # asyncio.run(test_preguntas()) # Descomentar si quieres ver chat (consume tokens API)
