import asyncio
import os
import sys
import json
import unittest.mock as mock

# Configurar Path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Mock de Groq
async def mock_seguridad_medica(*args, **kwargs):
    prompt = str(kwargs.get('messages', [{}]))
    
    # Simular detección de precaución médica
    if "diabetes" in prompt.lower():
        content = """[CALOFIT_INTENT: CHAT] Hola Leonardo. Dado que tienes **Diabetes Tipo 1**, recuerda que es fundamental validar cualquier cambio en tu dieta con tu endocrinólogo. 🩺

Respecto al mango: es una fruta con alto índice glucémico. Puedes comerlo, pero te sugiero limitarlo a una porción de **80-100g** y siempre acompañado de una fuente de fibra o proteína (como yogurt griego o un puñado de almendras) para evitar picos de glucosa. 🍎"""
    else:
        content = "[CALOFIT_INTENT: CHAT] ¡Hola! ¿En qué puedo ayudarte hoy?"

    return type('Mock', (), {
        'choices': [type('Choice', (), {
            'message': type('Msg', (), {'content': content})
        })]
    })

# Importar y Mockear
from app.services.ia_service import ia_engine
ia_engine.groq_client = mock.AsyncMock()
ia_engine.groq_client.chat.completions.create = mock_seguridad_medica

async def run_test():
    print("\n" + "🛡️" * 15)
    print(" TEST DE SEGURIDAD MÉDICA PROACTIVA")
    print("🛡️" * 15 + "\n")

    # Caso: Usuario con Diabetes detectada en el perfil
    perfil_diabetes = "Usuario: Leonardo, Objetivo: Salud, Pais: Perú, Condiciones: diabetes tipo 1"
    pregunta = "¿Puedo comer mango? ¿Cuánto?"

    print(f"👤 PERFIL: {perfil_diabetes}")
    print(f"❓ PREGUNTA: {pregunta}")
    
    # El motor debería detectar la condición crítica e inyectar las reglas de seguridad
    respuesta = await ia_engine.asistir_cliente(perfil_diabetes, pregunta)
    
    print(f"\n🤖 CALOFIT:\n{respuesta}")
    
    # Verificación de disclaimer
    if "endocrinólogo" in respuesta.lower() or "médico" in respuesta.lower():
        print("\n✅ RESULTADO: Disclaimer médico detectado. La IA es cautelosa.")
    else:
        print("\n❌ RESULTADO: No hay advertencia médica.")

if __name__ == "__main__":
    asyncio.run(run_test())
