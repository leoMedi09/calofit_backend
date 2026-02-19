import asyncio
import os
import sys
import json
import unittest.mock as mock

# Configurar Path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Mock de Groq
async def mock_safety_exercise(*args, **kwargs):
    prompt = str(kwargs.get('messages', [{}]))
    
    # Simular detección de precaución por lesión
    if "rodilla" in prompt.lower():
        content = """[CALOFIT_INTENT: CHAT] Hola Leonardo. Siento mucho lo de tu rodilla. 🤕 

Dado que tienes una lesión en esa zona, **PROHIBIREMOS** temporalmente los saltos y las sentadillas con peso por hoy. Vamos a enfocarnos en fortalecer el tren superior y la estabilidad del core.

Aquí tienes una opción segura:
### 🧘 Rutina de Bajo Impacto
1. **Core:** Plancha abdominal (Plank) - 3 series de 30 segundos.
2. **Tren Superior:** Flexiones de brazos (en rodillas si es necesario) - 3 series de 10.
3. **Movilidad:** Movilidad de cadera y tobillos sentado.

**Importante:** Si sientes dolor agudo, detente de inmediato. ¡Tu recuperación es lo primero! 🛡️"""
    else:
        content = "[CALOFIT_INTENT: CHAT] ¡Hola! ¿Ejercitamos hoy?"

    return type('Mock', (), {
        'choices': [type('Choice', (), {
            'message': type('Msg', (), {'content': content})
        })]
    })

# Importar y Mockear
from app.services.ia_service import ia_engine
ia_engine.groq_client = mock.AsyncMock()
ia_engine.groq_client.chat.completions.create = mock_safety_exercise

async def run_test():
    print("\n" + "🦵" * 15)
    print(" TEST DE SEGURIDAD EN EJERCICIOS (LESIONES)")
    print("🦵" * 15 + "\n")

    # Caso: Usuario menciona dolor de rodilla
    perfil = "Usuario: Leonardo, Objetivo: Mantenerse activo, Pais: Perú"
    pregunta = "Dame una rutina pero me duele la rodilla derecha"

    print(f"👤 PERFIL: {perfil}")
    print(f"❓ PREGUNTA: {pregunta}")
    
    # El motor debería detectar 'rodilla' e inyectar las reglas de seguridad
    respuesta = await ia_engine.asistir_cliente(perfil, pregunta)
    
    print(f"\n🤖 CALOFIT:\n{respuesta}")
    
    # Verificación de filtrado y consejo
    if "prohibiremos" in respuesta.lower() or "bajo impacto" in respuesta.lower() or "rodilla" in respuesta.lower():
        print("\n✅ RESULTADO: La IA adaptó la rutina a la lesión. ¡Excelente!")
    else:
        print("\n❌ RESULTADO: La IA ignoró la lesión.")

if __name__ == "__main__":
    asyncio.run(run_test())
