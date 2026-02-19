import asyncio
import os
import sys
import json
import unittest.mock as mock

# Configurar Path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Mock de Groq para que nos de respuestas bonitas y estructuradas sin gastar tokens
async def mock_respuestas_rapidas(*args, **kwargs):
    prompt = kwargs.get('messages', [{}])[-1].get('content', '').lower()
    
    # Simular una respuesta estructurada de CaloFit
    if "cena" in prompt:
        content = """¡Claro que sí! Para tu objetivo de **Ganar Masa Muscular**, te sugiero esta cena potente pero equilibrada:

### 🍳 Cena: Tortilla de Camote y Pollo
1. **Proteína:** 150g de pechuga de pollo deshilachada.
2. **Carbohidrato:** 100g de camote asado (ideal para recuperar energía).
3. **Grasas:** Media palta (aguacate).

**Tip de CaloFit:** No le temas a los carbohidratos de noche si entrenaste hoy. ¡Tus músculos los necesitan! 🚀"""
    elif "rutina" in prompt:
        content = """¡Dale con todo! Aquí tienes tu **Rutina Express de 15 min** (Sin equipo):

*   **Min 1-3:** Calentamiento (Jumping Jacks).
*   **Min 4-8:** 3 series de Sentadillas + Flexiones de pecho (Push-ups).
*   **Min 9-13:** 3 series de Zancadas (Lunges) + Plancha abdominal.
*   **Min 14-15:** Estiramiento suave.

¡Tú puedes, Leonardo! Cada minuto cuenta. 🔥"""
    else:
        content = "¡Hola! Estoy listo para ayudarte con tu plan nutricional y de entrenamiento. ¿Qué tienes en mente hoy? 🍎"

    return type('Mock', (), {
        'choices': [type('Choice', (), {
            'message': type('Msg', (), {'content': content})
        })]
    })

# Importar y Mockear
from app.services.ia_service import ia_engine
ia_engine.groq_client = mock.AsyncMock()
ia_engine.groq_client.chat.completions.create = mock_respuestas_rapidas

async def run_test():
    print("\n" + "✨" * 15)
    print(" TEST DE ACCIONES RÁPIDAS (UX/UI)")
    print("✨" * 15 + "\n")

    perfil = "Usuario: Leonardo, Objetivo: Ganar masa muscular, Contexto: Perú"
    
    casos = [
        {"nombre": "CENA LIGERA", "pregunta": "Dame opciones de cenas bajas en calorías"},
        {"nombre": "RUTINA EXPRESS", "pregunta": "Rutina de 15 min en casa"},
        {"nombre": "CONSEJO FUZZY", "pregunta": "¿Cómo voy con mi progreso hoy?"}
    ]

    for caso in casos:
        print(f"🔹 PROBANDO ACCIÓN: {caso['nombre']}")
        print(f"👤 User: {caso['pregunta']}")
        
        # Llamar al motor de la IA
        respuesta = await ia_engine.asistir_cliente(perfil, caso['pregunta'])
        
        print(f"🤖 CALOFIT:\n{respuesta}")
        print("-" * 50)

if __name__ == "__main__":
    asyncio.run(run_test())
