import sys
import os

# Ensure the root directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from app.services.ia_service import ia_engine

import asyncio

async def test_extraction():
    texts = [
        "Corrí 30 minutos",
        "Caminé 1 hora",
        "Hice pesas 45 min"
    ]

    print("--- Testing Macro Extraction ---")
    for text in texts:
        print(f"\nInput: '{text}'")
        try:
            result = await ia_engine.extraer_macros_de_texto(text)
            print("Result:", result)
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_extraction())
