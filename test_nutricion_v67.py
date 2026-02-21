import asyncio
import sys
import os

# Añadir el directorio actual al path para importar app
sys.path.append(os.getcwd())

from app.services.ia_service import IAService
from app.services.nutricion_service import NutricionService

async def test_cases():
    ia = IAService()
    nut = NutricionService()
    ia.nutricion_service = nut
    
    print("\n" + "="*50)
    print("TEST V67.2: VALIDACIÓN DE NUTRICIÓN")
    print("="*50)
    
    casos = [
        {
            "nombre": "FRACCIONES (Error reportado '1/2')",
            "texto": "[CALOFIT_HEADER: AVENA CON CHICHAS]\n[CALOFIT_STATS: 100kcal, P: 5g, C: 20g, G: 0g]\n[CALOFIT_LIST]\n* 1/2 taza de avena\n* 1/4 taza de leche evaporada\n[CALOFIT_ACTION: ADD]"
        },
        {
            "nombre": "CHAUFA VEGANO (Plato Complejo)",
            "texto": "[CALOFIT_HEADER: CHAUFA VEGANO]\n[CALOFIT_STATS: 400kcal, P: 15g, C: 60g, G: 10g]\n[CALOFIT_LIST]\n* 1 taza de arroz cocido\n* 50 ml de tofu crujiente\n* 1 cucharada de aceite de ajonjolí\n* 1/2 taza de cebollita china\n[CALOFIT_ACTION: ADD]"
        },
        {
            "nombre": "AJÍ DE GALLINA (Plato Tradicional)",
            "texto": "[CALOFIT_HEADER: AJÍ DE GALLINA]\n[CALOFIT_STATS: 500kcal, P: 30g, C: 40g, G: 20g]\n[CALOFIT_LIST]\n* 150g de pechuga de pollo deshilachada\n* 1/2 taza de crema de ají amarillo\n* 1 papa amarilla cocida\n* 1/4 taza de arroz blanco\n[CALOFIT_ACTION: ADD]"
        },
        {
            "nombre": "CENA BAJA EN CALORÍAS (Cena Light)",
            "texto": "[CALOFIT_HEADER: CENA LIGERA]\n[CALOFIT_STATS: 200kcal, P: 20g, C: 10g, G: 5g]\n[CALOFIT_LIST]\n* 100g de pechuga de pavo\n* 1 taza de calabacín picado\n* 1 cucharadita de aceite de oliva\n* Sal y pimienta al gusto\n[CALOFIT_ACTION: ADD]"
        }
    ]
    
    for caso in casos:
        print(f"\n--- Probando: {caso['nombre']} ---")
        resultado = ia.validar_y_corregir_nutricion(caso['texto'])
        print(resultado)
        
        # Verificar si hay (0 kcal) en ingredientes que no deberían
        if "(0 kcal)" in resultado and "sal" not in resultado.lower() and "pimienta" not in resultado.lower() and "agua" not in resultado.lower():
            print("⚠️ ADVERTENCIA: Se detectó 0 kcal en un ingrediente nutricional.")
        else:
            print("✅ OK: Ingredientes procesados correctamente.")

if __name__ == "__main__":
    asyncio.run(test_cases())
