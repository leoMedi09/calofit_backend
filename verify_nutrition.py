from app.services.ia_service import ia_engine
from datetime import datetime

def test_nutrition_logic():
    print("🧪 Testing IAService Nutrition Logic...\n")
    
    test_cases = [
        {
            "name": "Standard Bulk (80kg Male)",
            "data": {
                "genero": "M",
                "edad": 30,
                "peso": 80.0,
                "talla": 180.0,
                "nivel_actividad": "Moderado",
                "objetivo": "Ganar masa (Volumen)",
                "condiciones_medicas": ""
            }
        },
        {
            "name": "Aggressive Deficit with Diabetes (60kg Female)",
            "data": {
                "genero": "F",
                "edad": 45,
                "peso": 60.0,
                "talla": 160.0,
                "nivel_actividad": "Sedentario",
                "objetivo": "Perder peso (Agresivo)",
                "condiciones_medicas": "Diabetes"
            }
        },
        {
            "name": "Lean Bulk with Injury (75kg Male)",
            "data": {
                "genero": "M",
                "edad": 25,
                "peso": 75.0,
                "talla": 175.0,
                "nivel_actividad": "Intenso",
                "objetivo": "Ganar masa (Limpio)",
                "condiciones_medicas": "Lesion lumbar"
            }
        }
    ]
    
    for case in test_cases:
        print(f"--- Case: {case['name']} ---")
        plan = ia_engine.generar_plan_inicial_automatico(case['data'])
        if plan:
            print(f"Calories: {plan['calorias_diarias']} kcal")
            print(f"Macros: P={plan['macros']['P']}g, C={plan['macros']['C']}g, G={plan['macros']['G']}g")
            print(f"Alert: {plan['dias'][0]['nota_asistente_ia']}")
            print(f"Security Alert: {plan['alerta_seguridad']}")
            
            # Verification check: P*4 + C*4 + G*9 approx calories
            calculated_cals = (plan['macros']['P']*4) + (plan['macros']['C']*4) + (plan['macros']['G']*9)
            diff = abs(calculated_cals - plan['calorias_diarias'])
            print(f"Verification Check: {calculated_cals:.1f} kcal (Diff: {diff:.1f})")
            print("✅ Verified" if diff < 5 else "❌ Verification Failed")
        else:
            print("❌ Plan generation failed")
        print("\n")

if __name__ == "__main__":
    test_nutrition_logic()
