
import os
import sys
import json
from colorama import init, Fore, Style

# Add root directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.ia_service import ia_engine

init()

def print_header(title):
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"{title.center(60)}")
    print(f"{'='*60}{Style.RESET_ALL}")

def run_smart_log_tests():
    print_header("PRUEBAS DE LOG INTELIGENTE (SCALOFIT NLP)")

    scenarios = [
        {
            "name": "Comida Chatarra (Slang)",
            "input": "Me clavé una hamburguesa doble con queso y papas fritas medianas.",
            "expected_type": "Comida",
        },
        {
            "name": "Ejercicio Intenso",
            "input": "Corrí 30 minutos a tope en la trotadora.",
            "expected_type": "Ejercicio",
        },
        {
            "name": "Comida Saludable (Simple)",
            "input": "Solo comí una manzana verde y un puñado de almendras.",
            "expected_type": "Comida",
        },
        {
            "name": "Ambigüedad (Cafe)",
            "input": "Me tomé un café americano sin azúcar.",
            "expected_type": "Comida",
        }
    ]

    for scenario in scenarios:
        print(f"\n{Fore.YELLOW}➤ ESCENARIO: '{scenario['name']}'{Style.RESET_ALL}")
        print(f"Input: \"{scenario['input']}\"")
        
        try:
            result = ia_engine.extraer_macros_de_texto(scenario['input'])
            
            if result:
                is_food = result.get('es_comida', False)
                is_exercise = result.get('es_ejercicio', False)
                
                print(f"{Fore.GREEN}Resultado IA:{Style.RESET_ALL}")
                print(json.dumps(result, indent=2, ensure_ascii=False))

                # Verification Logic
                correct_type = False
                if scenario['expected_type'] == "Comida" and is_food and not is_exercise:
                    correct_type = True
                elif scenario['expected_type'] == "Ejercicio" and is_exercise and not is_food:
                    correct_type = True
                
                if correct_type:
                    print(f"{Fore.BLUE}✅ Tipo Correcto: {scenario['expected_type']}{Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}❌ Tipo INCORRECTO (Esperado: {scenario['expected_type']}){Style.RESET_ALL}")

                # Check for reasonable calories (non-zero)
                cals = result.get('calorias', 0)
                if cals > 0:
                     print(f"{Fore.BLUE}✅ Calorías Detectadas: {cals}{Style.RESET_ALL}")
                else:
                     print(f"{Fore.RED}❌ Calorías CERO o No Detectadas{Style.RESET_ALL}")

            else:
                print(f"{Fore.RED}❌ Error: Resultado Nulo{Style.RESET_ALL}")

        except Exception as e:
            print(f"{Fore.RED}Error Ejecución: {e}{Style.RESET_ALL}")

if __name__ == "__main__":
    run_smart_log_tests()
