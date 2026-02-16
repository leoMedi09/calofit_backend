
import sys
import os
import json
import re
from colorama import init, Fore, Style

# Añadir el path para importar app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.ia_service import ia_engine

init()

def print_test_header(title):
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"{title.center(60)}")
    print(f"{'='*60}{Style.RESET_ALL}")

def run_scenario(name, user_input, context=""):
    print_test_header(f"ESCENARIO: {name}")
    print(f"{Fore.YELLOW}INPUT:{Style.RESET_ALL} {user_input}")
    
    # Mock de contexto si no se provee
    if not context:
        context = "Perfil: Juan, 28 años, Masculino, Peso: 80kg, Altura: 175cm, Objetivo: Perder peso, Actividad: Moderado, Condiciones: Ninguna."
    
    try:
        # Usamos asistir_cliente para simular el flujo real
        respuesta = ia_engine.asistir_cliente(contexto=context, mensaje_usuario=user_input)
        
        print(f"\n{Fore.GREEN}RESPUESTA PROCESADA:{Style.RESET_ALL}\n")
        print(respuesta)
        
        # Validaciones de criterio
        validations = {
            "Tiene Intent Tag": bool(re.search(r'\[CALOFIT_INTENT:.*?\]', respuesta)),
            "Tiene Header Blindado": "[CALOFIT_HEADER]" in respuesta,
            "Tiene Stats Reales": "P: " in respuesta and "Cal: " in respuesta,
            "Lógica de Cocción (si aplica)": "(Aceite incluido)" in respuesta if "frito" in user_input.lower() else "N/A",
            "Conocimiento Peruano": "Plátano" in respuesta if "tacacho" in user_input.lower() else "N/A"
        }
        
        print(f"\n{Fore.BLUE}VERIFICACIÓN DE CRITERIOS:{Style.RESET_ALL}")
        for k, v in validations.items():
            color = Fore.GREEN if v is True or v == "N/A" else Fore.RED
            print(f"- {k}: {color}{v}{Style.RESET_ALL}")
            
    except Exception as e:
        print(f"{Fore.RED}ERROR EN ESCENARIO:{Style.RESET_ALL} {e}")

def main():
    # 1. Chat General
    run_scenario("Chat/Saludo", "Hola, ¿quién eres y qué puedes hacer por mí?")
    
    # 2. Receta Compleja (Lógica de Cocción + Cultural)
    run_scenario("Receta Selva (Lógica + Frito)", "¿Me puedes dar una receta de Tacacho con Cecina pero que la cecina sea frita?")
    
    # 3. Salud y Advertencia
    run_scenario("Alerta de Salud", "Me duele mucho el pecho cuando hago ejercicio y estoy mareado.")
    
    # 4. Multi-Sección (Entrenamiento + Nutrición)
    run_scenario("Multi-Sección", "Quiero una rutina para mis tríceps y algo ligero para cenar que tenga pollo.")
    
    # 5. Escudo Calórico (Límite estricto)
    run_scenario("Escudo Calórico", "Dame una cena con pollo pero que no pase de 300 calorías, sé estricto con las porciones.")

if __name__ == "__main__":
    main()
