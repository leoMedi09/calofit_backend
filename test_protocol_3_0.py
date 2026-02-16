
import os
import sys
import json
import re
from colorama import init, Fore, Style

# Añadir el directorio raíz al path para importar los servicios
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.ia_service import ia_engine
from app.services.response_parser import parsear_respuesta_para_frontend

init()

def print_banner(text):
    print(f"\n{Fore.MAGENTA}{'='*80}")
    print(f"{text.center(80)}")
    print(f"{'='*80}{Style.RESET_ALL}\n")

def test_protocol_zero_failure():
    print_banner("DEBUG DE PROTOCOLO CALOFIT V3.0")
    
    # Simular contexto de usuario
    contexto = """
    Perfil: Juan | Objetivo: Masa Muscular | Peso: 80.0 kg | Talla: 175 cm | Edad: 25
    Consumidas: 1200 kcal | Meta: 2800 kcal | Restantes: 1600 kcal
    Preferencia: Comida peruana (Tacacho, Cecina, Arroz), entrenamiento en casa.
    """
    
    mensaje_usuario = "Hola Coach! Recomiéndame un almuerzo: Tacacho con Cecina. Y una rutina de brazos."
    
    print(f"{Fore.CYAN}Mensaje del Usuario:{Style.RESET_ALL} {mensaje_usuario}")
    print(f"{Fore.YELLOW}Iniciando motor de IA...{Style.RESET_ALL}")
    
    try:
        # 1. Obtener respuesta de la IA
        respuesta_ia = ia_engine.asistir_cliente(contexto=contexto, mensaje_usuario=mensaje_usuario)
        
        # Guardar para inspección profunda
        with open("last_test_response.txt", "w", encoding="utf-8") as f:
            f.write(respuesta_ia)
            
        print(f"\n{Fore.GREEN}--- RESPUESTA OBTENIDA (Primeros 300 caps) ---{Style.RESET_ALL}")
        print(respuesta_ia[:300] + "...")
        
        # 2. Análisis de Etiquetas
        print_banner("ANÁLISIS DE ETIQUETAS")
        
        tags_requeridos = [
            "[CALOFIT_INTENT:",
            "[CALOFIT_HEADER]",
            "[CALOFIT_STATS]",
            "[CALOFIT_LIST]",
            "[CALOFIT_ACTION]"
        ]
        
        all_ok = True
        for tag in tags_requeridos:
            if tag in respuesta_ia:
                print(f"{Fore.GREEN}✓ {tag} presente{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}✗ {tag} AUSENTE{Style.RESET_ALL}")
                all_ok = False
        
        # 3. Verificar Inyección de Macros
        stats_match = re.search(r'\[CALOFIT_STATS\](.*?)\[/CALOFIT_STATS\]', respuesta_ia)
        if stats_match:
            stats_content = stats_match.group(1).strip()
            print(f"\n{Fore.CYAN}Macros Detectados:{Style.RESET_ALL} {stats_content}")
            if "0.0g" in stats_content and "Cal: 0kcal" in stats_content:
                print(f"{Fore.RED}⚠ ALERTA: Los macros están en cero. Posible fallo en detección de ingredientes/ejercicios.{Style.RESET_ALL}")
                all_ok = False
            else:
                print(f"{Fore.GREEN}✓ Macros inyectados con valores reales.{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}✗ Error crítico: No se encontró bloque de STATS en la respuesta final.{Style.RESET_ALL}")
            all_ok = False

        # 4. Probar el Parser
        print_banner("PRUEBA DEL PARSER")
        resultado = parsear_respuesta_para_frontend(respuesta_ia, mensaje_usuario)
        
        print(f"Intent Resultante: {resultado.get('intent')}")
        print(f"Número de Secciones: {len(resultado.get('secciones', []))}")
        
        if not resultado.get('secciones'):
            print(f"{Fore.RED}✗ El parser no detectó ninguna sección estructurada.{Style.RESET_ALL}")
            all_ok = False
        else:
            for s in resultado['secciones']:
                print(f" - {s['tipo'].upper()}: {s['nombre']}")

        if all_ok:
            print(f"\n{Fore.GREEN}🔥 TEST EXITOSO: El sistema está operando correctamente con el Protocolo 3.0.{Style.RESET_ALL}")
        else:
            print(f"\n{Fore.RED}❌ TEST FALLIDO: Hay problemas de integridad o validación.{Style.RESET_ALL}")

    except Exception as e:
        print(f"\n{Fore.RED}Error: {e}{Style.RESET_ALL}")

if __name__ == "__main__":
    test_protocol_zero_failure()
