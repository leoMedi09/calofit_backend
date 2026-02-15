
import sys
import os

# Ajustar path
sys.path.append(os.getcwd())

from app.services.ia_service import ia_engine
from app.services.response_parser import parsear_respuesta_para_frontend
import json

# SIMULACIÓN DE CONTEXTO GENERADO POR EL BACKEND (Sin intervención manual)
contexto_simulado = (
    "Eres el coach experto de Leonardo. "
    "- Perfil: 70kg, Objetivo: Definir. "
    "\n🚨 RESTRICCIONES CRÍTICAS: "
    "- DIETA: Vegano (PROHIBIDO: carne, pollo, pescado, lácteos, huevos). "
    "- ALERGIAS: Maní y Mariscos (PROHIBIDO: cualquier traza). "
    "\nSTATUS: Le quedan 600 kcal para cenar. "
    "\nREGLA: Sugiere algo PERUANO VEGANO."
)

mensaje = "Hola, recomiéndame una cena peruana rápida para hoy."

print("🤖 Generando respuesta para usuario Vegano/Alérgico...")
respuesta = ia_engine.asistir_cliente(
    contexto=contexto_simulado,
    mensaje_usuario=mensaje,
    tono_aplicado="empático pero directo"
)

# Aplicar el parser que corregimos hoy
estructurada = parsear_respuesta_para_frontend(respuesta)

print("\n--- RESPUESTA ESTRUCTURADA (JSON) ---")
print(json.dumps(estructurada, indent=4, ensure_ascii=False))
