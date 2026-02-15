
import sys
import os
sys.path.append(os.getcwd())
from app.services.ia_service import ia_engine
from app.services.response_parser import parsear_respuesta_para_frontend
import json

contexto_clinico = (
    "Eres el coach clínico de Leonardo. DATOS: 95kg. "
    "\n🚨 RESTRICCIONES MÉDICAS: "
    "- Diabetes Tipo 2 y Hipertensión (0 azúcar, BAJÍSIMO SODIO). "
    "- Dieta Cetogénica (Keto - Prohibido arroz, papa, camote, choclo, legumbres). "
    "- Alergia a Frutos Secos (No nueces/almendras). "
    "\nSTATUS: Máximo 400 kcal para la cena."
)

mensaje = "Dame una cena peruana keto, sin sal, sin azúcar y sin nueces."

print("🧠 Procesando Consulta Clínica Extrema...")
respuesta = ia_engine.asistir_cliente(
    contexto=contexto_clinico,
    mensaje_usuario=mensaje,
    tono_aplicado="directo y empático"
)

estructurada = parsear_respuesta_para_frontend(respuesta)
print("\n--- RESULTADO CLÍNICO (JSON) ---")
print(json.dumps(estructurada, indent=4, ensure_ascii=False))
