import sys
sys.path.insert(0, '/app')

from app.services.response_parser import parsear_respuesta_para_frontend

# Simular una respuesta de la IA
texto_ia = """
¡Hola! Aquí tienes una rutina de 15 minutos para hacer en casa:

[CALOFIT_INTENT: ITEM_WORKOUT]
[CALOFIT_HEADER]RUTINA DE 15 MINUTOS[/CALOFIT_HEADER]
[CALOFIT_STATS]Cal: 160kcal[/CALOFIT_STATS]
[CALOFIT_LIST]
- 3 series x 15 Sentadillas
- 3 series x 10 Flexiones
- 2 series x 30seg Plancha
[/CALOFIT_LIST]
[CALOFIT_ACTION]
1. Sentadillas: Pies al ancho de hombros, baja hasta 90°
2. Flexiones: Manos al ancho de hombros, baja el pecho
3. Plancha: Mantén el cuerpo recto, no bajes la cadera
[/CALOFIT_ACTION]
"""

resultado = parsear_respuesta_para_frontend(texto_ia)

print("=" * 80)
print("TEXTO CONVERSACIONAL:")
print(resultado["texto_conversacional"])
print("\n" + "=" * 80)
print(f"INTENT: {resultado['intent']}")
print(f"NÚMERO DE SECCIONES: {len(resultado['secciones'])}")
print("\n" + "=" * 80)

for i, sec in enumerate(resultado["secciones"]):
    print(f"\nSECCIÓN {i+1}:")
    print(f"  Tipo: {sec['tipo']}")
    print(f"  Nombre: {sec['nombre']}")
    print(f"  Ejercicios: {sec.get('ejercicios', [])}")
    print(f"  Instrucciones: {sec.get('instrucciones', [])}")
    print(f"  Macros: {sec.get('macros', 'N/A')}")
