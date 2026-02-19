import sys
import os
import json

# Añadir el directorio raíz al path para poder importar 'app'
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from app.services.nutricion_service import nutricion_service

def imprimir_resultado(titulo, data):
    print(f"\n🔍 PRUEBA: {titulo}")
    if not data:
        print("❌ Error: No se encontraron datos.")
        return
    
    print(f"✅ Encontrado: {data.get('alimento')} ({data.get('detalle', '')})")
    print(f"🌍 Origen: {data.get('origen', 'Desconocido')}")
    
    # Verificar Macros Clásicos
    print("📊 Macros Básicos:")
    print(f"   - Calorías: {data.get('calorias_100g')} kcal")
    print(f"   - Proteína: {data.get('proteina_100g')} g")
    print(f"   - Grasas: {data.get('grasas_100g')} g")
    print(f"   - Carbos: {data.get('carbohindratos_100g')} g")
    
    # Verificar Nuevos Micros (Si existen)
    print("🔬 Micros & Detalles (NUEVO):")
    nuevos_campos = ['azucar_100g', 'fibra_100g', 'sodio_mg_100g', 'grasas_saturada_100g', 'vitamina_c_mg_100g']
    encontrados = 0
    for campo in nuevos_campos:
        val = data.get(campo)
        if val is not None:
            print(f"   - {campo}: {val}")
            encontrados += 1
    
    if encontrados > 0:
        print("✨ ¡ÉXITO! Se están recibiendo los nuevos datos.")
    else:
        print("⚠️ ALERTA: No se ven los campos nuevos en este producto.")

if __name__ == "__main__":
    print("🛠️ Iniciando Test de Integración Nutricional...")
    
    # 1. Prueba Local (Debe ser rápida y venir del JSON)
    data_local = nutricion_service.obtener_info_alimento("Manzana")
    imprimir_resultado("Producto Local (Manzana)", data_local)
    
    # 2. Prueba Mundial (Debe venir de SQLite)
    # Buscamos algo muy específico internacional
    data_mundial = nutricion_service.obtener_info_alimento("Nutella")
    imprimir_resultado("Producto Mundial (Nutella)", data_mundial)
