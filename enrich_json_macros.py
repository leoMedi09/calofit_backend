
import json
import re

def estimar_fibra_azucar(item):
    nombre = item.get("alimento", "").lower()
    carbos = item.get("carbohindratos_100g", 0)
    grasas = item.get("grasas_100g", 0)
    proteina = item.get("proteina_100g", 0)
    calorias = item.get("calorias_100g", 0)
    
    fibra = 0.0
    azucar = 0.0
    
    # 1. Lógica para Bebidas Azucaradas (Gaseosas, Jugos)
    if any(x in nombre for x in ["coca cola", "inca kola", "gaseosa", "frugos", "bebida", "sporade", "powerade", "pepsi", "sprite", "fanta", "néctar", "jugo"]):
        fibra = 0.0
        azucar = carbos * 0.95 # Casi todo el carbo es azúcar
        
    # 2. Lógica para Chocolates y Golosinas
    elif any(x in nombre for x in ["chocolate", "galleta", "oreo", "casino", "morochas", "doña pepa", "sublime", "triángulo", "princesa", "golosina", "caramelo", "gomita", "marshmallow", "turrón"]):
        fibra = carbos * 0.05 # Poca fibra
        azucar = carbos * 0.60 # 60% azúcar aprox
        if "bitter" in nombre or "negro" in nombre or "dark" in nombre:
            fibra = carbos * 0.15 # Más fibra en chocolate oscuro
            azucar = carbos * 0.40
            
    # 3. Lógica para Cereales y Panes
    elif any(x in nombre for x in ["pan", "tostada", "galleta soda", "avena", "cereal", "granola", "barrita", "trigo", "arroz", "fideos", "pasta"]):
        if "integral" in nombre or "avena" in nombre or "granola" in nombre:
            fibra = carbos * 0.15 # Alto en fibra
            azucar = carbos * 0.10 # Poco azúcar (salvo granola)
            if "granola" in nombre:
                azucar = carbos * 0.30
        else: # Refinados
            fibra = carbos * 0.03 # Muy poca fibra
            azucar = carbos * 0.05
            
    # 4. Lógica para Lácteos
    elif any(x in nombre for x in ["leche", "yogurt", "queso", "mantequilla", "margarina"]):
        fibra = 0.0
        if "yogurt" in nombre or "bebible" in nombre:
            azucar = carbos * 0.80 # Lactosa + Azúcar añadido
        else:
            azucar = carbos # Lactosa es azúcar simple
            
    # 5. Lógica para Frutas y Verduras (Si hubiera en envasados)
    elif any(x in nombre for x in ["fruta", "verdura", "ensalada", "conserva"]):
        fibra = carbos * 0.20
        azucar = carbos * 0.50
        
    # 6. Lógica para Proteínas (Atún, Pollo, Carne)
    elif any(x in nombre for x in ["atún", "pollo", "carne", "pavo", "huevo", "conserva de pescado"]):
        fibra = 0.0
        azucar = 0.0
        
    # 7. Lógica para Grasas (Aceite)
    elif "aceite" in nombre:
        fibra = 0.0
        azucar = 0.0

    # Ajustes finales de coherencia
    if azucar > carbos: azucar = carbos
    if fibra > carbos: fibra = carbos * 0.5 # Fibra no suele ser > 50% carbos salvo excepciones
    
    # Redondear
    item["fibra_100g"] = round(fibra, 2)
    item["azucar_100g"] = round(azucar, 2)
    return item

# Procesar archivo
rutas = [
    "app/data/alimentos_peru_off.json",
    "app/data/alimentos_peru_ins.json"
]

for ruta in rutas:
    try:
        with open(ruta, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        print(f"Procesando {ruta} ({len(data)} items)...")
        
        nuevos_data = [estimar_fibra_azucar(item) for item in data]
        
        with open(ruta, "w", encoding="utf-8") as f:
            json.dump(nuevos_data, f, indent=4, ensure_ascii=False)
            
        print(f"✅ {ruta} actualizado con estimaciones inteligentes.")
        
    except Exception as e:
        print(f"❌ Error en {ruta}: {e}")
