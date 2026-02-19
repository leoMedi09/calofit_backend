import requests
import json
import os

def descargar_productos_peru():
    url = "https://world.openfoodfacts.org/cgi/search.pl"
    
    # Parámetros para buscar productos de Perú
    params = {
        "action": "process",
        "tagtype_0": "countries",
        "tag_contains_0": "contains",
        "tag_0": "peru",
        "json": "1",
        "page_size": 1000,  # Intentar traer 1000 de una
        "fields": "product_name,brands,nutriments,id,image_url"
    }
    
    print("🇵🇪 Iniciando búsqueda de productos peruanos en OpenFoodFacts...")
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        productos_raw = data.get("products", [])
        print(f"✅ Se encontraron {len(productos_raw)} productos crudos.")
        
        productos_limpios = []
        
        for p in productos_raw:
            # Filtro de Calidad: Solo productos con nombre y calorías
            nombre = p.get("product_name", "").strip()
            nutris = p.get("nutriments", {})
            marcas = p.get("brands", "").strip()
            
            calorias = nutris.get("energy-kcal_100g")
            proteina = nutris.get("proteins_100g")
            
            # Solo guardar si tienen datos nutricionales mínimos
            if nombre and calorias is not None:
                # Limpieza de nombre (ej: quitar marcas repetidas)
                nombre_completo = f"{nombre} - {marcas}" if marcas else nombre
                
                producto_clean = {
                    "alimento": nombre_completo,
                    "calorias_100g": float(calorias),
                    "proteina_100g": float(proteina) if proteina else 0.0,
                    "carbohindratos_100g": float(nutris.get("carbohydrates_100g", 0.0)),
                    "grasas_100g": float(nutris.get("fat_100g", 0.0)),
                    "origen": "OpenFoodFacts-PE",
                    "id_externo": p.get("id")
                }
                productos_limpios.append(producto_clean)
        
        print(f"✨ Filtrado completado: {len(productos_limpios)} productos nutricionalmente válidos.")
        
        # Guardar en archivo
        output_path = os.path.join(os.path.dirname(__file__), "..", "app", "data", "alimentos_peru_off.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(productos_limpios, f, indent=4, ensure_ascii=False)
            
        print(f"💾 Guardado exitoso en: {output_path}")
        
    except Exception as e:
        print(f"❌ Error al descargar: {e}")

if __name__ == "__main__":
    descargar_productos_peru()
