
import json
import requests
import time

def fetch_real_data():
    ruta = "app/data/alimentos_peru_off.json"
    
    try:
        with open(ruta, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        print(f"Iniciando validación de {len(data)} productos con OpenFoodFacts API...")
        
        actualizados = 0
        errores = 0
        
        # Limitamos a los primeros 20 para la demostración rápida
        # En producción se quitaría el [:20]
        for i, item in enumerate(data): 
            barcode = item.get("id_externo")
            if not barcode:
                continue
                
            url = f"https://world.openfoodfacts.org/api/v0/product/{barcode}.json"
            try:
                # Respetamos rate limits
                time.sleep(0.5) 
                
                resp = requests.get(url, timeout=5)
                if resp.status_code == 200:
                    prod_data = resp.json()
                    if prod_data.get("status") == 1:
                        nutriments = prod_data["product"].get("nutriments", {})
                        
                        # Extraer datos reales
                        real_sugar = nutriments.get("sugars_100g")
                        real_fiber = nutriments.get("fiber_100g")
                        
                        cambios = []
                        if real_sugar is not None:
                            item["azucar_100g"] = float(real_sugar)
                            item["fuente_azucar"] = "Verificado (OpenFoodFacts)"
                            cambios.append(f"Azúcar: {real_sugar}")
                            
                        if real_fiber is not None:
                            item["fibra_100g"] = float(real_fiber)
                            item["fuente_fibra"] = "Verificado (OpenFoodFacts)"
                            cambios.append(f"Fibra: {real_fiber}")
                            
                        if cambios:
                            actualizados += 1
                            print(f"✅ {item['alimento']}: {', '.join(cambios)}")
                        else:
                            print(f"⚠️ {item['alimento']}: Sin datos nuevos en API.")
                    else:
                        print(f"❌ {item['alimento']}: No encontrado en OFF.")
                else:
                    print(f"❌ Error HTTP {resp.status_code}")
                    
            except Exception as e:
                print(f"Error conectando: {e}")
                errores += 1
                
            if i % 10 == 0:
                # Guardado parcial
                with open(ruta, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)

        # Guardado final
        with open(ruta, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
            
        print(f"\nResumen: {actualizados} productos actualizados con datos reales.")
        
    except Exception as e:
        print(f"Error general: {e}")

if __name__ == "__main__":
    fetch_real_data()
