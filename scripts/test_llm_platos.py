import asyncio
from app.services.ai.llm_service import LLMService

async def main():
    llm = LLMService()
    prompt = (
        "Actúa como chef nutricionista. Crea 1 plato distinto para 'cena'.\n"
        "Deben sumar aproximadamente: 350 kcal, 30g proteína, "
        "20g carbohidratos, 10g grasas.\n"
        "OBLIGATORIO: Todas las recetas DEBEN contener el ingrediente 'huevo' (como ingrediente principal o base).\n"
        "Usa ingredientes reales y comunes en Perú. No uses medidas como 'tazas' ni 'cucharadas', "
        "USA SOLO GRAMOS EXACTOS (ej. '150').\n\n"
        "Responde ÚNICAMENTE con un arreglo JSON válido con esta estructura:\n"
        "[\n"
        "  {\n"
        "    \"nombre_plato\": \"Pollo a la Plancha con Arroz y Ensalada\",\n"
        "    \"ingredientes\": [\n"
        "      {\"nombre\": \"pechuga de pollo\", \"gramos\": 150},\n"
        "      {\"nombre\": \"arroz blanco\", \"gramos\": 100},\n"
        "      {\"nombre\": \"lechuga\", \"gramos\": 50}\n"
        "    ]\n"
        "  }\n"
        "]"
    )
    res = await llm.generar_json(prompt, max_tokens=1500)
    print("LLM RESPONSE:")
    print(res)

    from app.services.nutrition.plate.plate_builder import PlateBuilderService
    from app.core.database import SessionLocal
    db = SessionLocal()
    pb = PlateBuilderService(db)
    
    for prop in res:
        nombre = prop.get("nombre_plato")
        ings = prop.get("ingredientes", [])
        print(f"\nConstruyendo: {nombre}")
        print("Ingredientes crudos:", ings)
        r = pb.construir_plato(nombre, ings, 1, "cena")
        print(f"Exito: {r.exito}, Kcal totales: {r.macros_totales.calorias if not isinstance(r.macros_totales, dict) else r.macros_totales.get('calorias')}")
        for i in r.ingredientes:
            print(f"  - {i.gramos}g {i.nombre} ({i.macros_totales.get('calorias', 0) if isinstance(i.macros_totales, dict) else (i.macros_totales.calorias if i.macros_totales else 0)} kcal) exito={i.confianza>0}")

if __name__ == "__main__":
    asyncio.run(main())
