import asyncio
from app.core.database import SessionLocal
from app.services.recomendador_platos import RecomendadorPlatosConfiables
from app.services.nutrition.plate.plate_builder import PlatoBuilder
from app.services.nutrition.food.resolver.cache_manager import CacheManager
from app.services.nutrition.food.resolver.source_resolver import FoodSourceResolver

async def main():
    db = SessionLocal()
    cache = CacheManager(db)
    resolver = FoodSourceResolver(db, cache)
    pb = PlatoBuilder(db, resolver, cache)
    rec = RecomendadorPlatosConfiables(db, plate_builder=pb)
    
    platos = rec.recomendar(
        client_id=1,
        deficit_kcal=1200,
        deficit_proteina=80,
        deficit_carb=150,
        deficit_grasas=30,
        momento_dia='almuerzo',
        n=3,
        ingrediente_clave="pescado",
        excluir_nombres=['Plátano Con Yogur', 'Pollo Al Horno Con Plátano', 'Arroz Integral Con Lentejas Y Verduras']
    )
    
    print(f'Platos recomendados: {len(platos)}')
    for p in platos:
        m = p['macros']
        print(f'  [{p.get("fuente", "?")}] {p.get("nombre", "?")}')
        print(f'    Kcal: {m.get("calorias", 0)} | P: {m.get("proteinas_g", 0)}g | C: {m.get("carbohidratos_g", 0)}g | G: {m.get("grasas_g", 0)}g')
    db.close()

asyncio.run(main())
