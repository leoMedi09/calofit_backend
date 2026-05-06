from app.core.database import SessionLocal
from app.services.recomendador_platos import RecomendadorPlatosConfiables

db = SessionLocal()
rec = RecomendadorPlatosConfiables(db)

platos = rec.recomendar(
    client_id=1,
    deficit_kcal=600,
    deficit_proteina=30,
    deficit_carb=80,
    deficit_grasas=15,
    momento_dia='almuerzo',
    n=3,
)

print(f'Platos recomendados: {len(platos)}')
print()
for p in platos:
    m = p['macros']
    fuente = p['fuente']
    nombre = p['nombre']
    print(f'  [{fuente}] {nombre}')
    print(f'    Kcal: {m["calorias"]} | P: {m["proteinas_g"]}g | C: {m["carbohidratos_g"]}g | G: {m["grasas_g"]}g')
    print(f'    Confianza: {p["confianza"]}% | Score: {p["score"]:.1f}')
    print()

db.close()
