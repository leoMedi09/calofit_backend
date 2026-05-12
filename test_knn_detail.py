import warnings, logging
warnings.filterwarnings('ignore')
logging.disable(logging.CRITICAL)

from app.core.database import SessionLocal
from app.models.client import Client
from app.services.ml_service import ml_recomendador

db = SessionLocal()
try:
    # Test KNN directo — vector de deficit real
    print("=== KNN directo (deficit 2000kcal / 150g prot / 220g carb / 60g grasa) ===")
    recs = ml_recomendador.obtener_recomendaciones(
        calorias_faltantes=2000, prote_faltante=150,
        carbo_faltante=220, grasa_faltante=60,
        n_recomendaciones=5, excluir_nombres=[]
    )
    print("Keys disponibles en resultado:", list(recs[0].keys()) if recs else "SIN RESULTADOS")
    for r in recs:
        print(" ", r)

    print()
    print("=== KNN deficit alto proteína (1200kcal / 80g prot / 100g carb / 30g grasa) ===")
    recs2 = ml_recomendador.obtener_recomendaciones(
        calorias_faltantes=1200, prote_faltante=80,
        carbo_faltante=100, grasa_faltante=30,
        n_recomendaciones=5, excluir_nombres=[]
    )
    for r in recs2:
        print(" ", r)

    print()
    print("=== KNN deficit bajo (400kcal / 30g prot / 50g carb / 10g grasa) ===")
    recs3 = ml_recomendador.obtener_recomendaciones(
        calorias_faltantes=400, prote_faltante=30,
        carbo_faltante=50, grasa_faltante=10,
        n_recomendaciones=5, excluir_nombres=[]
    )
    for r in recs3:
        print(" ", r)

finally:
    db.close()
