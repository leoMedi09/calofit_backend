import warnings, logging
warnings.filterwarnings('ignore')
logging.disable(logging.CRITICAL)

from app.core.database import SessionLocal
from app.models.client import Client
from app.services.asistente.asistente_recomendaciones import recomendaciones_handler

db = SessionLocal()
try:
    print("=" * 65)
    print("RF - Prediccion de perfil (15 usuarios de prueba)")
    print("=" * 65)
    clientes = db.query(Client).filter(Client.id >= 55, Client.id <= 69).order_by(Client.id).all()
    correctos = 0
    for c in clientes:
        perfil, conf = recomendaciones_handler.predecir_perfil(c, db)
        f = recomendaciones_handler.preparar_features_rf(c, db)
        esperado = 'PERFIL_A' if c.id <= 59 else ('PERFIL_B' if c.id <= 64 else 'PERFIL_C')
        ok = "OK" if perfil == esperado else "FAIL"
        if ok == "OK": correctos += 1
        print("%s | ID %d | pred=%s(%.0f%%) esp=%s | freq=%d/sem | session=%.1fh | wt=%s" % (
            ok, c.id, perfil, conf, esperado,
            f['workout_freq'], f['session_hours'], f['workout_type']
        ))
    print()
    print("Precision RF: %d/15 (%.0f%%)" % (correctos, correctos/15*100))

    print()
    print("=" * 65)
    print("KNN - Recomendaciones con 3 vectores de deficit distintos")
    print("=" * 65)
    casos = [
        ("Alto  (2000kcal/150P/220C/60G)", 2000, 150, 220, 60),
        ("Medio (700kcal/50P/90C/20G)",     700,  50,  90, 20),
        ("Bajo  (300kcal/25P/40C/8G)",      300,  25,  40,  8),
    ]
    for label, kcal, p, c_, g in casos:
        recs = recomendaciones_handler.obtener_recomendaciones_knn.__wrapped__ if hasattr(recomendaciones_handler.obtener_recomendaciones_knn, '__wrapped__') else None
        from app.services.ml_service import ml_recomendador
        recs = ml_recomendador.obtener_recomendaciones(kcal, p, c_, g, 3, [])
        print("Deficit %s:" % label)
        for r in recs:
            print("  %-45s kcal=%.0f P=%.0fg sim=%.0f%%" % (
                r['alimento'][:45], r['calorias_100g'], r['proteina_100g'], r['similitud']
            ))
finally:
    db.close()
