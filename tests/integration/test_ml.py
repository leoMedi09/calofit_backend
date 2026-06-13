import warnings, logging
warnings.filterwarnings('ignore')
logging.disable(logging.CRITICAL)

from app.core.database import SessionLocal
from app.models.client import Client
from app.services.asistente.asistente_recomendaciones import recomendaciones_handler

db = SessionLocal()
try:
    # ── KNN TEST ──────────────────────────────────────────────────────────────
    print("=" * 60)
    print("KNN - Recomendaciones por perfil")
    print("=" * 60)
    for cid in [55, 60, 65]:
        c = db.query(Client).filter(Client.id == cid).first()
        plan_hoy = {'calorias_dia': 2000, 'proteinas_g': 150, 'carbohidratos_g': 220, 'grasas_g': 60}
        recs = recomendaciones_handler.obtener_recomendaciones_knn(c, plan_hoy, db, n=3)
        perfil_esp = 'A' if cid <= 59 else ('B' if cid <= 64 else 'C')
        print("Perfil %s (ID %d):" % (perfil_esp, cid))
        for r in recs:
            nombre = r.get("alimento", "?")
            kcal   = r.get("calorias_100g", 0)
            prot   = r.get("proteinas_100g", 0)
            sim    = r.get("similitud", 0)
            print("  - %s | %.0f kcal/100g | P=%.1fg | sim=%.1f%%" % (nombre, kcal, prot, sim*100))
        if not recs:
            print("  [SIN RECOMENDACIONES]")
    print()
    
    # ── RF TEST ──────────────────────────────────────────────────────────────
    print("=" * 60)
    print("RF - Predicciones por perfil (freq real vs. esperada)")
    print("=" * 60)
    clientes = db.query(Client).filter(Client.id >= 55, Client.id <= 69).order_by(Client.id).all()
    correctos = 0
    for c in clientes:
        perfil, conf = recomendaciones_handler.predecir_perfil(c, db)
        f = recomendaciones_handler.preparar_features_rf(c, db)
        esperado = 'PERFIL_A' if c.id <= 59 else ('PERFIL_B' if c.id <= 64 else 'PERFIL_C')
        ok = "OK" if perfil == esperado else "FAIL"
        if ok == "OK":
            correctos += 1
        print("%s | ID %d | pred=%s(%.0f%%) esp=%s | freq=%d | session=%.1fh | wt=%s" % (
            ok, c.id, perfil, conf*100, esperado, f['workout_freq'], f['session_hours'], f['workout_type']
        ))
    print("Precision RF: %d/15" % correctos)
finally:
    db.close()
