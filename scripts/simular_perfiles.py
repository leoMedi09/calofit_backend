"""
Simula el comportamiento de los 3 perfiles de usuario en CaloFit.
Verifica que el RF prediga el perfil correcto para cada grupo de usuarios de prueba.

Ejecutar: docker exec calofit_backend python scripts/simular_perfiles.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.database import SessionLocal
from app.models.client import Client
from app.services.asistente_recomendaciones import RecomendacionesHandler

PERFILES_ESPERADOS = {
    **{cid: "PERFIL_A" for cid in range(55, 60)},
    **{cid: "PERFIL_B" for cid in range(60, 65)},
    **{cid: "PERFIL_C" for cid in range(65, 70)},
}

GRUPOS = {
    "A — Disciplinado (adherencia ≥90%, workout 4-5×/sem)": list(range(55, 60)),
    "B — Intermedio  (adherencia ~60%, workout 2-3×/sem)":  list(range(60, 65)),
    "C — Crítico     (exceso carbs/grasas, sedentarios)":   list(range(65, 70)),
}


def simular():
    db = SessionLocal()
    handler = RecomendacionesHandler()
    aciertos = 0
    total = 0

    try:
        for nombre_grupo, ids in GRUPOS.items():
            print(f"\n{'='*65}")
            print(f"PERFIL {nombre_grupo}")
            print(f"{'='*65}")
            for client_id in ids:
                try:
                    perfil_obj = db.query(Client).filter(Client.id == client_id).first()
                    if not perfil_obj:
                        print(f"  Client {client_id}: NO ENCONTRADO en BD")
                        total += 1
                        continue

                    perfil_pred, confianza = handler.predecir_perfil(perfil_obj, db)
                    esperado = PERFILES_ESPERADOS[client_id]
                    ok = "✅" if perfil_pred == esperado else "❌"
                    total += 1
                    if perfil_pred == esperado:
                        aciertos += 1

                    nombre = (perfil_obj.first_name or '?')
                    print(
                        f"  Client {client_id} ({nombre}): "
                        f"pred={perfil_pred} ({confianza:.0f}%) | "
                        f"esperado={esperado} {ok}"
                    )
                except Exception as e:
                    print(f"  Client {client_id}: ERROR — {e}")
                    total += 1

        print(f"\n{'='*65}")
        acc = aciertos / max(total, 1) * 100
        print(f"RESUMEN: {aciertos}/{total} predicciones correctas ({acc:.0f}%)")
        if aciertos == total:
            print("  ✅ RF alineado con los 3 perfiles de prueba — listo para tesis")
        elif acc >= 80:
            print(f"  ~ Accuracy aceptable ({acc:.0f}%) — revisar clientes con ❌")
        else:
            print(f"  ⚠ Accuracy bajo ({acc:.0f}%) — considerar ajustar features en ml_service.py")
    finally:
        db.close()


if __name__ == "__main__":
    simular()
