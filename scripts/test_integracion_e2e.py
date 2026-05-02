"""
Prueba de Integración End-to-End — Módulo Nutrición CaloFit
Verifica la cadena completa: KNN → Plato dinámico → Trinidad → RF para 3 perfiles.

Ejecutar: docker exec calofit_backend python scripts/test_integracion_e2e.py
"""
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text

from app.core.database import SessionLocal
from app.models.client import Client
from app.services.asistente_nutricion import _cargar_ingredientes_bd
from app.services.asistente_recomendaciones import RecomendacionesHandler
from app.services.plato_constructor import crear_plato_dinamico

SUJETOS = {55: "PERFIL_A", 60: "PERFIL_B", 65: "PERFIL_C"}
NOMBRE_PLATO_TEST = "Sudado de pescado"
SEP = "=" * 65


def _get_plan_hoy(db, client_id: int) -> dict:
    row = db.execute(text("""
        SELECT pd.calorias_dia, pd.proteinas_g, pd.carbohidratos_g, pd.grasas_g
        FROM planes_nutricionales pn
        JOIN planes_diarios pd ON pd.plan_id = pn.id
        WHERE pn.client_id = :cid
          AND pn.status IN ('validado','activo','pendiente')
          AND pd.dia_numero = EXTRACT(DOW FROM NOW())::int
        ORDER BY pn.id DESC
        LIMIT 1
    """), {"cid": client_id}).fetchone()
    if row:
        return {
            "calorias_dia":    float(row[0] or 0),
            "proteinas_g":     float(row[1] or 0),
            "carbohidratos_g": float(row[2] or 0),
            "grasas_g":        float(row[3] or 0),
        }
    return {"calorias_dia": 2000.0, "proteinas_g": 120.0, "carbohidratos_g": 250.0, "grasas_g": 55.0}


def _get_consumido_hoy(db, client_id: int) -> dict:
    from app.models.historial import ProgresoCalorias
    from app.core.utils import get_peru_date
    prog = db.query(ProgresoCalorias).filter(
        ProgresoCalorias.client_id == client_id,
        ProgresoCalorias.fecha == get_peru_date(),
    ).first()
    if prog:
        return {
            "calorias":    float(prog.calorias_consumidas or 0),
            "proteinas":   float(prog.proteinas_consumidas or 0),
            "carbos":      float(prog.carbohidratos_consumidos or 0),
            "grasas":      float(prog.grasas_consumidas or 0),
        }
    return {"calorias": 0.0, "proteinas": 0.0, "carbos": 0.0, "grasas": 0.0}


# ─── FASE 1: Recomendaciones KNN ─────────────────────────────────────────────

def fase_recomendaciones(db, handler: RecomendacionesHandler) -> dict:
    """Devuelve {client_id: {"reco_nombre": ..., "reco_kcal": ..., "deficit_kcal": ...}}"""
    resultados = {}
    print(f"\n{SEP}")
    print("FASE 1 — RECOMENDACIONES KNN (Cena personalizada)")
    print(SEP)

    for cid, perfil_esperado in SUJETOS.items():
        perfil_obj = db.query(Client).filter(Client.id == cid).first()
        plan = _get_plan_hoy(db, cid)
        consumido = _get_consumido_hoy(db, cid)

        deficit_kcal = max(0.0, plan["calorias_dia"] - consumido["calorias"])
        superavit = consumido["calorias"] > plan["calorias_dia"]

        print(f"\n  [{perfil_esperado}] {perfil_obj.first_name} (ID {cid})")
        print(f"  Plan: {plan['calorias_dia']:.0f} kcal | "
              f"Consumido: {consumido['calorias']:.0f} kcal | "
              f"{'⚠ SUPERÁVIT' if superavit else f'Déficit: {deficit_kcal:.0f} kcal'}")

        recos = handler.obtener_recomendaciones_knn(perfil_obj, plan, db, n=3)

        if recos:
            print(f"  Recomendaciones para cena:")
            for i, r in enumerate(recos[:3], 1):
                sim_pct = r.get("similitud", 0)
                nombre_alim = r.get("alimento", r.get("nombre", "?"))
                print(f"    {i}. {nombre_alim:<35} "
                      f"{r.get('calorias_100g', 0):.0f} kcal/100g | "
                      f"P:{r.get('proteina_100g', 0):.1f}g | "
                      f"C:{r.get('carbohindratos_100g', r.get('carbohidratos_100g', 0)):.1f}g | "
                      f"G:{r.get('grasas_100g', 0):.1f}g "
                      f"(sim {sim_pct}%)")
            reco0_nombre = recos[0].get("alimento", recos[0].get("nombre", "?"))
            resultados[cid] = {
                "nombre": perfil_obj.first_name,
                "perfil_esperado": perfil_esperado,
                "reco_nombre": reco0_nombre,
                "reco_kcal": round(recos[0].get("calorias_100g", 0), 1),
                "deficit_kcal": round(deficit_kcal, 1),
                "superavit": superavit,
            }
        else:
            print(f"  Sin recomendaciones (déficit = 0 o historial reciente lleno)")
            resultados[cid] = {
                "nombre": perfil_obj.first_name,
                "perfil_esperado": perfil_esperado,
                "reco_nombre": "Sin déficit",
                "reco_kcal": 0,
                "deficit_kcal": 0,
                "superavit": superavit,
            }

    return resultados


# ─── FASE 2: Registro dinámico ────────────────────────────────────────────────

async def fase_registro_dinamico(db) -> dict:
    """Crea 'Sudado de pescado', valida Trinidad y encoding."""
    print(f"\n{SEP}")
    print(f"FASE 2 — REGISTRO DINÁMICO: '{NOMBRE_PLATO_TEST}'")
    print(SEP)

    plato = await crear_plato_dinamico(db, NOMBRE_PLATO_TEST, tipo_plato="almuerzo")

    if not plato:
        print(f"  ❌ ERROR: crear_plato_dinamico retornó None")
        return {"ok": False, "error": "No se pudo crear el plato"}

    macros = plato.calcular_macros()
    desglose = _cargar_ingredientes_bd(db, plato.id)

    # Validaciones
    checks = {}

    # 1. Ingredientes suficientes
    n_ing = len(plato.ingredientes)
    checks["ingredientes_min_2"] = n_ing >= 2
    print(f"\n  Plato creado: '{plato.nombre}' (id={plato.id}, origen={plato.origen})")
    print(f"  Ingredientes: {n_ing} {'✅' if checks['ingredientes_min_2'] else '❌'}")

    # 2. Preparación sin caracteres ??
    prep_json = json.dumps(plato.preparacion, ensure_ascii=False) if plato.preparacion else ""
    checks["sin_encoding_error"] = "??" not in prep_json
    print(f"  Encoding (sin ??): {'✅' if checks['sin_encoding_error'] else '❌ HAY ??' }")

    # 3. Todos los ingredientes aparecen en preparación
    nombres_ing = [ing.alimento.nombre.lower() for ing in plato.ingredientes if ing.alimento]
    prep_texto = " ".join(plato.preparacion or []).lower()
    faltantes = []
    for nombre in nombres_ing:
        # Verificar con primera palabra del nombre del ingrediente (ej. "arroz" de "Arroz blanco")
        primera_palabra = nombre.split()[0]
        if primera_palabra not in prep_texto:
            faltantes.append(nombre)
    checks["todos_ingredientes_en_prep"] = len(faltantes) == 0
    if faltantes:
        print(f"  Ingredientes no mencionados en preparación: {faltantes}")
        print(f"  Todos en preparación: ⚠ ({len(faltantes)} faltantes)")
    else:
        print(f"  Todos los ingredientes en preparación: ✅")

    # 4. Desglose Trinidad
    print(f"\n  📊 Desglose Trinidad (platos→plato_ingredientes→alimentos):")
    for linea in desglose:
        print(f"    • {linea}")
    total_line = (
        f"Total: {macros['calorias']} kcal | "
        f"P:{macros['proteinas_g']}g | "
        f"C:{macros['carbohidratos_g']}g | "
        f"G:{macros['grasas_g']}g"
    )
    print(f"    ━━ {total_line}")

    # 5. Preparación (primeros 2 pasos)
    print(f"\n  Preparación (primeros 2 pasos):")
    for paso in (plato.preparacion or [])[:2]:
        print(f"    → {paso}")

    return {
        "ok": True,
        "plato_id": plato.id,
        "plato_nombre": plato.nombre,
        "n_ingredientes": n_ing,
        "macros": macros,
        "desglose": desglose,
        "total_line": total_line,
        "checks": checks,
        "preparacion_pasos": len(plato.preparacion or []),
    }


# ─── FASE 3: Verificación RF post-registro ────────────────────────────────────

def fase_verificacion_rf(db, handler: RecomendacionesHandler, resultados_reco: dict) -> dict:
    """Confirma que el RF mantiene los perfiles correctos después del registro."""
    print(f"\n{SEP}")
    print("FASE 3 — VERIFICACIÓN RF POST-REGISTRO")
    print(SEP)
    aciertos = 0
    rf_resultados = {}

    for cid, perfil_esperado in SUJETOS.items():
        perfil_obj = db.query(Client).filter(Client.id == cid).first()
        perfil_pred, confianza = handler.predecir_perfil(perfil_obj, db)
        ok = perfil_pred == perfil_esperado
        if ok:
            aciertos += 1
        nombre = resultados_reco.get(cid, {}).get("nombre", "?")
        print(f"  {nombre} ({cid}): pred={perfil_pred} ({confianza:.0f}%) "
              f"esperado={perfil_esperado} {'✅' if ok else '❌'}")
        rf_resultados[cid] = {"pred": perfil_pred, "confianza": confianza, "ok": ok}

    print(f"\n  RF Accuracy post-registro: {aciertos}/3 ({'100%' if aciertos == 3 else f'{aciertos/3*100:.0f}%'})")
    return rf_resultados


# ─── Main ─────────────────────────────────────────────────────────────────────

async def main():
    db = SessionLocal()
    handler = RecomendacionesHandler()

    try:
        # Fase 1
        resultados_reco = fase_recomendaciones(db, handler)

        # Fase 2
        resultado_plato = await fase_registro_dinamico(db)

        # Fase 3
        rf_resultados = fase_verificacion_rf(db, handler, resultados_reco)

        # Resumen final
        print(f"\n{SEP}")
        print("RESUMEN FINAL — E2E INTEGRATION TEST")
        print(SEP)
        print(f"\n  {'Usuario':<20} {'Perfil':<10} {'Recomendación':<30} {'Déficit':<12} {'RF Post':<12}")
        print(f"  {'-'*20} {'-'*10} {'-'*30} {'-'*12} {'-'*12}")
        for cid, data in resultados_reco.items():
            rf = rf_resultados.get(cid, {})
            deficit_str = "SUPERÁVIT" if data.get("superavit") else f"{data['deficit_kcal']:.0f} kcal"
            print(f"  {data['nombre']:<20} {data['perfil_esperado']:<10} "
                  f"{data['reco_nombre'][:28]:<30} {deficit_str:<12} "
                  f"{'✅ ' + rf.get('pred','?') if rf.get('ok') else '❌ ' + rf.get('pred','?'):<12}")

        if resultado_plato.get("ok"):
            macros = resultado_plato["macros"]
            checks = resultado_plato["checks"]
            enc_ok = '✅' if checks.get('sin_encoding_error') else '❌'
            ing_ok = '✅' if checks.get('todos_ingredientes_en_prep') else '⚠'
            print(f"\n  Plato dinámico: '{resultado_plato['plato_nombre']}' (id={resultado_plato['plato_id']})")
            print(f"  Total kcal (Trinidad): {macros['calorias']} kcal | "
                  f"P:{macros['proteinas_g']}g | C:{macros['carbohidratos_g']}g | G:{macros['grasas_g']}g")
            print(f"  Encoding sin ??: {enc_ok} | Ingredientes en preparación: {ing_ok}")
            print(f"  Ingredientes: {resultado_plato['n_ingredientes']} | Pasos: {resultado_plato['preparacion_pasos']}")

        all_rf_ok = all(v["ok"] for v in rf_resultados.values())
        plato_ok  = resultado_plato.get("ok", False)
        enc_ok    = resultado_plato.get("checks", {}).get("sin_encoding_error", False) if plato_ok else False

        print(f"\n  {'='*40}")
        if all_rf_ok and plato_ok and enc_ok:
            print(f"  ✅ MÓDULO NUTRICIÓN — LISTO PARA ENTREGA FINAL")
        else:
            print(f"  ⚠  Revisar checks fallidos antes de la entrega")
        print(f"  {'='*40}\n")

    finally:
        db.close()


if __name__ == "__main__":
    asyncio.run(main())
