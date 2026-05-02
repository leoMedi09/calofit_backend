"""
Repara las preparaciones con caracteres corruptos (??) regenerándolas con el LLM.
El fix unicodedata.normalize('NFC') en _generar_preparacion_llm() ya está activo,
por lo que las nuevas preparaciones estarán libres de corrupción.

Ejecutar: docker exec calofit_backend python scripts/reparar_preparaciones_encoding.py
"""
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from app.core.database import SessionLocal
from app.models.plato import Plato
from app.services.plato_constructor import _generar_preparacion_llm


async def reparar():
    db = SessionLocal()
    try:
        # Encontrar platos con ?? en preparacion
        rows = db.execute(text(
            "SELECT id, nombre FROM platos WHERE preparacion::text LIKE '%??%' ORDER BY id"
        )).fetchall()

        print(f"Platos con encoding corrupto: {len(rows)}")
        print("=" * 60)

        reparados = 0
        fallidos = []

        for plato_id, plato_nombre in rows:
            plato = db.query(Plato).filter(Plato.id == plato_id).first()
            if not plato:
                fallidos.append((plato_id, "No encontrado en ORM"))
                continue

            nombres_ings = [
                ing.alimento.nombre
                for ing in plato.ingredientes
                if ing.alimento
            ]

            if len(nombres_ings) < 2:
                print(f"  [{plato_id}] {plato_nombre} — sin ingredientes, omitido")
                fallidos.append((plato_id, "Sin ingredientes resueltos"))
                continue

            print(f"  [{plato_id}] {plato_nombre} ({len(nombres_ings)} ing.)  ...", end=" ", flush=True)

            nueva_prep = await _generar_preparacion_llm(plato_nombre, nombres_ings)

            if nueva_prep and len(nueva_prep) >= 3:
                # Verificar que la nueva preparación no tiene ??
                prep_str = json.dumps(nueva_prep, ensure_ascii=False)
                if "??" in prep_str:
                    print("⚠ aún tiene ?? — reintentando con NFC extra")
                    import unicodedata
                    nueva_prep = [unicodedata.normalize("NFC", p) for p in nueva_prep]

                plato.preparacion = nueva_prep
                db.add(plato)
                db.flush()
                reparados += 1
                print(f"✅  ({len(nueva_prep)} pasos)")
            else:
                print("❌ LLM no devolvió pasos válidos")
                fallidos.append((plato_id, "LLM sin respuesta"))

        db.commit()

        print(f"\n{'='*60}")
        print(f"Reparados: {reparados}/{len(rows)}")
        if fallidos:
            print(f"Fallidos ({len(fallidos)}):")
            for pid, motivo in fallidos:
                print(f"  id={pid}: {motivo}")

        # Verificación final
        restantes = db.execute(text(
            "SELECT COUNT(*) FROM platos WHERE preparacion::text LIKE '%??%'"
        )).scalar()
        print(f"\nPlatos con ?? restantes: {restantes} {'✅' if restantes == 0 else '⚠'}")

    finally:
        db.close()


if __name__ == "__main__":
    asyncio.run(reparar())
