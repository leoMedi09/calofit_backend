import asyncio, sys
sys.path.insert(0, '/app')
from app.core.database import SessionLocal
from app.models.plato import Plato
from app.services.plato_constructor import _generar_preparacion_llm

async def regen():
    db = SessionLocal()
    try:
        for pid in [28, 176, 177, 178, 180]:
            p = db.query(Plato).filter(Plato.id == pid).first()
            if not p: continue
            ings = [i.alimento.nombre for i in p.ingredientes if i.alimento]
            print(f"  id={pid} '{p.nombre}' — {ings}")
            nueva = await _generar_preparacion_llm(p.nombre, ings)
            if nueva and len(nueva) >= 3:
                p.preparacion = nueva
                db.add(p)
                print(f"  → {len(nueva)} pasos OK")
            else:
                print(f"  → LLM fail")
        db.commit()
        print("Commit OK")
    finally:
        db.close()

asyncio.run(regen())
