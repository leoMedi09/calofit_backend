import asyncio, sys, os
sys.path.insert(0, '/app')
from app.core.database import SessionLocal
from app.models.plato import Plato
from app.services.plato_constructor import _generar_preparacion_llm

async def regen():
    db = SessionLocal()
    try:
        for pid in [138, 175]:
            p = db.query(Plato).filter(Plato.id == pid).first()
            if not p:
                print(f'  id={pid} -> NOT FOUND')
                continue
            ings = [i.alimento.nombre for i in p.ingredientes if i.alimento]
            print(f'  id={pid} [{p.nombre}] -- {len(ings)} ings')
            nueva = await _generar_preparacion_llm(p.nombre, ings)
            if nueva and len(nueva) >= 3:
                p.preparacion = nueva
                db.add(p)
                print(f'  -> {len(nueva)} pasos OK')
                for i, paso in enumerate(nueva, 1):
                    print(f'     {i}. {paso}')
            else:
                print(f'  -> LLM fail: {nueva}')
        db.commit()
        print('Commit OK')
    finally:
        db.close()

asyncio.run(regen())
