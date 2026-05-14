"""
Regenera el campo `tecnica` de ejercicios que solo tienen el placeholder genérico.
Usa la misma API Groq ya integrada en el proyecto.
Ejecutar: docker exec calofit_backend python scripts/generar_tecnica_ejercicios.py
"""
import asyncio
import time
import sys

sys.path.insert(0, "/app")

from sqlalchemy import text
from app.core.database import SessionLocal
from app.services.ia_service import ia_engine

PROMPT_TECNICA = """Eres un entrenador de gym profesional. Escribe la técnica correcta para el ejercicio "{nombre}".

Responde SOLO con 4-5 pasos numerados en este formato exacto:
1. [primer paso breve y claro].
2. [segundo paso].
3. [tercer paso].
4. [cuarto paso].
5. [quinto paso opcional si aplica].

Reglas:
- Máximo 20 palabras por paso.
- Sin introducción ni explicación extra.
- Solo los pasos numerados, nada más."""


async def generar_tecnica(nombre: str) -> str:
    prompt = PROMPT_TECNICA.format(nombre=nombre)
    try:
        resp = await ia_engine._llamar_groq(prompt=prompt, max_tokens=200, temp=0.3)
        # Verificar que tenga al menos un paso numerado
        if "1." in resp:
            return resp.strip()
    except Exception as e:
        print(f"  Error LLM: {e}")
    return ""


async def main():
    db = SessionLocal()
    rows = db.execute(text(
        "SELECT id, nombre FROM ejercicios "
        "WHERE tecnica LIKE '%técnica controlada%' "
        "ORDER BY nombre"
    )).fetchall()

    print(f"Ejercicios a actualizar: {len(rows)}")
    actualizados = 0
    errores = 0

    for i, (eid, nombre) in enumerate(rows):
        print(f"[{i+1}/{len(rows)}] {nombre} ... ", end="", flush=True)
        tecnica = await generar_tecnica(nombre)
        if tecnica:
            db.execute(
                text("UPDATE ejercicios SET tecnica = :t WHERE id = :id"),
                {"t": tecnica, "id": eid}
            )
            db.commit()
            print("OK")
            actualizados += 1
        else:
            print("SKIP (sin respuesta)")
            errores += 1

        # Pausa para no saturar rate limit de Groq
        if (i + 1) % 10 == 0:
            print("  [pausa 3s para rate limit...]")
            await asyncio.sleep(3)
        else:
            await asyncio.sleep(0.5)

    db.close()
    print(f"\nResumen: {actualizados} actualizados, {errores} con error.")


if __name__ == "__main__":
    asyncio.run(main())
