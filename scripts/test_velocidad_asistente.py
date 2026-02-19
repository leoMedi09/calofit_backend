"""
🔬 TEST DE VELOCIDAD DEL ASISTENTE IA
Mide el tiempo real de cada fase del proceso para identificar cuellos de botella.
Uso: python scripts/test_velocidad_asistente.py
"""
import asyncio
import time
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.services.ia_service import ia_engine

# Contexto simulado (igual al que genera asistente.py en producción)
CONTEXTO_TEST = (
    "Eres el coach de Leonardo. "
    "PERFIL: 75kg, 172cm, 22 años. "
    "ALERGIAS: Ninguna. "
    "PREFERENCIAS DIETÉTICAS: Omnívoro. "
    "CONDICIONES MÉDICAS: Ninguna. "
    "STATUS DEL DÍA: Meta: 3000 kcal. Consumido: 200 kcal. Restante: 2800 kcal. "
    "Adherencia: 30%, Progreso: 50%. ¡Sigue esforzándote, estás mejorando!"
)

MENSAJES_TEST = [
    ("SALUDO",           "Hola, buenos días"),
    ("INFO SIMPLE",      "¿El aguacate engorda?"),
    ("RECETA SIMPLE",    "Dame una receta de almuerzo peruano"),
    ("OPCIONES (LENTO)", "Dame opciones de cenas bajas en calorías"),
    ("RUTINA",           "Dame una rutina de 30 minutos para casa"),
]

async def medir_llamada_ia(mensaje: str, etiqueta: str):
    """Mide el tiempo de una llamada individual al asistente."""
    print(f"\n{'─'*55}")
    print(f"🧪 Test: {etiqueta}")
    print(f"💬 Mensaje: \"{mensaje}\"")
    
    t0 = time.perf_counter()
    
    try:
        respuesta = await ia_engine.asistir_cliente(
            contexto=CONTEXTO_TEST,
            mensaje_usuario=mensaje,
            historial=None,
            tono_aplicado="Usa un tono motivador."
        )
        t1 = time.perf_counter()
        duracion = t1 - t0
        
        tokens_aprox = len(respuesta.split())
        tiene_receta = "[CALOFIT_HEADER]" in respuesta
        tiene_stats  = "[CALOFIT_STATS]" in respuesta
        
        print(f"⏱️  Tiempo total: {duracion:.2f}s")
        print(f"📝 Tokens aprox: {tokens_aprox} palabras")
        print(f"🃏 Tiene receta: {'✅' if tiene_receta else '❌'}")
        print(f"📊 Tiene stats:  {'✅' if tiene_stats else '❌'}")
        print(f"📄 Inicio resp:  {respuesta[:100].strip()}...")
        
        # Clasificar velocidad
        if duracion < 3.0:
            estado = "🟢 RÁPIDO"
        elif duracion < 6.0:
            estado = "🟡 ACEPTABLE"
        else:
            estado = "🔴 LENTO"
        print(f"Estado: {estado}")
        
        return duracion
        
    except Exception as e:
        t1 = time.perf_counter()
        duracion = t1 - t0
        print(f"❌ ERROR ({duracion:.2f}s): {e}")
        return duracion

async def medir_intencion_salud(mensaje: str):
    """Mide el tiempo de identificar_intencion_salud por separado."""
    print(f"\n{'─'*55}")
    print(f"🧪 Test: ANÁLISIS SALUD (background task)")
    print(f"💬 Mensaje: \"{mensaje}\"")
    
    t0 = time.perf_counter()
    try:
        resultado = await ia_engine.identificar_intencion_salud(mensaje)
        t1 = time.perf_counter()
        duracion = t1 - t0
        print(f"⏱️  Tiempo: {duracion:.2f}s → {resultado}")
        return duracion
    except Exception as e:
        t1 = time.perf_counter()
        print(f"❌ ERROR ({t1-t0:.2f}s): {e}")
        return t1 - t0

async def main():
    print("=" * 55)
    print("  🔬 BENCHMARK ASISTENTE CALOFIT IA")
    print(f"  Hora: {time.strftime('%H:%M:%S')}")
    print("=" * 55)
    
    # Test 1: ¿Tarda el análisis de salud?
    print("\n📌 FASE 1: Verificar modelo de análisis de salud")
    t_salud = await medir_intencion_salud("Me duele la rodilla cuando corro")
    
    # Test 2: Medir cada tipo de mensaje principal
    print("\n📌 FASE 2: Benchmark del asistente principal")
    tiempos = []
    for etiqueta, mensaje in MENSAJES_TEST:
        t = await medir_llamada_ia(mensaje, etiqueta)
        tiempos.append((etiqueta, t))
        await asyncio.sleep(1)  # Pequeña pausa entre llamadas para no saturar la API
    
    # Resumen final
    print(f"\n{'='*55}")
    print("📊 RESUMEN FINAL")
    print(f"{'='*55}")
    print(f"{'Tipo':<25} {'Tiempo':>10}  {'Estado'}")
    print(f"{'─'*55}")
    print(f"{'Análisis Salud (bg)':<25} {t_salud:>9.2f}s  {'🟢' if t_salud < 3 else '🔴'}")
    for etiqueta, t in tiempos:
        estado = "🟢" if t < 3 else ("🟡" if t < 6 else "🔴")
        print(f"{etiqueta:<25} {t:>9.2f}s  {estado}")
    
    promedio = sum(t for _, t in tiempos) / len(tiempos)
    print(f"{'─'*55}")
    print(f"⌀ Promedio mensajes:   {promedio:.2f}s")
    print(f"\n✅ Test completado.")

if __name__ == "__main__":
    asyncio.run(main())
