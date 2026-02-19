"""
🔬 TEST HTTP DE VELOCIDAD - Llama al endpoint real del servidor
Uso: python scripts/test_velocidad_http.py
REQUISITO: El servidor debe estar corriendo en http://localhost:8000
"""
import requests
import time
import sys

# ──────────────────────────────────────────────
# CONFIGURACIÓN (ajusta si es necesario)
BASE_URL = "http://localhost:8000"
EMAIL    = "alfaelmejor0902@gmail.com"
PASSWORD = "alfa0902"
# ──────────────────────────────────────────────

MENSAJES_TEST = [
    ("SALUDO",           "Hola, buenos días"),
    ("INFO SIMPLE",      "¿El aguacate engorda?"),
    ("RECETA SIMPLE",    "Dame una receta de almuerzo peruano"),
    ("OPCIONES CENA",    "Dame opciones de cenas bajas en calorías"),
]

def login():
    print(f"🔐 Iniciando sesión como {EMAIL}...")
    resp = requests.post(f"{BASE_URL}/auth/login", json={
        "email": EMAIL,
        "password": PASSWORD,
        "remember_me": False,
        "firebase_uid": "",
        "user_type": "client"
    }, timeout=15)
    if resp.status_code != 200:
        print(f"❌ Login fallido: {resp.status_code} - {resp.text[:200]}")
        sys.exit(1)
    token = resp.json().get("access_token") or resp.json().get("token")
    print(f"✅ Token obtenido.")
    return token

def consultar_asistente(token: str, mensaje: str):
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    t0 = time.perf_counter()
    try:
        resp = requests.post(
            f"{BASE_URL}/asistente/consultar",
            json={"mensaje": mensaje, "historial": []},
            headers=headers,
            timeout=60
        )
        print(f"   [DEBUG] HTTP Status: {resp.status_code}")
        t1 = time.perf_counter()
        duracion = t1 - t0
        
        if resp.status_code == 200:
            data = resp.json()
            respuesta = data.get("respuesta_ia", "")
            intent = data.get("respuesta_estructurada", {}).get("intent", "?")
            tiene_header = "[CALOFIT_HEADER]" in respuesta
            tiene_stats  = "[CALOFIT_STATS]"  in respuesta
            return duracion, True, intent, tiene_header, tiene_stats, len(respuesta.split())
        else:
            error_detail = f"HTTP {resp.status_code}: {resp.text[:150]}"
            return duracion, False, error_detail, False, False, 0
    except requests.exceptions.Timeout:
        t1 = time.perf_counter()
        return t1 - t0, False, "TIMEOUT", False, False, 0
    except Exception as e:
        t1 = time.perf_counter()
        return t1 - t0, False, str(e)[:40], False, False, 0

def main():
    print("=" * 60)
    print("  🔬 BENCHMARK HTTP - ASISTENTE CALOFIT IA")
    print(f"  Servidor: {BASE_URL}")
    print(f"  Hora: {time.strftime('%H:%M:%S')}")
    print("=" * 60)
    
    # Login
    try:
        token = login()
    except Exception as e:
        print(f"❌ No se pudo conectar al servidor: {e}")
        print("   ¿Está el servidor corriendo? Ejecuta: uvicorn app.main:app --reload")
        sys.exit(1)
    
    # Tests
    resultados = []
    for etiqueta, mensaje in MENSAJES_TEST:
        print(f"\n{'─'*60}")
        print(f"🧪 [{etiqueta}] → \"{mensaje}\"")
        print("   ⏳ Esperando respuesta...")
        
        duracion, ok, intent, tiene_header, tiene_stats, palabras = consultar_asistente(token, mensaje)
        
        if ok:
            estado = "🟢 RÁPIDO" if duracion < 3 else ("🟡 ACEPTABLE" if duracion < 7 else "🔴 LENTO")
            print(f"   ⏱️  Tiempo:  {duracion:.2f}s  {estado}")
            print(f"   🎯 Intent:  {intent}")
            print(f"   🃏 Receta:  {'✅' if tiene_header else '❌'}  |  📊 Stats: {'✅' if tiene_stats else '❌'}")
            print(f"   📝 Palabras generadas: ~{palabras}")
        else:
            print(f"   ❌ FALLO ({duracion:.2f}s) - {intent}")
        
        resultados.append((etiqueta, duracion, ok))
        time.sleep(1)
    
    # Resumen
    print(f"\n{'='*60}")
    print("📊 RESUMEN BENCHMARK")
    print(f"{'='*60}")
    print(f"  {'Tipo':<22} {'Tiempo':>8}  Estado")
    print(f"  {'─'*50}")
    for etiqueta, t, ok in resultados:
        if ok:
            estado = "🟢" if t < 3 else ("🟡" if t < 7 else "🔴")
            print(f"  {etiqueta:<22} {t:>7.2f}s  {estado}")
        else:
            print(f"  {etiqueta:<22} {'FALLO':>8}  ❌")
    
    tiempos_ok = [t for _, t, ok in resultados if ok]
    if tiempos_ok:
        print(f"  {'─'*50}")
        print(f"  {'Promedio':<22} {sum(tiempos_ok)/len(tiempos_ok):>7.2f}s")
        print(f"  {'Máximo':<22} {max(tiempos_ok):>7.2f}s")
        print(f"  {'Mínimo':<22} {min(tiempos_ok):>7.2f}s")
    
    print("\n✅ Benchmark completado.")

if __name__ == "__main__":
    main()
