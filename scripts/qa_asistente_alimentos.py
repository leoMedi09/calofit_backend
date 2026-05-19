"""
qa_asistente_alimentos.py
=========================
QA automatizado del motor de registro de alimentos de CaloFit.

Llama directamente al service layer (sin HTTP) usando registro_comida_handler.registrar().
Esto evita el ruido del LLM principal de consultar() y prueba el pipeline NLP 5 capas.

Uso:
    docker exec calofit_backend python scripts/qa_asistente_alimentos.py

Cliente de prueba: ID 55 (Carlos, Perfil A — disciplinado, datos validados).
"""
from __future__ import annotations

import asyncio
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

# ── Bootstrapping de paths (necesario para imports dentro de Docker) ──────────
sys.path.insert(0, "/app")

# ── Imports del proyecto ──────────────────────────────────────────────────────
from app.core.database import SessionLocal
from app.models.client import Client
from app.services.asistente_registro_comida import registro_comida_handler
from app.services.ia_service import ia_engine


# ══════════════════════════════════════════════════════════════════════════════
# Constantes de presentación
# ══════════════════════════════════════════════════════════════════════════════

SEP  = "─" * 80
SEP2 = "═" * 80

STATUS_PASS  = "PASS "
STATUS_FAIL  = "FAIL "
STATUS_ERROR = "ERROR"
STATUS_SKIP  = "SKIP "


# ══════════════════════════════════════════════════════════════════════════════
# Definición de casos de prueba
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TestCase:
    test_id: str
    grupo: str
    mensaje: str

    # Expectativas de ÉXITO (comida registrada)
    expect_registro: bool = True          # True → debe registrarse (success+kcal>0)
    kcal_min: float = 0.0
    kcal_max: float = 9999.0
    foods_min: int = 0                    # mínimo de alimentos detectados en result["alimentos"]
    foods_max: int = 99                   # máximo de alimentos detectados

    # Expectativa de BLOQUEO (no debe registrarse)
    expect_blocked: bool = False          # True → debe ser bloqueado (no_alimento, ficcion, etc.)

    # Etiqueta descriptiva
    descripcion: str = ""

    # Resultado real (rellenado durante la ejecución)
    resultado: Optional[dict] = field(default=None, repr=False)
    status: str = ""
    detalle: str = ""
    kcal_real: float = 0.0
    foods_real: list = field(default_factory=list, repr=False)
    duracion_s: float = 0.0


# ── Definición de todos los casos ─────────────────────────────────────────────

CASOS: list[TestCase] = [

    # ── GRUPO 1: Alimentos individuales con/sin gramos ────────────────────────

    TestCase(
        test_id="G1-01",
        grupo="Individuales",
        mensaje="comí un huevo frito",
        expect_registro=True,
        kcal_min=60,
        kcal_max=200,
        foods_min=1,
        descripcion="Huevo frito — porcion estándar ~44g × 196 kcal/100g",
    ),
    TestCase(
        test_id="G1-02",
        grupo="Individuales",
        mensaje="comí 2 huevos cocidos",
        expect_registro=True,
        kcal_min=100,
        kcal_max=250,
        foods_min=1,
        descripcion="2 huevos cocidos — ~155 kcal totales",
    ),
    TestCase(
        test_id="G1-03",
        grupo="Individuales",
        mensaje="comí 150g de pollo a la plancha",
        expect_registro=True,
        kcal_min=120,
        kcal_max=280,
        foods_min=1,
        descripcion="Pollo 150g con gramaje explícito",
    ),
    TestCase(
        test_id="G1-04",
        grupo="Individuales",
        mensaje="tomé un vaso de leche",
        expect_registro=True,
        kcal_min=80,
        kcal_max=250,
        foods_min=1,
        descripcion="Leche — vaso ~200ml, ~130-180 kcal",
    ),
    TestCase(
        test_id="G1-05",
        grupo="Individuales",
        mensaje="comí una manzana",
        expect_registro=True,
        kcal_min=40,
        kcal_max=120,
        foods_min=1,
        descripcion="Manzana — porción entera ~120g, ~50-80 kcal",
    ),

    # ── GRUPO 2: Platos compuestos (registro único) ───────────────────────────

    TestCase(
        test_id="G2-06",
        grupo="Platos compuestos",
        mensaje="comí lomo saltado",
        expect_registro=True,
        kcal_min=400,
        kcal_max=1000,
        foods_min=1,
        descripcion="Lomo saltado — plato completo 400-900 kcal",
    ),
    TestCase(
        test_id="G2-07",
        grupo="Platos compuestos",
        mensaje="comí ceviche de pescado",
        expect_registro=True,
        kcal_min=100,
        kcal_max=400,
        foods_min=1,
        descripcion="Ceviche de pescado — plato peruano 150-350 kcal",
    ),
    TestCase(
        test_id="G2-08",
        grupo="Platos compuestos",
        mensaje="tomé avena con leche",
        expect_registro=True,
        kcal_min=100,
        kcal_max=400,
        foods_min=1,
        descripcion="Avena con leche — debe registrarse como un solo item",
    ),

    # ── GRUPO 3: Concatenación (varios alimentos en un mensaje) ───────────────

    TestCase(
        test_id="G3-09",
        grupo="Concatenacion",
        mensaje="comí arroz con pollo y una manzana",
        expect_registro=True,
        kcal_min=400,
        kcal_max=1000,
        foods_min=2,
        descripcion="2 items: arroz+pollo y manzana — total 500-900 kcal",
    ),
    TestCase(
        test_id="G3-10",
        grupo="Concatenacion",
        mensaje="tomé un vaso de cocoa y un pan",
        expect_registro=True,
        kcal_min=80,
        kcal_max=400,
        foods_min=1,
        descripcion="Cocoa + pan — sin duplicados, total 100-300 kcal",
    ),
    TestCase(
        test_id="G3-11",
        grupo="Concatenacion",
        mensaje="comí 200g de pechuga de pollo y 100g de arroz blanco",
        expect_registro=True,
        kcal_min=250,
        kcal_max=600,
        foods_min=2,
        descripcion="Pechuga 200g + arroz 100g — dos gramajes explícitos",
    ),

    # ── GRUPO 4: Alimentos nuevos / complejos ─────────────────────────────────

    TestCase(
        test_id="G4-12",
        grupo="Complejos",
        mensaje="comí quinoa con verduras",
        expect_registro=True,
        kcal_min=100,
        kcal_max=600,
        foods_min=1,
        descripcion="Quinua con verduras — puede ser plato dinámico LLM",
    ),
    TestCase(
        test_id="G4-13",
        grupo="Complejos",
        mensaje="comí causa ferreñafana",
        expect_registro=True,
        kcal_min=350,
        kcal_max=700,
        foods_min=1,
        descripcion="Causa ferreñafana — plato regional Lambayeque 400-600 kcal",
    ),
    TestCase(
        test_id="G4-14",
        grupo="Complejos",
        mensaje="tomé un batido de plátano con avena",
        expect_registro=True,
        kcal_min=150,
        kcal_max=600,
        foods_min=1,
        descripcion="Batido plátano+avena — smoothie 200-500 kcal",
    ),

    # ── GRUPO 5: Anti-fraude / no-alimentos ───────────────────────────────────

    TestCase(
        test_id="G5-15",
        grupo="Anti-fraude",
        mensaje="comí juegos frito",
        expect_registro=False,
        expect_blocked=True,
        descripcion="'juegos' no es alimento — debe bloquearse",
    ),
    TestCase(
        test_id="G5-16",
        grupo="Anti-fraude",
        mensaje="comí clavo de hierro",
        expect_registro=False,
        expect_blocked=True,
        descripcion="'clavo de hierro' no es alimento — debe bloquearse",
    ),
    TestCase(
        test_id="G5-17",
        grupo="Anti-fraude",
        mensaje="comí mesa frita",
        expect_registro=False,
        expect_blocked=True,
        descripcion="'mesa' no es alimento — debe bloquearse",
    ),
]

# ── Grupo 6: suma acumulativa (dos registros separados) ───────────────────────
# Se definen como casos especiales procesados por _run_grupo6()

CASO_SUM_A = TestCase(
    test_id="G6-18",
    grupo="Suma",
    mensaje="comí 100g de arroz blanco",
    expect_registro=True,
    kcal_min=70,    # arroz blanco COCIDO 100g = 97 kcal (INS/CENAN)
    kcal_max=160,
    foods_min=1,
    descripcion="Arroz blanco 100g — Paso 1 de suma (~97 kcal cocido)",
)
CASO_SUM_B = TestCase(
    test_id="G6-19",
    grupo="Suma",
    mensaje="comí 100g de pollo al horno",
    expect_registro=True,
    kcal_min=100,
    kcal_max=250,
    foods_min=1,
    descripcion="Pollo al horno 100g — Paso 2 de suma",
)
CASO_SUM_VERIFY = TestCase(
    test_id="G6-20",
    grupo="Suma",
    mensaje="(verificacion matematica G6-18 + G6-19)",
    descripcion="kcal(18) + kcal(19) debe coincidir con la suma — tolerancia 5 kcal",
)


# ══════════════════════════════════════════════════════════════════════════════
# Motor de ejecución
# ══════════════════════════════════════════════════════════════════════════════

async def _ejecutar_caso(caso: TestCase, perfil, plan_hoy_data: dict, db) -> None:
    """Ejecuta un único caso de prueba y rellena caso.status / caso.detalle."""
    t0 = time.perf_counter()
    try:
        result = await registro_comida_handler.registrar(
            mensaje=caso.mensaje,
            perfil=perfil,
            plan_hoy_data=plan_hoy_data,
            db=db,
            ia_engine=ia_engine,
        )
    except Exception as exc:
        caso.duracion_s = time.perf_counter() - t0
        caso.status = STATUS_ERROR
        caso.detalle = f"EXCEPCION: {type(exc).__name__}: {exc}"
        return

    caso.duracion_s = time.perf_counter() - t0
    caso.resultado = result

    # Extraer campos clave del resultado
    success       = result.get("success", False)
    tipo          = result.get("tipo_detectado", "")
    datos         = result.get("datos") or {}
    kcal          = float(datos.get("calorias") or 0)
    alimentos     = result.get("alimentos") or []
    n_alimentos   = len(alimentos)

    caso.kcal_real  = kcal
    caso.foods_real = alimentos

    # ── Evaluación: casos esperados como BLOQUEADOS ───────────────────────────
    if caso.expect_blocked:
        # Un bloqueo correcto implica: success=False Y kcal=0
        is_blocked = (not success) and (kcal == 0)
        if is_blocked:
            caso.status  = STATUS_PASS
            caso.detalle = f"Bloqueado correctamente (tipo={tipo})"
        else:
            caso.status  = STATUS_FAIL
            caso.detalle = (
                f"ESPERABA BLOQUEO pero got success={success}, kcal={kcal:.1f}, "
                f"tipo={tipo}, alimentos={alimentos}"
            )
        return

    # ── Evaluación: casos esperados como REGISTRO ─────────────────────────────
    if caso.expect_registro:
        problemas = []

        if not success:
            problemas.append(f"success=False (tipo={tipo})")
        if kcal <= 0:
            problemas.append(f"kcal={kcal:.1f} (debe ser > 0)")
        if kcal > 0 and kcal < caso.kcal_min:
            problemas.append(f"kcal={kcal:.1f} < min_esperado={caso.kcal_min}")
        if kcal > caso.kcal_max:
            problemas.append(f"kcal={kcal:.1f} > max_esperado={caso.kcal_max}")
        if n_alimentos < caso.foods_min:
            problemas.append(f"alimentos={n_alimentos} < min_esperado={caso.foods_min}")
        if n_alimentos > caso.foods_max:
            problemas.append(f"alimentos={n_alimentos} > max_esperado={caso.foods_max}")

        if not problemas:
            caso.status  = STATUS_PASS
            caso.detalle = f"kcal={kcal:.1f}, alimentos({n_alimentos})={alimentos[:3]}"
        else:
            caso.status  = STATUS_FAIL
            caso.detalle = " | ".join(problemas) + f" | alimentos={alimentos}"
        return

    # Caso sin expectativa definida → marcar skip
    caso.status  = STATUS_SKIP
    caso.detalle = "Sin expectativa definida"


async def _run_grupo6(perfil, plan_hoy_data: dict, db) -> TestCase:
    """
    Ejecuta los dos registros de G6 y verifica que la suma matemática sea correcta.
    Retorna el TestCase de verificación con status rellenado.
    """
    await _ejecutar_caso(CASO_SUM_A, perfil, plan_hoy_data, db)
    await _ejecutar_caso(CASO_SUM_B, perfil, plan_hoy_data, db)

    kcal_a = CASO_SUM_A.kcal_real
    kcal_b = CASO_SUM_B.kcal_real
    suma   = round(kcal_a + kcal_b, 1)

    CASO_SUM_VERIFY.kcal_real  = suma
    CASO_SUM_VERIFY.foods_real = []

    problemas = []
    if CASO_SUM_A.status not in (STATUS_PASS,):
        problemas.append(f"G6-18 no pasó: {CASO_SUM_A.status} — {CASO_SUM_A.detalle}")
    if CASO_SUM_B.status not in (STATUS_PASS,):
        problemas.append(f"G6-19 no pasó: {CASO_SUM_B.status} — {CASO_SUM_B.detalle}")
    if not problemas:
        # Verificar que ambas kcal sean razonables individualmente
        if kcal_a <= 0 or kcal_b <= 0:
            problemas.append(f"Una de las kcal es 0: kcal_A={kcal_a}, kcal_B={kcal_b}")
        else:
            # La suma debe estar dentro del rango combinado esperado
            sum_min = CASO_SUM_A.kcal_min + CASO_SUM_B.kcal_min
            sum_max = CASO_SUM_A.kcal_max + CASO_SUM_B.kcal_max
            if suma < sum_min - 5 or suma > sum_max + 5:
                problemas.append(
                    f"suma={suma:.1f} fuera de [{sum_min},{sum_max}] "
                    f"(kcal_A={kcal_a:.1f} + kcal_B={kcal_b:.1f})"
                )

    if not problemas:
        CASO_SUM_VERIFY.status  = STATUS_PASS
        CASO_SUM_VERIFY.detalle = (
            f"kcal_A={kcal_a:.1f} + kcal_B={kcal_b:.1f} = {suma:.1f} kcal — matematica OK"
        )
    else:
        CASO_SUM_VERIFY.status  = STATUS_FAIL
        CASO_SUM_VERIFY.detalle = " | ".join(problemas)

    return CASO_SUM_VERIFY


# ══════════════════════════════════════════════════════════════════════════════
# Presentación de resultados
# ══════════════════════════════════════════════════════════════════════════════

def _print_caso_inline(caso: TestCase) -> None:
    """Imprime una línea de resultado para la tabla de resumen."""
    status_icon = {
        STATUS_PASS:  "OK  ",
        STATUS_FAIL:  "FAIL",
        STATUS_ERROR: "ERR ",
        STATUS_SKIP:  "SKIP",
    }.get(caso.status, "????")

    foods_str = ", ".join(caso.foods_real[:2]) if caso.foods_real else "—"
    if len(caso.foods_real) > 2:
        foods_str += f" (+{len(caso.foods_real)-2})"

    kcal_str = f"{caso.kcal_real:.1f}" if caso.kcal_real > 0 else "0"

    print(
        f"  [{status_icon}] {caso.test_id:<7} "
        f"| {caso.grupo:<15} "
        f"| kcal={kcal_str:<7} "
        f"| foods: {foods_str[:28]:<28} "
        f"| {caso.duracion_s:.1f}s"
    )


def _print_detalle_caso(caso: TestCase) -> None:
    """Imprime el bloque completo de diagnóstico de un caso."""
    print(f"\n  {SEP}")
    print(f"  [{caso.status}] {caso.test_id} — {caso.grupo}")
    print(f"  Mensaje   : \"{caso.mensaje}\"")
    print(f"  Descripcion: {caso.descripcion}")
    print(f"  kcal_real : {caso.kcal_real:.1f}")
    print(f"  Alimentos : {caso.foods_real}")
    print(f"  Duracion  : {caso.duracion_s:.2f}s")
    print(f"  Detalle   : {caso.detalle}")

    # Mostrar la respuesta del asistente si está disponible
    if caso.resultado:
        msg = caso.resultado.get("mensaje") or ""
        if msg:
            msg_short = msg[:160] + ("..." if len(msg) > 160 else "")
            print(f"  Respuesta : {msg_short}")


# ══════════════════════════════════════════════════════════════════════════════
# Main asíncrono
# ══════════════════════════════════════════════════════════════════════════════

async def main_async() -> int:
    """
    Ejecuta todos los casos de prueba.
    Retorna 0 si todos pasan, 1 si hay algún fallo.
    """
    from datetime import datetime
    print(f"\n{SEP2}")
    print("  QA — Motor de Registro de Alimentos CaloFit")
    print(f"  Hora   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Cliente: ID 55 (Carlos — Perfil A)")
    print(f"  Modo   : Directo al service layer (sin HTTP)")
    print(SEP2)

    # ── Obtener perfil de prueba ──────────────────────────────────────────────
    db = SessionLocal()
    try:
        perfil = db.query(Client).filter(Client.id == 55).first()
        if not perfil:
            print("  ERROR CRITICO: Cliente ID 55 no encontrado en la BD.")
            print("  Asegurate de haber ejecutado seed_perfiles.py primero.")
            return 1

        print(f"\n  Perfil cargado: {perfil.first_name} {perfil.last_name_paternal or ''}")
        print(f"  Email: {perfil.email}")
        print(f"  Peso: {perfil.weight}kg | Altura: {perfil.height}cm")

        # Plan del día mínimo (requerido por el handler)
        plan_hoy_data = {
            "calorias_dia":       2000,
            "proteinas_g":        150,
            "carbohidratos_g":    220,
            "grasas_g":            55,
        }

        # ── Ejecutar G1–G5 ────────────────────────────────────────────────────
        print(f"\n{SEP}")
        print("  EJECUTANDO CASOS G1–G5 (casos individuales + anti-fraude)")
        print(SEP)

        todos_los_casos: list[TestCase] = []

        for i, caso in enumerate(CASOS, 1):
            print(f"  [{i:02d}/{len(CASOS)}] {caso.test_id} — {caso.mensaje[:55]}...")
            await _ejecutar_caso(caso, perfil, plan_hoy_data, db)
            _print_caso_inline(caso)
            todos_los_casos.append(caso)

        # ── Ejecutar G6 (suma acumulativa) ────────────────────────────────────
        print(f"\n{SEP}")
        print("  EJECUTANDO GRUPO 6 — Suma acumulativa (2 registros separados)")
        print(SEP)

        print(f"  [18/20] {CASO_SUM_A.test_id} — {CASO_SUM_A.mensaje}")
        print(f"  [19/20] {CASO_SUM_B.test_id} — {CASO_SUM_B.mensaje}")

        await _run_grupo6(perfil, plan_hoy_data, db)

        _print_caso_inline(CASO_SUM_A)
        _print_caso_inline(CASO_SUM_B)
        todos_los_casos.append(CASO_SUM_A)
        todos_los_casos.append(CASO_SUM_B)

        print(f"\n  [20/20] {CASO_SUM_VERIFY.test_id} — {CASO_SUM_VERIFY.descripcion}")
        _print_caso_inline(CASO_SUM_VERIFY)
        todos_los_casos.append(CASO_SUM_VERIFY)

        # ── Tabla de resumen completa ─────────────────────────────────────────
        print(f"\n{SEP2}")
        print("  TABLA DE RESULTADOS DETALLADA")
        print(SEP2)
        print(
            f"  {'ID':<7} | {'GRUPO':<15} | {'MENSAJE (truncado)':<40} | "
            f"{'KCAL':>6} | {'N_FOODS':>7} | {'ESTADO':<5}"
        )
        print(f"  {'-'*7}-+-{'-'*15}-+-{'-'*40}-+-{'-'*6}-+-{'-'*7}-+-{'-'*5}")

        for caso in todos_los_casos:
            msg_t   = caso.mensaje[:38] + (".." if len(caso.mensaje) > 38 else "  ")
            kcal_s  = f"{caso.kcal_real:.1f}" if caso.kcal_real > 0 else "—"
            nfoods  = str(len(caso.foods_real)) if caso.foods_real else "—"
            print(
                f"  {caso.test_id:<7} | {caso.grupo:<15} | {msg_t:<40} | "
                f"{kcal_s:>6} | {nfoods:>7} | {caso.status}"
            )

        # ── Diagnóstico de casos fallidos ─────────────────────────────────────
        fallidos = [c for c in todos_los_casos if c.status in (STATUS_FAIL, STATUS_ERROR)]
        if fallidos:
            print(f"\n{SEP2}")
            print("  DIAGNOSTICO DE CASOS FALLIDOS")
            for caso in fallidos:
                _print_detalle_caso(caso)

        # ── Conteo final ──────────────────────────────────────────────────────
        total   = len(todos_los_casos)
        n_pass  = sum(1 for c in todos_los_casos if c.status == STATUS_PASS)
        n_fail  = sum(1 for c in todos_los_casos if c.status == STATUS_FAIL)
        n_error = sum(1 for c in todos_los_casos if c.status == STATUS_ERROR)
        n_skip  = sum(1 for c in todos_los_casos if c.status == STATUS_SKIP)

        print(f"\n{SEP2}")
        print("  RESUMEN FINAL")
        print(SEP)
        print(f"  Total de casos : {total}")
        print(f"  PASS           : {n_pass}")
        print(f"  FAIL           : {n_fail}")
        print(f"  ERROR          : {n_error}")
        print(f"  SKIP           : {n_skip}")
        print(SEP)

        if n_fail == 0 and n_error == 0:
            print(f"  RESULTADO: TODOS LOS CASOS PASARON ({n_pass}/{total})")
            exit_code = 0
        else:
            print(f"  RESULTADO: {n_fail + n_error} CASO(S) FALLARON — revisar diagnostico arriba")
            exit_code = 1

        print(SEP2)
        return exit_code

    finally:
        db.close()


def main() -> None:
    exit_code = asyncio.run(main_async())
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
