"""
Batería de DIAGNÓSTICO (no de hardening) para reproducir bugs reportados por
usuarios reales del registro nutricional conversacional. NO mockea Groq a
propósito — el objetivo es ver el comportamiento real exacto que vieron los
usuarios, no una simulación. Por eso estas pruebas son más lentas y pueden
fallar por motivos ajenos al código (cuota de Groq) — eso también es una
señal diagnóstica válida, se reporta como tal.

No se aplican fixes aquí. Cada test documenta resultado esperado vs actual
y, si falla, la causa probable (extracción LLM / validación posterior /
caché / cálculo de macros / lógica determinista).

UBICACIÓN (tests/external/): depende de la API real de Groq — está excluida
de la corrida por defecto (`pytest tests/` usa `-m "not external"`, ver
tests/pytest.ini). Ejecutar a mano con:
    pytest tests/external/test_auditoria_registro_real.py -v
"""
import asyncio

import pytest

from app.core.utils import get_peru_date
from app.services.llm_registro import registrar_comida_llm, _macro_cache
from app.services.ia_service import ia_engine
from app.models.comida_registro import ComidaRegistro


def _limpiar_cache():
    _macro_cache.clear()


def _normalizar_lista(nombres: list[str]) -> list[str]:
    return [n.lower().strip() for n in nombres]


async def _registrar(mensaje: str, perfil, plan_hoy: dict, db) -> dict:
    """Ejecuta registrar_comida_llm contra Groq REAL y empaqueta el
    diagnóstico relevante en un dict plano para imprimir/comparar."""
    resultado = await registrar_comida_llm(mensaje, perfil, plan_hoy, db, ia_engine)
    filas = (
        db.query(ComidaRegistro)
        .filter(ComidaRegistro.client_id == perfil.id, ComidaRegistro.fecha == get_peru_date())
        .order_by(ComidaRegistro.id.desc())
        .all()
    )
    diag = {
        "mensaje_usuario": mensaje,
        "success": resultado.get("success"),
        "tipo_detectado": resultado.get("tipo_detectado"),
        "alimentos": resultado.get("alimentos", []),
        "kcal": (resultado.get("datos") or {}).get("calorias"),
        "prot_g": (resultado.get("datos") or {}).get("proteinas_g"),
        "carb_g": (resultado.get("datos") or {}).get("carbohidratos_g"),
        "grasa_g": (resultado.get("datos") or {}).get("grasas_g"),
        "mensaje_respuesta": resultado.get("mensaje"),
        "filas_comida_registro_recientes": [
            (f.nombre_alimento, f.kcal) for f in filas[: max(1, len(resultado.get("alimentos", [])) * 3)]
        ],
    }
    print("\n" + "=" * 90)
    print(f"MENSAJE: {mensaje!r}")
    for k, v in diag.items():
        if k == "mensaje_usuario":
            continue
        print(f"  {k}: {v}")
    print("=" * 90)
    return diag


@pytest.fixture
def plan_hoy():
    return {"calorias_dia": 2616, "proteinas_g": 160, "carbohidratos_g": 240, "grasas_g": 60}


@pytest.fixture(autouse=True)
def _aislar_cache_por_test():
    """Aislamiento explícito además del autouse global de conftest.py —
    redundante a propósito: este archivo necesita controlar el momento
    exacto de limpieza dentro de cada test (ver Caso 4)."""
    _limpiar_cache()
    yield
    _limpiar_cache()


@pytest.mark.integration
@pytest.mark.external
class TestAuditoriaRegistroReal:

    # ════════════════════════════════════════════════════════════════════
    # CASO 1 — Plato compuesto tipo contenedor + ingredientes
    # ════════════════════════════════════════════════════════════════════
    @pytest.mark.asyncio
    async def test_caso1_batido_contenedor_e_ingredientes(self, db, sample_client, plan_hoy):
        diag = await _registrar(
            "Me hice un batido de avena, leche, dos plátanos, miel",
            sample_client, plan_hoy, db,
        )
        nombres_norm = _normalizar_lista(diag["alimentos"])
        tiene_contenedor = any("batido" in n for n in nombres_norm)
        ingredientes_sueltos = [
            n for n in nombres_norm
            if any(ing in n for ing in ("avena", "leche", "plátano", "platano", "miel"))
        ]
        duplicado = tiene_contenedor and len(ingredientes_sueltos) >= 2

        assert not duplicado, (
            f"BUG CONFIRMADO (Caso 1): se registró el contenedor 'Batido' Y "
            f"{len(ingredientes_sueltos)} de sus ingredientes por separado en la misma "
            f"extracción -> doble/triple conteo de kcal. alimentos={diag['alimentos']} "
            f"kcal_total={diag['kcal']}"
        )

    # ════════════════════════════════════════════════════════════════════
    # CASO 2 — Alimento inexistente / alucinación ("umas")
    # ════════════════════════════════════════════════════════════════════
    @pytest.mark.asyncio
    async def test_caso2_alimento_inexistente_umas(self, db, sample_client, plan_hoy):
        diag = await _registrar(
            "Comí tres umas y dos tazas de café", sample_client, plan_hoy, db,
        )
        nombres_norm = _normalizar_lista(diag["alimentos"])
        tiene_umas = any("uma" in n for n in nombres_norm)

        assert not tiene_umas, (
            f"BUG CONFIRMADO (Caso 2): 'umas' (no es un alimento real) se registró "
            f"con macros inventados. alimentos={diag['alimentos']} kcal_total={diag['kcal']} "
            f"prot_g={diag['prot_g']}"
        )

    # ════════════════════════════════════════════════════════════════════
    # CASO 3 — Palabras de momento del día registradas como comida
    # ════════════════════════════════════════════════════════════════════
    @pytest.mark.asyncio
    @pytest.mark.parametrize("mensaje", [
        "Hoy en el desayuno comí avena",
        "Registré mi almuerzo",
    ])
    async def test_caso3_momento_dia_no_es_alimento(self, db, sample_client, plan_hoy, mensaje):
        diag = await _registrar(mensaje, sample_client, plan_hoy, db)
        nombres_norm = _normalizar_lista(diag["alimentos"])
        _MOMENTOS = ("desayuno", "almuerzo", "cena", "merienda")
        momentos_como_alimento = [n for n in nombres_norm if any(m in n for m in _MOMENTOS)]

        assert not momentos_como_alimento, (
            f"BUG CONFIRMADO (Caso 3): la palabra de momento del día quedó como "
            f"alimento registrado: {momentos_como_alimento}. mensaje={mensaje!r} "
            f"alimentos={diag['alimentos']}"
        )

    # ════════════════════════════════════════════════════════════════════
    # CASO 4 — Consistencia del mismo mensaje (con y sin caché entre llamadas)
    # ════════════════════════════════════════════════════════════════════
    @pytest.mark.asyncio
    async def test_caso4a_consistencia_sin_limpiar_cache_entre_llamadas(self, db, sample_client, plan_hoy):
        """Replica exactamente lo que pasa en la app real: el caché de macros
        NO se limpia entre mensajes de una misma sesión."""
        mensaje = "Comí tres umas y dos tazas de café"
        diag1 = await _registrar(mensaje, sample_client, plan_hoy, db)
        await asyncio.sleep(2)
        diag2 = await _registrar(mensaje, sample_client, plan_hoy, db)

        set1, set2 = set(_normalizar_lista(diag1["alimentos"])), set(_normalizar_lista(diag2["alimentos"]))
        kcal1, kcal2 = diag1["kcal"] or 0, diag2["kcal"] or 0
        diff_pct = abs(kcal1 - kcal2) / max(kcal1, kcal2, 1) * 100

        assert set1 == set2, (
            f"BUG CONFIRMADO (Caso 4a, sin limpiar caché): el mismo mensaje extrajo "
            f"alimentos distintos en las 2 llamadas -> 1ra: {diag1['alimentos']} | "
            f"2da: {diag2['alimentos']}. Causa probable: _macro_cache contaminado "
            f"por la 1ra llamada altera la 2da (cache hit parcial/inesperado)."
        )
        assert diff_pct < 20, (
            f"BUG CONFIRMADO (Caso 4a, sin limpiar caché): mismo mensaje, kcal "
            f"distinta en {diff_pct:.0f}% -> 1ra: {kcal1} kcal | 2da: {kcal2} kcal."
        )

    @pytest.mark.asyncio
    async def test_caso4b_consistencia_limpiando_cache_entre_llamadas(self, db, sample_client, plan_hoy):
        """Mismo mensaje, pero con _macro_cache limpio antes de cada llamada —
        aísla si la inconsistencia es del caché o del LLM en sí (temp=0.0
        debería ser casi determinista, pero la API real no lo garantiza al 100%)."""
        mensaje = "Comí tres umas y dos tazas de café"
        _limpiar_cache()
        diag1 = await _registrar(mensaje, sample_client, plan_hoy, db)
        await asyncio.sleep(2)
        _limpiar_cache()
        diag2 = await _registrar(mensaje, sample_client, plan_hoy, db)

        set1, set2 = set(_normalizar_lista(diag1["alimentos"])), set(_normalizar_lista(diag2["alimentos"]))
        kcal1, kcal2 = diag1["kcal"] or 0, diag2["kcal"] or 0
        diff_pct = abs(kcal1 - kcal2) / max(kcal1, kcal2, 1) * 100

        assert set1 == set2, (
            f"BUG CONFIRMADO (Caso 4b, CON caché limpio): el mismo mensaje extrajo "
            f"alimentos distintos en las 2 llamadas incluso con caché limpio -> "
            f"1ra: {diag1['alimentos']} | 2da: {diag2['alimentos']}. "
            f"Causa probable: no es el caché, es no-determinismo real del LLM/extractor."
        )
        assert diff_pct < 20, (
            f"BUG CONFIRMADO (Caso 4b, CON caché limpio): mismo mensaje, kcal "
            f"distinta en {diff_pct:.0f}% -> 1ra: {kcal1} kcal | 2da: {kcal2} kcal. "
            f"Causa probable: no-determinismo del LLM, no el caché."
        )

    # ════════════════════════════════════════════════════════════════════
    # CASO 5 — Gramaje explícito → piso mínimo de macros razonable
    # ════════════════════════════════════════════════════════════════════
    @pytest.mark.asyncio
    async def test_caso5_gramaje_explicito_macros_razonables(self, db, sample_client, plan_hoy):
        diag = await _registrar(
            "arroz y cerdo 200gr + ensalada con medio pepinillo + media palta",
            sample_client, plan_hoy, db,
        )
        kcal = diag["kcal"] or 0
        prot = diag["prot_g"] or 0
        # Pisos conservadores: 200g de arroz cocido (~130kcal/100g) + 200g de
        # cerdo (mínimo realista ~150kcal/100g en corte magro) ya dan >550kcal
        # y >25g de proteína solo entre esos dos, sin contar palta/pepinillo.
        assert kcal >= 400, (
            f"BUG CONFIRMADO (Caso 5): 200g arroz + 200g cerdo + palta + pepinillo "
            f"dieron solo {kcal} kcal — muy por debajo del piso realista (>=400). "
            f"Causa probable: cálculo de macros del LLM subestima cantidades con "
            f"gramaje explícito. alimentos={diag['alimentos']}"
        )
        assert prot >= 15, (
            f"BUG CONFIRMADO (Caso 5): proteína total {prot}g muy baja para 200g "
            f"de cerdo (mínimo realista >=15g solo del cerdo). alimentos={diag['alimentos']}"
        )

    # ════════════════════════════════════════════════════════════════════
    # CASO 6 — Multiplicador de cantidad
    # ════════════════════════════════════════════════════════════════════
    @pytest.mark.asyncio
    async def test_caso6_multiplicador_cantidad_dos_mandarinas(self, db, sample_client, plan_hoy):
        diag = await _registrar("comí dos mandarinas", sample_client, plan_hoy, db)
        nombres_norm = _normalizar_lista(diag["alimentos"])
        tiene_multiplicador = any("×2" in n or "x2" in n for n in nombres_norm)

        filas_mandarina = [
            f for f in diag["filas_comida_registro_recientes"]
            if "mandarin" in f[0].lower()
        ]

        assert tiene_multiplicador or len(filas_mandarina) == 2, (
            f"BUG CONFIRMADO (Caso 6): 'dos mandarinas' no se refleja como x2 — "
            f"alimentos={diag['alimentos']}, filas en comida_registros={filas_mandarina} "
            f"(se esperaban 2 filas de Mandarina o un nombre con '×2')."
        )
