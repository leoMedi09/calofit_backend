"""
Casos A-G: los filtros nuevos (contenedor genérico, alucinaciones, momento
del día) no deben romper registros normales del asistente conversacional.
NO mockea Groq a propósito — el objetivo es confirmar el comportamiento real
de producción, no una simulación.

UBICACIÓN (tests/external/): depende de la API real de Groq — separada del
resto de tests/integration/test_regresion_post_fixes.py (que sí mockea) y
excluida de la corrida por defecto (`pytest tests/` usa -m "not external",
ver tests/pytest.ini). Ejecutar a mano con:
    pytest tests/external/test_filtros_no_rompen_registros_normales.py -v
"""
import pytest

from app.core.utils import get_peru_date
from app.services.llm_registro import registrar_comida_llm, _macro_cache
from app.services.ia_service import ia_engine
from app.models.comida_registro import ComidaRegistro


def _filas_de_hoy(db, client_id):
    return (
        db.query(ComidaRegistro)
        .filter(ComidaRegistro.client_id == client_id, ComidaRegistro.fecha == get_peru_date())
        .all()
    )


async def _registrar_real(mensaje: str, perfil, plan_hoy: dict, db) -> dict:
    """Registra contra Groq REAL (sin mock) y devuelve un diagnóstico plano."""
    resultado = await registrar_comida_llm(mensaje, perfil, plan_hoy, db, ia_engine)
    filas = _filas_de_hoy(db, perfil.id)
    diag = {
        "mensaje_usuario": mensaje,
        "success": resultado.get("success"),
        "tipo_detectado": resultado.get("tipo_detectado"),
        "alimentos": resultado.get("alimentos", []),
        "kcal": (resultado.get("datos") or {}).get("calorias"),
        "mensaje_respuesta": resultado.get("mensaje"),
        "filas": [(f.nombre_alimento, f.kcal) for f in filas],
    }
    print("\n" + "=" * 90)
    print(f"MENSAJE: {mensaje!r}")
    for k, v in diag.items():
        if k != "mensaje_usuario":
            print(f"  {k}: {v}")
    print("=" * 90)
    return diag


@pytest.fixture
def plan_hoy():
    return {"calorias_dia": 2616, "proteinas_g": 160, "carbohidratos_g": 240, "grasas_g": 60}


@pytest.fixture(autouse=True)
def _limpiar_cache_explicito():
    _macro_cache.clear()
    yield
    _macro_cache.clear()


@pytest.mark.integration
@pytest.mark.external
class TestFiltrosNoRompenRegistrosNormales:

    @pytest.mark.asyncio
    async def test_caso_a_batido_de_avena_sin_duplicar(self, db, sample_client, plan_hoy):
        diag = await _registrar_real("Comí un batido de avena", sample_client, plan_hoy, db)
        assert diag["success"] is True
        nombres_norm = [n.lower() for n in diag["alimentos"]]
        # No debe haber MÁS de un ítem que mencione "avena" — si "avena" y
        # "batido de avena" coexisten, son la misma cosa contada 2 veces.
        items_avena = [n for n in nombres_norm if "avena" in n or "batido" in n]
        assert len(items_avena) <= 1, (
            f"Posible doble conteo de avena/batido: {diag['alimentos']} ({diag['kcal']} kcal)"
        )

    @pytest.mark.asyncio
    async def test_caso_b_arroz_con_pollo_no_se_elimina_mal(self, db, sample_client, plan_hoy):
        diag = await _registrar_real("Comí arroz con pollo", sample_client, plan_hoy, db)
        assert diag["success"] is True
        nombres_norm = [n.lower() for n in diag["alimentos"]]
        # Debe quedar UN plato que mencione ambos, o ambos ingredientes
        # presentes en algún lugar de la lista — nunca deben perderse los dos.
        menciona_arroz = any("arroz" in n for n in nombres_norm)
        menciona_pollo = any("pollo" in n for n in nombres_norm)
        assert menciona_arroz and menciona_pollo, (
            f"Se perdió arroz o pollo del registro: {diag['alimentos']}"
        )
        assert len(diag["alimentos"]) == 1, (
            f"'Arroz con pollo' no debería duplicarse en varios ítems: {diag['alimentos']}"
        )

    @pytest.mark.asyncio
    async def test_caso_c_ensalada_no_se_descarta_sin_duplicados(self, db, sample_client, plan_hoy):
        diag = await _registrar_real(
            "Comí ensalada de pollo con palta", sample_client, plan_hoy, db,
        )
        assert diag["success"] is True
        nombres_norm = [n.lower() for n in diag["alimentos"]]
        # Si pollo/palta NO están duplicados como ítems sueltos junto a un
        # nombre de ensalada que YA los menciona, no debe perderse información:
        # debe quedar O bien "ensalada..." con pollo/palta en el nombre, O
        # bien pollo Y palta como ítems propios (nunca los 3 a la vez, eso
        # sería duplicado real).
        assert len(diag["alimentos"]) <= 2, (
            f"Demasiados ítems para una sola ensalada — posible duplicado: {diag['alimentos']}"
        )
        texto_unido = " ".join(nombres_norm)
        assert "pollo" in texto_unido and "palta" in texto_unido, (
            f"Se perdió pollo o palta: {diag['alimentos']}"
        )

    @pytest.mark.asyncio
    async def test_caso_d_avena_no_se_elimina_por_palabra_desayuno(self, db, sample_client, plan_hoy):
        diag = await _registrar_real("Hoy desayuné avena", sample_client, plan_hoy, db)
        assert diag["success"] is True
        nombres_norm = [n.lower() for n in diag["alimentos"]]
        assert any("avena" in n for n in nombres_norm), (
            f"'avena' no quedó registrada: {diag['alimentos']}"
        )
        assert not any(n.strip() == "desayuno" for n in nombres_norm), (
            f"'desayuno' quedó como ítem aparte: {diag['alimentos']}"
        )

    @pytest.mark.asyncio
    async def test_caso_e_almuerzo_no_crea_alimento(self, db, sample_client, plan_hoy):
        diag = await _registrar_real("Registré mi almuerzo", sample_client, plan_hoy, db)
        assert diag["success"] is False
        assert diag["tipo_detectado"] == "no_identificado"
        assert len(_filas_de_hoy(db, sample_client.id)) == 0

    @pytest.mark.asyncio
    async def test_caso_f_sushi_no_se_rechaza_por_no_estar_en_bd_local(self, db, sample_client, plan_hoy):
        diag = await _registrar_real("Comí sushi", sample_client, plan_hoy, db)
        assert diag["success"] is True, (
            f"'sushi' fue rechazado aunque es un alimento real (solo no está en la BD "
            f"local INS/CENAN) — el LLM sí le da macros propios: {diag}"
        )
        assert any("sushi" in n.lower() for n in diag["alimentos"])

    @pytest.mark.asyncio
    async def test_caso_g_cafe_se_acepta_aunque_tenga_pocas_kcal(self, db, sample_client, plan_hoy):
        diag = await _registrar_real("Comí café", sample_client, plan_hoy, db)
        assert diag["success"] is True, (
            f"'café' fue rechazado pese a ser una bebida real de ~0 kcal: {diag}"
        )
        assert any("caf" in n.lower() for n in diag["alimentos"])
