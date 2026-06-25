"""
Prueba del fallback cuando Groq responde algo que no tiene formato de plato
reconocible (ni bullets con macros, ni bullets simples) — encontrado en la
auditoría final pre-demo: ese texto se devolvía tal cual al usuario en vez de
caer al fallback determinístico. Ver llm_registro.py::respuesta_recomendacion_llm,
paso 7 (fallback final).
"""
import pytest
from unittest.mock import patch

from app.services.llm_registro import respuesta_recomendacion_llm, _FALLBACKS_OPCIONES
from app.services.ia_service import ia_engine


@pytest.mark.integration
class TestFallbackRespuestaLLMInvalida:

    @pytest.mark.asyncio
    async def test_respuesta_no_parseable_cae_a_fallback_determinista(self, sample_client):
        async def mock_llamar_groq(prompt, max_tokens=800, temp=0.7, model=None):
            # Ni JSON, ni bullets "- Nombre (~XXX kcal, P:Xg C:Yg G:Zg)", ni
            # error reconocido (sin "timeout"/"429"/"[Error:") — texto roto
            # cualquiera que el parser de platos no puede usar.
            return "{esto no es json valido, falta cierre"

        with patch.object(ia_engine, "_llamar_groq", new=mock_llamar_groq):
            resultado = await respuesta_recomendacion_llm(
                mensaje="¿qué puedo comer?",
                perfil=sample_client,
                consumido=500.0,
                meta=2000.0,
                quemado=0.0,
                ia_engine=ia_engine,
                modo="comida",
                historial=[],
                db=None,
            )

        assert "esto no es json valido" not in resultado
        assert "falta cierre" not in resultado
        # Debe ser exactamente uno de los fallbacks deterministas conocidos.
        todas_las_opciones = [
            opcion for lista in _FALLBACKS_OPCIONES.values() for opcion in lista
        ]
        assert any(
            _texto_plato_en(opcion) in resultado for opcion in todas_las_opciones
        ), f"La respuesta no coincide con ningún fallback conocido: {resultado!r}"


def _texto_plato_en(opcion_fallback: str) -> str:
    """'- Fruta picada con chía (~90 kcal, P:2g C:20g G:1g)' -> 'Fruta picada con chía'"""
    return opcion_fallback.split("(~")[0].lstrip("- ").strip()
