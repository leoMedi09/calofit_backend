"""
Servicio LLM centralizado — wrapper sobre Groq/Llama-3.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)

try:
    from groq import AsyncGroq
    _groq_available = True
except ImportError:
    AsyncGroq = None
    _groq_available = False

_DEFAULT_MODEL = "llama-3.1-8b-instant"
_DEFAULT_TEMP   = 0.3
_DEFAULT_TOKENS = 512


class LLMService:
    """
    Wrapper asíncrono sobre la API de Groq (Llama-3).

    Uso:
        llm = LLMService()
        texto = await llm.completar("¿Qué comer en el desayuno?")
        data  = await llm.generar_json("Dame 3 ingredientes...", schema_hint="lista")
    """

    def __init__(self) -> None:
        if not _groq_available:
            raise RuntimeError("groq SDK no instalado. Ejecutar: pip install groq")
        self._client = AsyncGroq(api_key=settings.GROQ_API_KEY)

    # ──────────────────────────────────────────────────────────────────
    # API pública
    # ──────────────────────────────────────────────────────────────────

    async def completar(
        self,
        prompt: str,
        system: str = "",
        model: str = _DEFAULT_MODEL,
        temperature: float = _DEFAULT_TEMP,
        max_tokens: int = _DEFAULT_TOKENS,
    ) -> str:
        """Genera texto libre."""
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            resp = await self._client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:
            logger.error("LLMService.completar error: %s", exc)
            return ""

    async def generar_json(
        self,
        prompt: str,
        system: str = "",
        model: str = _DEFAULT_MODEL,
        temperature: float = 0.05,
        max_tokens: int = 600,
    ) -> Optional[Any]:
        """
        Genera una respuesta y la parsea como JSON.
        Retorna el objeto parseado, o None si el JSON es inválido.
        """
        system_json = (system + "\nResponde SOLO con JSON válido, sin texto adicional.").strip()
        raw = await self.completar(
            prompt=prompt,
            system=system_json,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return self._parsear_json(raw)

    async def analizar_intencion(
        self,
        mensaje: str,
        opciones: List[str],
        model: str = _DEFAULT_MODEL,
    ) -> str:
        """
        Clasifica el mensaje en una de las opciones dadas.
        Retorna la opción con mayor probabilidad o la primera opción como fallback.
        """
        opciones_str = " | ".join(opciones)
        system = (
            f"Clasifica el mensaje del usuario en UNA de estas categorías: {opciones_str}. "
            "Responde SOLO con el nombre exacto de la categoría, sin explicación."
        )
        resultado = await self.completar(
            prompt=mensaje,
            system=system,
            temperature=0.0,
            max_tokens=32,
        )
        resultado = resultado.strip().lower()
        for op in opciones:
            if op.lower() in resultado:
                return op
        return opciones[0]

    # ──────────────────────────────────────────────────────────────────
    # Helpers privados
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _parsear_json(texto: str) -> Optional[Any]:
        """Intenta parsear JSON del texto, extrayendo bloques ```json``` si existen."""
        import re
        # Extraer bloque ```json ... ```
        bloque = re.search(r"```(?:json)?\s*([\s\S]*?)```", texto)
        candidato = bloque.group(1).strip() if bloque else texto.strip()
        try:
            return json.loads(candidato)
        except json.JSONDecodeError:
            # Intentar limpiar texto extra antes/después del JSON
            inicio = candidato.find("{") if "{" in candidato else candidato.find("[")
            if inicio != -1:
                try:
                    return json.loads(candidato[inicio:])
                except json.JSONDecodeError:
                    pass
        logger.warning("LLMService: no se pudo parsear JSON: %.80s", texto)
        return None
