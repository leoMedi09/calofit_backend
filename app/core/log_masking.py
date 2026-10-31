"""Enmascara datos personales y secretos en TODO lo que la aplicacion escribe a consola/logs.

Cubre los print() y los logging de cualquier modulo o libreria: correos (a***@dominio), tokens JWT, cabeceras
Bearer y valores de password/token/api_key. Es una red de seguridad: no sustituye evitar imprimir esos datos."""
import logging
import re
import sys

_EMAIL = re.compile(r"([A-Za-z0-9])[A-Za-z0-9._%+-]*@([A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,})")
_JWT = re.compile(r"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}")
_BEARER = re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]{12,}")
_CLAVE = re.compile(r"(?i)((?:password|passwd|contrase[nñ]a|secret|api[_-]?key|access_token|refresh_token|token)['\"]?\s*[:=]\s*['\"]?)([^\s,'\"})&]{4,})")


def enmascarar(texto):
    if not isinstance(texto, str) or not texto:
        return texto
    texto = _JWT.sub("[JWT]", texto)
    texto = _BEARER.sub(r"\1[OCULTO]", texto)
    texto = _CLAVE.sub(r"\1[OCULTO]", texto)
    return _EMAIL.sub(r"\1***@\2", texto)


class FiltroEnmascarado(logging.Filter):
    def filter(self, record):
        try:
            if record.name == "uvicorn.access" and isinstance(record.args, tuple):
                # uvicorn desempaqueta record.args (5 valores) para armar la línea: no se pueden vaciar, se enmascara cada uno
                record.args = tuple(enmascarar(a) if isinstance(a, str) else a for a in record.args)
                return True
            record.msg = enmascarar(record.getMessage())
            record.args = ()
        except Exception:
            pass
        return True


class _FlujoEnmascarado:
    """Envuelve stdout/stderr: todo lo que se escribe pasa por enmascarar()."""

    def __init__(self, flujo):
        self._flujo = flujo

    def write(self, s):
        self._flujo.write(enmascarar(s))
        return len(s) if isinstance(s, str) else 0

    def __getattr__(self, nombre):
        return getattr(self._flujo, nombre)


def instalar():
    """Idempotente. Llamar al arrancar la aplicacion."""
    for nombre in ("stdout", "stderr"):
        actual = getattr(sys, nombre)
        if not isinstance(actual, _FlujoEnmascarado):
            setattr(sys, nombre, _FlujoEnmascarado(actual))
    filtro = FiltroEnmascarado()
    for logger in [logging.getLogger()] + [logging.getLogger(n) for n in ("uvicorn", "uvicorn.error", "uvicorn.access")]:
        for h in logger.handlers:
            if not any(isinstance(f, FiltroEnmascarado) for f in h.filters):
                h.addFilter(filtro)
