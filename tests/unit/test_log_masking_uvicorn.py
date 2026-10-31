"""El enmascarado de logs no debe romper el log de acceso de uvicorn (AccessFormatter desempaqueta record.args)."""
import logging

from uvicorn.logging import AccessFormatter

from app.core.log_masking import FiltroEnmascarado


def _registrar_acceso(ruta):
    errores, salida = [], []

    class Manejador(logging.Handler):
        def emit(self, record):
            try:
                salida.append(self.format(record))
            except Exception as e:  # lo que en producción sale como "--- Logging error ---"
                errores.append(e)

    h = Manejador()
    h.setFormatter(AccessFormatter('%(client_addr)s - "%(request_line)s" %(status_code)s', use_colors=False))
    h.addFilter(FiltroEnmascarado())
    lg = logging.Logger("uvicorn.access")
    lg.addHandler(h)
    lg.info('%s - "%s %s HTTP/%s" %d', "10.31.107.140:0", "GET", ruta, "1.1", 200)
    return errores, salida


def test_log_de_acceso_se_formatea_sin_error():
    errores, salida = _registrar_acceso("/dashboard/clientes/1/resumen-diario")
    assert not errores, errores
    assert salida == ['10.31.107.140:0 - "GET /dashboard/clientes/1/resumen-diario HTTP/1.1" 200 OK']


def test_log_de_acceso_sigue_enmascarando_correos_en_la_ruta():
    errores, salida = _registrar_acceso("/auth/x?email=ana.perez@gmail.com")
    assert not errores, errores
    assert "ana.perez@gmail.com" not in salida[0] and "a***@gmail.com" in salida[0]
