"""
Generador de fingerprints SHA-256 deterministas para platos, rutinas y alimentos.
"""
import hashlib
import json
from typing import Dict, Any, List


class FingerprintGenerator:
    """
    Genera fingerprints SHA-256 reproducibles.
    El mismo contenido siempre produce el mismo hash;
    cualquier cambio en ingredientes/macros produce un hash diferente.
    """

    @staticmethod
    def generar_fingerprint_plato(
        nombre: str,
        ingredientes: List[Dict[str, Any]],
        macros: Dict[str, float],
    ) -> str:
        datos = {
            "nombre": nombre.lower().strip(),
            "ingredientes": sorted(
                ingredientes, key=lambda x: x.get("nombre", "").lower()
            ),
            "macros": {k: round(v, 2) if isinstance(v, float) else v
                       for k, v in macros.items()},
        }
        return _sha256(datos)

    @staticmethod
    def generar_fingerprint_rutina(
        nombre: str,
        ejercicios: List[Dict[str, Any]],
        intensidad: str,
    ) -> str:
        datos = {
            "nombre": nombre.lower().strip(),
            "ejercicios": sorted(ejercicios, key=lambda x: str(x.get("ejercicio_id", ""))),
            "intensidad": intensidad.lower(),
        }
        return _sha256(datos)

    @staticmethod
    def generar_fingerprint_alimento(
        nombre: str,
        calorias_100g: float,
        proteina_100g: float,
        carbohidratos_100g: float,
        grasas_100g: float,
        source: str,
    ) -> str:
        datos = {
            "nombre": nombre.lower().strip(),
            "macros_100g": {
                "calorias":       round(calorias_100g, 2),
                "proteina":       round(proteina_100g, 2),
                "carbohidratos":  round(carbohidratos_100g, 2),
                "grasas":         round(grasas_100g, 2),
            },
            "source": source.lower(),
        }
        return _sha256(datos)

    @staticmethod
    def comparar(fp1: str, fp2: str) -> bool:
        return fp1 == fp2

    @staticmethod
    def comparar_fingerprints(fp1: str, fp2: str) -> bool:
        """Alias de comparar."""
        return fp1 == fp2

    @staticmethod
    def detectar_cambio(fp_viejo: str, fp_nuevo: str) -> Dict[str, Any]:
        cambio = fp_viejo != fp_nuevo
        return {
            "cambio_detectado": cambio,
            "fingerprint_viejo": fp_viejo,
            "fingerprint_nuevo": fp_nuevo,
            "mensaje": "Contenido cambió" if cambio else "Sin cambios",
        }


def _sha256(datos: Any) -> str:
    raw = json.dumps(datos, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
