"""
Checker de consistencia: detecta anomalías en los datos de la BD.
"""
from typing import Dict, Any, List
from sqlalchemy.orm import Session
from app.services.validators.base_validator import BaseValidator, ValidationResult
from app.models import Alimento, Plato, Ejercicio
from app.models.routine_models import Rutina


class ConsistencyChecker(BaseValidator):
    """
    Detecta inconsistencias en la BD para alimentos, platos, ejercicios y rutinas.

    Entrada (datos):
        tipo : str  — "alimentos" | "platos" | "ejercicios" | "rutinas" | "completo"
    """

    def __init__(self, db: Session) -> None:
        super().__init__("ConsistencyChecker")
        self.db = db

    def validar(self, datos: Dict[str, Any]) -> ValidationResult:
        tipo = datos.get("tipo", "completo")

        errores:     List[str] = []
        advertencias: List[str] = []

        if tipo in ("alimentos", "completo"):
            r = self._chequear_alimentos()
            errores.extend(r["errores"])
            advertencias.extend(r["advertencias"])

        if tipo in ("platos", "completo"):
            r = self._chequear_platos()
            errores.extend(r["errores"])
            advertencias.extend(r["advertencias"])

        if tipo in ("ejercicios", "completo"):
            r = self._chequear_ejercicios()
            errores.extend(r["errores"])
            advertencias.extend(r["advertencias"])

        if tipo in ("rutinas", "completo"):
            r = self._chequear_rutinas()
            errores.extend(r["errores"])
            advertencias.extend(r["advertencias"])

        return self._crear_resultado(
            es_valido=len(errores) == 0,
            confianza=max(0, 100 - len(advertencias) * 5 - len(errores) * 15),
            errores=errores,
            advertencias=advertencias,
            metadata={
                "tipo_chequeo":       tipo,
                "total_errores":      len(errores),
                "total_advertencias": len(advertencias),
            },
        )

    # ─── Checks individuales ─────────────────────────────────────────────────

    def _chequear_alimentos(self) -> Dict[str, List[str]]:
        errores:     List[str] = []
        advertencias: List[str] = []

        for ali in self.db.query(Alimento).all():
            if not ali.calorias_100g:
                errores.append(f"Alimento sin calorías: '{ali.nombre}'")
                continue
            if not (0 < ali.calorias_100g <= 900):
                advertencias.append(
                    f"Alimento '{ali.nombre}': kcal/100g={ali.calorias_100g} fuera de rango (0-900)"
                )
            if (not ali.proteina_100g and not ali.carbohidratos_100g and not ali.grasas_100g):
                advertencias.append(f"Alimento '{ali.nombre}' sin macros (P/C/G todos 0)")

        return {"errores": errores, "advertencias": advertencias}

    def _chequear_platos(self) -> Dict[str, List[str]]:
        errores:     List[str] = []
        advertencias: List[str] = []

        for plato in self.db.query(Plato).all():
            n_ingr = len(plato.ingredientes) if plato.ingredientes else 0
            if n_ingr == 0:
                errores.append(f"Plato sin ingredientes: '{plato.nombre}'")
            elif n_ingr > 15:
                advertencias.append(
                    f"Plato '{plato.nombre}' tiene {n_ingr} ingredientes (inusualmente alto)"
                )

        return {"errores": errores, "advertencias": advertencias}

    def _chequear_ejercicios(self) -> Dict[str, List[str]]:
        errores:     List[str] = []
        advertencias: List[str] = []

        for ej in self.db.query(Ejercicio).all():
            if ej.met is None or not (0 < ej.met <= 20):
                advertencias.append(
                    f"Ejercicio '{ej.nombre}': MET={ej.met} fuera de rango (0-20)"
                )
            if not ej.grupo_muscular:
                advertencias.append(f"Ejercicio '{ej.nombre}' sin grupo muscular")

        return {"errores": errores, "advertencias": advertencias}

    def _chequear_rutinas(self) -> Dict[str, List[str]]:
        errores:     List[str] = []
        advertencias: List[str] = []

        for rutina in self.db.query(Rutina).all():
            n_ej = len(rutina.ejercicios) if rutina.ejercicios else 0
            if n_ej == 0:
                errores.append(f"Rutina sin ejercicios: '{rutina.nombre}'")
            if rutina.tiempo_min and not (5 <= rutina.tiempo_min <= 300):
                advertencias.append(
                    f"Rutina '{rutina.nombre}': tiempo_min={rutina.tiempo_min} fuera de rango (5-300)"
                )

        return {"errores": errores, "advertencias": advertencias}
