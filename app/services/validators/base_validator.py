"""
Clase base para todos los validadores.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from datetime import datetime


class ValidationResult(BaseModel):
    es_valido: bool
    confianza: int  # 0-100
    errores: List[str] = []
    advertencias: List[str] = []
    sugerencias: List[str] = []
    metadata: Dict[str, Any] = {}
    timestamp: Optional[datetime] = None

    def model_post_init(self, __context: Any) -> None:
        if self.timestamp is None:
            object.__setattr__(self, "timestamp", datetime.utcnow())

    def __repr__(self) -> str:
        return (
            f"<ValidationResult(valido={self.es_valido}, "
            f"confianza={self.confianza}, errores={len(self.errores)})>"
        )


class BaseValidator(ABC):
    def __init__(self, nombre: str) -> None:
        self.nombre = nombre

    @abstractmethod
    def validar(self, datos: Dict[str, Any]) -> ValidationResult:
        pass

    def _crear_resultado(
        self,
        es_valido: bool,
        confianza: int = 100,
        errores: Optional[List[str]] = None,
        advertencias: Optional[List[str]] = None,
        sugerencias: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        return ValidationResult(
            es_valido=es_valido,
            confianza=max(0, min(100, confianza)),
            errores=errores or [],
            advertencias=advertencias or [],
            sugerencias=sugerencias or [],
            metadata=metadata or {},
        )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(nombre={self.nombre!r})>"
