from dataclasses import dataclass, field
from typing import List, Dict, Any
from app.core.objetivo_utils import normalizar_objetivo, MANTENIMIENTO

@dataclass(frozen=True)
class UserContext:
    perfil_id: int
    nombre: str
    objetivo_normalizado: str  # DEFICIT, MANTENIMIENTO, SUPERAVIT
    condiciones_medicas: List[str] = field(default_factory=list)
    restricciones_alimentarias: List[str] = field(default_factory=list)
    
    # Balance del día
    consumido: float = 0.0
    meta: float = 2000.0
    quemado: float = 0.0
    
    # Plan nutricional actual
    plan_actual: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        perfil,
        consumido: float,
        quemado: float,
        plan_actual: Dict[str, Any]
    ) -> "UserContext":
        return cls(
            perfil_id=perfil.id,
            nombre=perfil.first_name or "",
            objetivo_normalizado=normalizar_objetivo(perfil.goal),
            condiciones_medicas=list(perfil.medical_conditions or []),
            restricciones_alimentarias=list(perfil.forbidden_foods or []),
            consumido=consumido,
            meta=plan_actual.get("calorias_dia", 2000.0),
            quemado=quemado,
            plan_actual=plan_actual
        )
