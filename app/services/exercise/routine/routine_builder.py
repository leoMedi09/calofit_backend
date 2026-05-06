"""
Constructor de rutinas (RoutineBuilder).
"""
from typing import Optional, List, Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session
from app.models import (
    Rutina,
    RutinaEjercicio,
    Ejercicio,
    AppCacheRutinas,
    Client,
)
from app.services.validators import FingerprintGenerator
from app.services.exercise.routine.models import (
    RoutineConstructionResultDTO,
    EjercicioEnRutinaDTO,
)
import logging

logger = logging.getLogger(__name__)


class RoutineBuilder:
    """
    Constructor de rutinas VALIDADAS.
    
    Proceso:
    1. Resuelve ejercicios
    2. Valida existencia de ejercicios
    3. Calcula intensidad total
    4. Estima kcal quemadas
    5. Genera fingerprint
    6. Cachea resultado
    7. Retorna RoutineConstructionResult
    """
    
    # Constantes MET (Metabolic Equivalent of Task)
    MET_MIN = 1.0
    MET_MAX = 20.0
    
    def __init__(self, db: Session):
        self.db = db
    
    def construir_rutina(
        self,
        nombre_rutina: str,
        tipo_entrenamiento: str,
        intensidad: str,
        objetivo: str,
        ejercicios: List[Dict[str, Any]],
        client_id: int,
    ) -> RoutineConstructionResultDTO:
        """
        Construye y valida una rutina.
        
        Args:
            nombre_rutina: nombre de la rutina
            tipo_entrenamiento: fuerza|cardio|HIIT|flexibilidad
            intensidad: baja|media|alta
            objetivo: hipertrofia|resistencia|definicion
            ejercicios: [{'nombre': str, 'series': int, 'reps': int, ...}, ...]
            client_id: ID del cliente
            
        Returns:
            RoutineConstructionResultDTO
        """
        
        logger.info(f"Construyendo rutina: {nombre_rutina}")
        
        # 1. Verificar si está en caché
        nombre_norm = nombre_rutina.lower().strip()
        resultado_cache = self._buscar_en_cache(nombre_norm, client_id)
        if resultado_cache:
            logger.info(f"✅ Rutina en caché: {nombre_rutina}")
            resultado_cache.cached = True
            return resultado_cache
        
        # 2. Resolver ejercicios
        ejercicios_resueltos = self._resolver_ejercicios(ejercicios)
        
        ejercicios_fallidos = [
            ej for ej in ejercicios_resueltos if ej is None
        ]
        
        # 3. Validar ejercicios
        es_valido = len(ejercicios_fallidos) == 0
        
        # 4. Calcular métricas
        duracion_estimada = self._calcular_duracion(ejercicios_resueltos)
        
        # Obtener peso del cliente para calcular kcal
        cliente = self.db.query(Client).filter(Client.id == client_id).first()
        peso_kg = cliente.weight if cliente else 70  # Default 70kg
        
        kcal_quemadas = self._calcular_kcal_quemadas(
            ejercicios_resueltos,
            duracion_estimada,
            peso_kg,
        )
        
        intensidad_total = self._calcular_intensidad_total(
            intensidad,
            ejercicios_resueltos,
        )
        
        # 5. Generar fingerprint
        fingerprint = FingerprintGenerator.generar_fingerprint_rutina(
            nombre=nombre_rutina,
            ejercicios=[
                {
                    'ejercicio_id': ej.get('ejercicio_id'),
                    'series': ej.get('series'),
                    'reps': ej.get('repeticiones'),
                }
                for ej in ejercicios_resueltos if ej
            ],
            intensidad=intensidad,
        )
        
        # 6. Guardar rutina en BD
        rutina_id = self._guardar_rutina(
            nombre_rutina,
            tipo_entrenamiento,
            intensidad,
            objetivo,
            duracion_estimada,
            ejercicios_resueltos,
        )
        
        # 7. Cachear resultado
        confianza = 100 if es_valido else 70
        self._cachear_resultado(
            rutina_id=rutina_id,
            client_id=client_id,
            fingerprint=fingerprint,
            intensidad=intensidad,
            kcal_estimadas=kcal_quemadas,
            confianza=confianza,
        )
        
        # 8. Construir DTO
        resultado = RoutineConstructionResultDTO(
            exito=es_valido,
            rutina_id=rutina_id,
            nombre=nombre_rutina,
            tipo_entrenamiento=tipo_entrenamiento,
            intensidad=intensidad,
            objetivo=objetivo,
            duracion_estimada_minutos=duracion_estimada,
            ejercicios=[
                EjercicioEnRutinaDTO(
                    ejercicio_id=ej['ejercicio_id'],
                    nombre=ej['nombre'],
                    series=ej['series'],
                    repeticiones=ej['repeticiones'],
                    peso_kg_recomendado=ej.get('peso_kg'),
                    descanso_segundos=ej.get('descanso', 60),
                    orden=idx + 1,
                    met=ej.get('met'),
                    kcal_por_minuto=ej.get('kcal_por_minuto'),
                )
                for idx, ej in enumerate(ejercicios_resueltos) if ej
            ],
            total_ejercicios=len([ej for ej in ejercicios_resueltos if ej]),
            kcal_quemadas_estimadas=kcal_quemadas,
            intensidad_total=intensidad_total,
            fingerprint=fingerprint,
            confianza_global=confianza,
            timestamp=datetime.utcnow().isoformat(),
        )
        
        logger.info(f"✅ Rutina construida: {nombre_rutina} (válida={es_valido})")
        return resultado
    
    def _resolver_ejercicios(
        self,
        ejercicios: List[Dict[str, Any]],
    ) -> List[Optional[Dict[str, Any]]]:
        """Resuelve ejercicios por nombre."""
        resueltos = []
        
        for ej in ejercicios:
            nombre_ej = ej.get('nombre', '').lower().strip()
            
            # Buscar en BD
            ejercicio = self.db.query(Ejercicio).filter(
                Ejercicio.nombre_normalizado == nombre_ej
            ).first()
            
            if ejercicio:
                resueltos.append({
                    'ejercicio_id': ejercicio.id,
                    'nombre': ejercicio.nombre,
                    'series': ej.get('series', 3),
                    'repeticiones': ej.get('repeticiones', 10),
                    'peso_kg': ej.get('peso_kg'),
                    'descanso': ej.get('descanso_segundos', 60),
                    'met': ejercicio.met,
                    'kcal_por_minuto': ejercicio.met / 60 if ejercicio.met else 0,
                })
            else:
                logger.warning(f"Ejercicio no encontrado: {nombre_ej}")
                resueltos.append(None)
        
        return resueltos
    
    def _calcular_duracion(self, ejercicios: List[Optional[Dict]]) -> int:
        """Calcula duración estimada de rutina."""
        # Aproximado: 3 min por serie promedio + calentamiento
        series_totales = sum(
            ej['series'] for ej in ejercicios if ej
        )
        return 5 + (series_totales * 3)  # 5 min calentamiento + 3 min/serie
    
    def _calcular_kcal_quemadas(
        self,
        ejercicios: List[Optional[Dict]],
        duracion_minutos: int,
        peso_kg: float,
    ) -> float:
        """
        Calcula kcal quemadas totales.
        
        Fórmula: MET × peso_kg × 3.5 / 200 × minutos
        """
        total_kcal = 0
        
        for ej in ejercicios:
            if not ej:
                continue
            
            met = ej.get('met', 5.0)  # Default 5 MET
            
            # Duración por ejercicio (aproximado)
            duracion_ej = (ej['series'] * ej['repeticiones']) / 10  # ~6 seconds per rep
            
            kcal = (met * peso_kg * 3.5 / 200) * duracion_ej
            total_kcal += kcal
        
        return round(total_kcal, 2)
    
    def _calcular_intensidad_total(
        self,
        intensidad: str,
        ejercicios: List[Optional[Dict]],
    ) -> float:
        """Calcula intensidad total (0-100)."""
        # Mapeo de intensidad
        intensidad_map = {
            'baja': 30,
            'media': 60,
            'alta': 85,
            'muy alta': 100
        }
        
        base = intensidad_map.get(intensidad.lower(), 50)
        
        # Ajuste por volumen
        volumen = len([e for e in ejercicios if e])
        ajuste_volumen = min(15, volumen * 1.5)
        
        return min(100.0, float(base + ajuste_volumen))

    def _buscar_en_cache(self, nombre_norm: str, client_id: int) -> Optional[RoutineConstructionResultDTO]:
        """Busca rutina en caché."""
        # TODO: Implementar búsqueda real
        return None

    def _guardar_rutina(
        self,
        nombre_rutina: str,
        tipo_entrenamiento: str,
        intensidad: str,
        objetivo: str,
        duracion_estimada: int,
        ejercicios_resueltos: List[Optional[Dict]],
    ) -> int:
        """Guarda la rutina en BD."""
        try:
            nombre_norm = nombre_rutina.lower().strip()
            # TODO: Lógica real de guardado
            return 1 # ID simulado
        except Exception as e:
            logger.error(f"Error guardando rutina: {str(e)}")
            return 0

    def _cachear_resultado(
        self,
        rutina_id: int,
        client_id: int,
        fingerprint: str,
        intensidad: str,
        kcal_estimadas: float,
        confianza: int,
    ) -> bool:
        """Cachea el resultado de la construcción."""
        try:
            # TODO: Implementar guardado en AppCacheRutinas
            return True
        except Exception as e:
            logger.error(f"Error cacheando rutina: {str(e)}")
            return False
