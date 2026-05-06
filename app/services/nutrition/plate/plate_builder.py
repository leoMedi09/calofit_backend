"""
Constructor de platos (PlatoBuilder).
"""
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone, timedelta
from sqlalchemy.orm import Session
from app.models import Plato, PlatoIngrediente, Alimento, AppCachePlatos, MetaUsuario
from app.services.nutrition.food.resolver.source_resolver import FoodSourceResolver
from app.services.nutrition.food.resolver.cache_manager import CacheManager
from app.services.validators import (
    SemanticValidator,
    NutritionalValidator,
    FingerprintGenerator,
)
from app.services.nutrition.plate.models import (
    PlatoConstructionResultDTO,
    IngredienteDTO,
    MacrosDTO,
)
import logging

logger = logging.getLogger(__name__)


class PlatoBuilder:
    """
    Constructor de platos VALIDADOS.
    
    Proceso:
    1. Resuelve ingredientes (FoodSourceResolver)
    2. Valida semántica (¿tiene sentido?)
    3. Calcula macros en tiempo real
    4. Valida nutricional (¿macros válidas?)
    5. Genera fingerprint
    6. Cachea resultado
    7. Retorna PlatoConstructionResult
    """
    
    def __init__(
        self,
        db: Session,
        food_resolver: FoodSourceResolver,
        cache_manager: CacheManager,
    ):
        self.db = db
        self.food_resolver = food_resolver
        self.cache_manager = cache_manager
        self.semantic_validator = SemanticValidator(db)
        self.nutritional_validator = NutritionalValidator()
    
    def construir_plato(
        self,
        nombre_plato: str,
        ingredientes: List[Dict[str, Any]],
        client_id: int,
        tipo_plato: str = "cualquiera",
    ) -> PlatoConstructionResultDTO:
        """
        Construye y valida un plato.
        
        Args:
            nombre_plato: nombre del plato
            ingredientes: [{'nombre': str, 'gramos': float}, ...]
            client_id: ID del cliente (para caché + restricciones)
            tipo_plato: desayuno|almuerzo|cena|snack|cualquiera
            
        Returns:
            PlatoConstructionResultDTO
        """
        
        logger.info(f"Construyendo plato: {nombre_plato} para cliente {client_id}")
        
        # 1. Verificar si está en caché
        nombre_norm = nombre_plato.lower().strip()
        resultado_cache = self._buscar_en_cache(nombre_norm, client_id)
        if resultado_cache:
            logger.info(f"✅ Plato en caché: {nombre_plato}")
            resultado_cache.cached = True
            return resultado_cache
        
        # 2. Resolver ingredientes
        ingredientes_resueltos = self.food_resolver.resolver_ingredientes_lote(
            ingredientes=ingredientes,
            user_id=client_id,
        )
        
        # Verificar si todos se resolvieron
        ingredientes_fallidos = [
            ing for ing in ingredientes_resueltos if not ing['exito']
        ]
        if ingredientes_fallidos:
            logger.warning(f"⚠️ {len(ingredientes_fallidos)} ingredientes no resueltos")
        
        # 3. Calcular macros totales
        macros_totales = self._calcular_macros_totales(ingredientes_resueltos)
        peso_total = sum(ing.get('gramos', 0) for ing in ingredientes)
        
        # 4. Validar semántica
        validacion_semantica = self.semantic_validator.validar({
            'nombre_plato': nombre_plato,
            'ingredientes': [
                {'nombre': ing['nombre']} for ing in ingredientes_resueltos
            ],
            'client_id': client_id,
        })
        
        # 5. Validar nutricional
        meta = self.db.query(MetaUsuario).filter(MetaUsuario.client_id == client_id).first()
        tdee = meta.calorias_objetivo if meta else None
        
        validacion_nutricional = self.nutritional_validator.validar({
            'nombre_plato': nombre_plato,
            'peso_total_gramos': peso_total,
            'calorias_total': macros_totales['calorias'],
            'proteina_total': macros_totales['proteina'],
            'carbohidratos_total': macros_totales['carbohidratos'],
            'grasas_total': macros_totales['grasas'],
            'tdee_usuario': tdee,
            'momento_dia': tipo_plato,
        })
        
        # 6. Generar fingerprint
        fingerprint = FingerprintGenerator.generar_fingerprint_plato(
            nombre=nombre_plato,
            ingredientes=[
                {'nombre': ing['nombre'], 'gramos': ing.get('gramos', 0)}
                for ing in ingredientes_resueltos
            ],
            macros=macros_totales,
        )
        
        # 7. Calcular confianza global
        confianza_global = self._calcular_confianza_global(
            validacion_semantica.confianza,
            validacion_nutricional.confianza,
            len(ingredientes_fallidos),
            len(ingredientes),
        )
        
        # 8. Crear o actualizar plato en BD
        plato_id = self._guardar_plato(nombre_plato, ingredientes_resueltos)
        
        # 9. Cachear resultado
        self._cachear_resultado(
            nombre_norm=nombre_norm,
            plato_id=plato_id,
            client_id=client_id,
        )
        
        # 10. Construir DTO de respuesta
        es_valido = (
            validacion_semantica.es_valido and
            validacion_nutricional.es_valido and
            len(ingredientes_fallidos) == 0
        )
        
        resultado = PlatoConstructionResultDTO(
            exito=es_valido,
            plato_id=plato_id,
            nombre=nombre_plato,
            peso_total_gramos=peso_total,
            ingredientes=[
                IngredienteDTO(
                    nombre=ing['nombre'],
                    gramos=ing.get('gramos', 0),
                    alimento_id=ing.get('alimento_id'),
                    macros_100g=ing.get('macros_100g'),
                    macros_totales=ing.get('macros_totales'),
                    source=ing.get('source'),
                    confianza=ing.get('confianza', 0),
                    fingerprint=ing.get('fingerprint'),
                )
                for ing in ingredientes_resueltos
            ],
            macros_totales=MacrosDTO(**macros_totales),
            validacion_semantica=validacion_semantica,
            validacion_nutricional=validacion_nutricional,
            fingerprint=fingerprint,
            confianza_global=confianza_global,
            timestamp=datetime.utcnow().isoformat(),
        )
        
        logger.info(f"✅ Plato construido: {nombre_plato} (válido={es_valido})")
        return resultado
    
    def _buscar_en_cache(self, nombre_norm: str, client_id: int) -> Optional[PlatoConstructionResultDTO]:
        """Busca plato en caché (solo comprueba existencia — reconstrucción completa omitida)."""
        try:
            cache_plato = self.db.query(AppCachePlatos).filter(
                AppCachePlatos.plato_normalized == nombre_norm,
                AppCachePlatos.user_id == client_id,
            ).first()

            if not cache_plato:
                return None

            if cache_plato.expires_at and datetime.now(timezone.utc) > cache_plato.expires_at:
                self.db.delete(cache_plato)
                self.db.commit()
                return None

            # Reconstrucción completa requeriría re-ejecutar el pipeline;
            # retornar None para que el caller reconstruya y actualice caché.
            return None

        except Exception as exc:
            logger.error(f"Error buscando en caché: {exc}")
            return None
    
    def _calcular_macros_totales(self, ingredientes_resueltos: List[Dict]) -> Dict[str, float]:
        """Calcula macros totales del plato."""
        macros = {
            'calorias': 0,
            'proteina': 0,
            'carbohidratos': 0,
            'grasas': 0,
            'fibra': 0,
            'azucar': 0,
        }
        
        for ing in ingredientes_resueltos:
            if ing['exito'] and ing.get('macros_totales'):
                mt = ing['macros_totales']
                macros['calorias'] += mt.get('calorias', 0)
                macros['proteina'] += mt.get('proteina', 0)
                macros['carbohidratos'] += mt.get('carbohidratos', 0)
                macros['grasas'] += mt.get('grasas', 0)
        
        return macros
    
    def _calcular_confianza_global(
        self,
        conf_semantica: int,
        conf_nutricional: int,
        ingredientes_fallidos: int,
        total_ingredientes: int,
    ) -> int:
        """Calcula confianza global."""
        # Promedio ponderado
        confianza = (conf_semantica + conf_nutricional) / 2
        
        # Penalizar por ingredientes fallidos
        if total_ingredientes > 0:
            porcentaje_fallidos = (ingredientes_fallidos / total_ingredientes) * 100
            confianza -= porcentaje_fallidos * 0.5
        
        return max(0, min(100, int(confianza)))
    
    def _guardar_plato(
        self,
        nombre_plato: str,
        ingredientes_resueltos: List[Dict],
    ) -> int:
        """Guarda o actualiza plato en BD."""
        try:
            nombre_norm = nombre_plato.lower().strip()
            
            # Buscar plato existente
            plato = self.db.query(Plato).filter(
                Plato.nombre_normalizado == nombre_norm
            ).first()
            
            if not plato:
                # Crear nuevo
                plato = Plato(
                    nombre=nombre_plato,
                    nombre_normalizado=nombre_norm,
                    tipo_plato='cualquiera',
                    origen='llm',
                )
                self.db.add(plato)
                self.db.flush()
            
            # Limpiar ingredientes anteriores
            self.db.query(PlatoIngrediente).filter(
                PlatoIngrediente.plato_id == plato.id
            ).delete()
            
            # Agregar nuevos ingredientes
            for idx, ing in enumerate(ingredientes_resueltos):
                if ing['exito'] and ing.get('alimento_id'):
                    plato_ing = PlatoIngrediente(
                        plato_id=plato.id,
                        alimento_id=ing['alimento_id'],
                        gramos=ing.get('gramos', 100),
                        orden=idx + 1,
                    )
                    self.db.add(plato_ing)
            
            self.db.commit()
            return plato.id
        
        except Exception as e:
            logger.error(f"Error guardando plato: {str(e)}")
            self.db.rollback()
            return None
    
    def _cachear_resultado(
        self,
        nombre_norm: str,
        plato_id: int,
        client_id: int,
    ) -> bool:
        """Cachea referencia de plato construido."""
        try:
            expires = datetime.now(timezone.utc) + timedelta(days=30)
            cache_entry = AppCachePlatos(
                plato_normalized=nombre_norm,
                plato_id=plato_id,
                user_id=client_id,
                source='PlatoBuilder',
                expires_at=expires,
            )
            self.db.add(cache_entry)
            self.db.commit()
            logger.info(f"Plato cacheado: {plato_id}")
            return True

        except Exception as exc:
            logger.error(f"Error cacheando: {exc}")
            self.db.rollback()
            return False
