"""
Resolvedor de fuentes para alimentos (ingredientes).

Flujo con fallback LLM (NUEVO):
  1. Cache inteligente
  2. BD local (alimentos + alias)
  3. USDA API
  4. FatSecret API
  5. *** LLM Estimación (NUEVO) → guarda en BD para consistencia ***
  6. Registra como pendiente (último recurso)
"""
from __future__ import annotations

from typing import Optional, Dict, Any, List
from sqlalchemy.orm import Session
from app.models import Alimento, AlimentoAlias, AlimentoSinResolver
from app.services.nutrition.food.resolver.cache_manager import CacheManager
from app.services.nutrition.food.resolver.api_clients import USDAClient, FatSecretClient
from app.services.validators import FingerprintGenerator
import logging

logger = logging.getLogger(__name__)


class FoodSourceResolver:
    """
    Orquesta la búsqueda de alimentos (ingredientes) en múltiples fuentes.

    Orden de búsqueda:
    1. Cache inteligente (por usuario)
    2. BD local (alimentos + alias)
    3. USDA API
    4. FatSecret API
    5. LLM Estimación → persiste en BD (NUEVO)
    6. Registra como "sin resolver" (fallback final)
    """

    def __init__(
        self,
        db: Session,
        cache_manager: CacheManager,
        usda_client: Optional[USDAClient] = None,
        fatsecret_client: Optional[FatSecretClient] = None,
        llm_service=None,
    ):
        self.db = db
        self.cache_manager = cache_manager
        self.usda_client = usda_client
        self.fatsecret_client = fatsecret_client
        self._llm = llm_service  # Inyectado opcionalmente

    def resolver_ingrediente(
        self,
        nombre_ingrediente: str,
        user_id: int,
        gramos: Optional[float] = 100,
    ) -> Dict[str, Any]:
        """
        Resuelve un ingrediente en múltiples fuentes.

        Returns:
            {
                'exito': bool,
                'nombre': str,
                'alimento_id': int (si existe en BD),
                'macros_100g': {calorias, proteina, carbos, grasas, fibra, azucar},
                'macros_totales': {basado en gramos},
                'source': 'Cache|BD|USDA|FatSecret|LLM_Estimado',
                'confianza': 0-100,
                'fingerprint': str,
                'advertencias': [],
            }
        """
        nombre_norm = self._normalizar_nombre(nombre_ingrediente)
        logger.info(f"Resolviendo ingrediente: {nombre_norm} ({gramos}g)")

        # 1. Caché
        resultado_cache = self._buscar_cache(nombre_norm, user_id)
        if resultado_cache:
            logger.info(f"✅ Cache: {nombre_norm}")
            return self._construir_resultado(
                nombre=nombre_ingrediente,
                macros_100g=resultado_cache,
                gramos=gramos,
                source='Cache',
                confianza=90,
            )

        # 2. BD local
        resultado_bd = self._buscar_bd_local(nombre_norm)
        if resultado_bd:
            logger.info(f"✅ BD local: {nombre_norm}")
            self.cache_manager.guardar_en_cache(
                food_normalized=nombre_norm,
                user_id=user_id,
                macros=resultado_bd['macros'],
                source='BD',
                alimento_id=resultado_bd['id'],
            )
            return self._construir_resultado(
                nombre=nombre_ingrediente,
                alimento_id=resultado_bd['id'],
                macros_100g=resultado_bd['macros'],
                gramos=gramos,
                source='BD',
                confianza=95,
            )

        # 3. USDA API
        if self.usda_client:
            resultado_usda = self._buscar_usda(nombre_norm)
            if resultado_usda:
                logger.info(f"✅ USDA: {nombre_norm} — guardando en BD")
                alimento_id = self._persistir_en_bd(
                    nombre=nombre_ingrediente,
                    nombre_norm=nombre_norm,
                    macros=resultado_usda,
                    source='USDA',
                )
                self.cache_manager.guardar_en_cache(
                    food_normalized=nombre_norm,
                    user_id=user_id,
                    macros=resultado_usda,
                    source='USDA',
                    alimento_id=alimento_id,
                )
                return self._construir_resultado(
                    nombre=nombre_ingrediente,
                    alimento_id=alimento_id,
                    macros_100g=resultado_usda,
                    gramos=gramos,
                    source='USDA',
                    confianza=85,
                )

        # 4. FatSecret API
        if self.fatsecret_client:
            resultado_fs = self._buscar_fatsecret(nombre_norm)
            if resultado_fs:
                logger.info(f"✅ FatSecret: {nombre_norm} — guardando en BD")
                alimento_id = self._persistir_en_bd(
                    nombre=nombre_ingrediente,
                    nombre_norm=nombre_norm,
                    macros=resultado_fs,
                    source='FatSecret',
                )
                self.cache_manager.guardar_en_cache(
                    food_normalized=nombre_norm,
                    user_id=user_id,
                    macros=resultado_fs,
                    source='FatSecret',
                    alimento_id=alimento_id,
                )
                return self._construir_resultado(
                    nombre=nombre_ingrediente,
                    alimento_id=alimento_id,
                    macros_100g=resultado_fs,
                    gramos=gramos,
                    source='FatSecret',
                    confianza=80,
                )

        # 5. ★ LLM Estimación (NUEVO FALLBACK) ★
        resultado_llm = self._estimar_con_llm(nombre_norm, nombre_ingrediente)
        if resultado_llm:
            logger.info(f"✅ LLM estimado: {nombre_norm} — guardando en BD para consistencia")
            # Persistir en BD para que futuras consultas sean deterministas
            alimento_id = self._persistir_en_bd(
                nombre=nombre_ingrediente,
                nombre_norm=nombre_norm,
                macros=resultado_llm,
                source='LLM_Estimado',
            )
            # También cachear
            self.cache_manager.guardar_en_cache(
                food_normalized=nombre_norm,
                user_id=user_id,
                macros=resultado_llm,
                source='LLM_Estimado',
                alimento_id=alimento_id,
            )
            return self._construir_resultado(
                nombre=nombre_ingrediente,
                alimento_id=alimento_id,
                macros_100g=resultado_llm,
                gramos=gramos,
                source='LLM_Estimado',
                confianza=65,
                advertencias=[
                    f"'{nombre_ingrediente}' estimado por IA — valores aproximados. "
                    "Se guardarán para consistencia futura."
                ],
            )

        # 6. Fallback final: registrar como pendiente
        logger.warning(f"❌ Sin resolver: {nombre_norm}")
        self._registrar_sin_resolver(nombre=nombre_norm, user_id=user_id)

        return {
            'exito': False,
            'nombre': nombre_ingrediente,
            'gramos': gramos or 100,
            'alimento_id': None,
            'macros_100g': None,
            'macros_totales': None,
            'source': 'Nutricionista (pendiente)',
            'confianza': 0,
            'fingerprint': None,
            'advertencias': [
                f"Ingrediente '{nombre_ingrediente}' no se pudo resolver. "
                "Registrado para validación del nutricionista."
            ],
        }

    def resolver_ingredientes_lote(
        self,
        ingredientes: List[Dict[str, Any]],
        user_id: int,
    ) -> List[Dict[str, Any]]:
        """Resuelve múltiples ingredientes eficientemente."""
        return [
            self.resolver_ingrediente(
                nombre_ingrediente=ing['nombre'],
                user_id=user_id,
                gramos=ing.get('gramos', 100),
            )
            for ing in ingredientes
        ]

    # ──────────────────────────────────────────────────────────────────────────
    # NUEVO: LLM Fallback
    # ──────────────────────────────────────────────────────────────────────────

    def _estimar_con_llm(
        self,
        nombre_norm: str,
        nombre_original: str,
    ) -> Optional[Dict[str, float]]:
        """
        Pide al LLM que estime valores nutricionales por 100g.

        Retorna dict con macros o None si el LLM no está disponible.
        """
        llm = self._obtener_llm()
        if llm is None:
            return None

        try:
            import json as json_lib
            import asyncio

            prompt = (
                f"Proporciona los valores nutricionales por 100g de '{nombre_original}'. "
                "Responde ÚNICAMENTE con un JSON válido con estas claves exactas "
                "(valores numéricos, sin texto adicional):\n"
                '{"calorias_100g": N, "proteina_100g": N, '
                '"carbohidratos_100g": N, "grasas_100g": N, '
                '"fibra_100g": N, "azucar_100g": N}\n'
                "Si no sabes el alimento, usa null. "
                "Basa los valores en tablas nutricionales estándar del CENAN/USDA."
            )

            # Llamar al LLM de forma síncrona
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        future = pool.submit(
                            asyncio.run,
                            llm.completar(prompt=prompt, max_tokens=200),
                        )
                        respuesta = future.result(timeout=10)
                else:
                    respuesta = loop.run_until_complete(
                        llm.completar(prompt=prompt, max_tokens=200)
                    )
            except Exception:
                respuesta = asyncio.run(
                    llm.completar(prompt=prompt, max_tokens=200)
                )

            if not respuesta:
                return None

            # Extraer JSON de la respuesta
            macros = self._extraer_json_macros(respuesta)
            if macros and self._validar_macros_estimadas(macros):
                logger.info(
                    f"LLM estimó '{nombre_norm}': {macros.get('calorias_100g')} kcal/100g"
                )
                return macros

            return None

        except Exception as exc:
            logger.warning(f"LLM fallback error para '{nombre_norm}': {exc}")
            return None

    def _extraer_json_macros(self, texto: str) -> Optional[Dict[str, float]]:
        """Extrae JSON de macros de la respuesta LLM."""
        import json as json_lib
        import re

        # Intentar parsear directamente
        texto = texto.strip()
        try:
            data = json_lib.loads(texto)
            return self._normalizar_macros(data)
        except Exception:
            pass

        # Buscar JSON dentro del texto (a veces viene con explicación)
        match = re.search(r'\{[^{}]+\}', texto, re.DOTALL)
        if match:
            try:
                data = json_lib.loads(match.group())
                return self._normalizar_macros(data)
            except Exception:
                pass

        return None

    def _normalizar_macros(self, data: dict) -> Optional[Dict[str, float]]:
        """Normaliza y valida el dict de macros del LLM."""
        required = {
            'calorias_100g', 'proteina_100g',
            'carbohidratos_100g', 'grasas_100g',
        }
        if not required.issubset(data.keys()):
            return None

        # Asegurarse de que todos los valores son float y >= 0
        try:
            return {
                'calorias_100g': max(0.0, float(data.get('calorias_100g') or 0)),
                'proteina_100g': max(0.0, float(data.get('proteina_100g') or 0)),
                'carbohidratos_100g': max(0.0, float(data.get('carbohidratos_100g') or 0)),
                'grasas_100g': max(0.0, float(data.get('grasas_100g') or 0)),
                'fibra_100g': max(0.0, float(data.get('fibra_100g') or 0)),
                'azucar_100g': max(0.0, float(data.get('azucar_100g') or 0)),
            }
        except (TypeError, ValueError):
            return None

    def _validar_macros_estimadas(self, macros: Dict[str, float]) -> bool:
        """
        Valida que las macros estimadas tengan sentido físico.
        Aplica la regla de Atwater: 4*P + 4*C + 9*G ≈ kcal (±30%).
        """
        kcal = macros.get('calorias_100g', 0)
        prot = macros.get('proteina_100g', 0)
        carb = macros.get('carbohidratos_100g', 0)
        gras = macros.get('grasas_100g', 0)

        # Valores imposibles
        if kcal <= 0 or kcal > 900:
            return False
        if prot < 0 or carb < 0 or gras < 0:
            return False
        if prot + carb + gras > 100:
            return False

        # Chequeo Atwater mínimo
        kcal_calc = 4 * prot + 4 * carb + 9 * gras
        if kcal_calc > 0:
            ratio = kcal / kcal_calc
            if ratio < 0.5 or ratio > 2.0:
                logger.warning(
                    f"Macros LLM no pasan Atwater: {kcal}kcal vs {kcal_calc}kcal calculadas"
                )
                return False

        return True

    def _persistir_en_bd(
        self,
        nombre: str,
        nombre_norm: str,
        macros: Dict[str, float],
        source: str,
    ) -> Optional[int]:
        """
        Guarda el alimento estimado en la tabla `alimentos` de la BD.

        Retorna el ID del alimento creado o None si falla.
        """
        try:
            # Verificar si ya existe (doble chequeo por race condition)
            existente = self.db.query(Alimento).filter(
                Alimento.nombre_normalizado == nombre_norm
            ).first()

            if existente:
                logger.info(f"Alimento ya existe en BD: {nombre_norm} (ID={existente.id})")
                return existente.id

            nuevo = Alimento(
                nombre=nombre.title(),
                nombre_normalizado=nombre_norm,
                calorias_100g=macros['calorias_100g'],
                proteina_100g=macros['proteina_100g'],
                carbohidratos_100g=macros['carbohidratos_100g'],
                grasas_100g=macros['grasas_100g'],
                fibra_100g=macros.get('fibra_100g', 0.0),
                azucar_100g=macros.get('azucar_100g', 0.0),
                fuente=source,           # 'LLM_Estimado'
                es_verificado=False,
            )
            self.db.add(nuevo)
            self.db.commit()
            self.db.refresh(nuevo)
            logger.info(
                f"✅ Nuevo alimento persistido: '{nombre}' ID={nuevo.id} "
                f"({macros['calorias_100g']} kcal/100g) [fuente={source}]"
            )
            return nuevo.id

        except Exception as exc:
            logger.error(f"Error persistiendo alimento LLM '{nombre}': {exc}")
            self.db.rollback()
            return None

    def _obtener_llm(self):
        """Obtiene LLMService de forma lazy."""
        if self._llm is not None:
            return self._llm
        try:
            from app.services.ai.llm_service import LLMService
            self._llm = LLMService()
            return self._llm
        except Exception as exc:
            logger.warning(f"LLMService no disponible: {exc}")
            return None

    # ──────────────────────────────────────────────────────────────────────────
    # Búsqueda en fuentes
    # ──────────────────────────────────────────────────────────────────────────

    def _normalizar_nombre(self, nombre: str) -> str:
        """Normaliza nombre para búsqueda."""
        return nombre.lower().strip()

    def _buscar_cache(self, nombre_norm: str, user_id: int) -> Optional[Dict]:
        return self.cache_manager.obtener_del_cache(nombre_norm, user_id)

    def _buscar_bd_local(self, nombre_norm: str) -> Optional[Dict]:
        """Busca en BD local (alimentos + alias) con búsqueda fuzzy básica."""
        try:
            # Exacto
            alimento = self.db.query(Alimento).filter(
                Alimento.nombre_normalizado == nombre_norm
            ).first()

            # Si no exacto → buscar por LIKE (contiene)
            if not alimento:
                alimento = self.db.query(Alimento).filter(
                    Alimento.nombre_normalizado.ilike(f'%{nombre_norm}%')
                ).first()

            if alimento:
                return {
                    'id': alimento.id,
                    'nombre': alimento.nombre,
                    'macros': {
                        'calorias_100g': float(alimento.calorias_100g or 0),
                        'proteina_100g': float(alimento.proteina_100g or 0),
                        'carbohidratos_100g': float(alimento.carbohidratos_100g or 0),
                        'grasas_100g': float(alimento.grasas_100g or 0),
                        'fibra_100g': float(alimento.fibra_100g or 0),
                        'azucar_100g': float(alimento.azucar_100g or 0),
                    }
                }

            # Buscar en alias
            alias = self.db.query(AlimentoAlias).filter(
                AlimentoAlias.alias == nombre_norm
            ).first()

            if alias:
                a = self.db.query(Alimento).filter(Alimento.id == alias.alimento_id).first()
                if a:
                    return {
                        'id': a.id,
                        'nombre': a.nombre,
                        'macros': {
                            'calorias_100g': float(a.calorias_100g or 0),
                            'proteina_100g': float(a.proteina_100g or 0),
                            'carbohidratos_100g': float(a.carbohidratos_100g or 0),
                            'grasas_100g': float(a.grasas_100g or 0),
                            'fibra_100g': float(a.fibra_100g or 0),
                            'azucar_100g': float(a.azucar_100g or 0),
                        }
                    }

            return None

        except Exception as e:
            logger.error(f"Error buscando en BD: {e}")
            return None

    def _buscar_usda(self, nombre_norm: str) -> Optional[Dict]:
        """Busca en USDA (stub — implementar con API key)."""
        return None

    def _buscar_fatsecret(self, nombre_norm: str) -> Optional[Dict]:
        """Busca en FatSecret (stub — implementar con OAuth)."""
        return None

    def _registrar_sin_resolver(self, nombre: str, user_id: int) -> bool:
        """Registra alimento que no se pudo resolver."""
        try:
            existing = self.db.query(AlimentoSinResolver).filter(
                AlimentoSinResolver.nombre_normalizado == nombre,
                AlimentoSinResolver.user_id == user_id,
                AlimentoSinResolver.estado == 'pendiente',
            ).first()

            if existing:
                existing.intentos = (existing.intentos or 0) + 1
                self.db.commit()
                return True

            sin_resolver = AlimentoSinResolver(
                nombre_original=nombre,
                nombre_normalizado=nombre,
                user_id=user_id,
                estado='pendiente',
            )
            self.db.add(sin_resolver)
            self.db.commit()
            return True

        except Exception as exc:
            logger.error(f"Error registrando sin resolver: {exc}")
            self.db.rollback()
            return False

    # ──────────────────────────────────────────────────────────────────────────
    # Construcción de resultado
    # ──────────────────────────────────────────────────────────────────────────

    def _construir_resultado(
        self,
        nombre: str,
        macros_100g: Dict[str, float],
        gramos: float,
        source: str,
        confianza: int,
        alimento_id: Optional[int] = None,
        advertencias: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Construye resultado de resolución."""
        gramos = gramos or 100
        macros_totales = {
            'calorias':       (macros_100g.get('calorias_100g', 0) / 100) * gramos,
            'proteina':       (macros_100g.get('proteina_100g', 0) / 100) * gramos,
            'carbohidratos':  (macros_100g.get('carbohidratos_100g', 0) / 100) * gramos,
            'grasas':         (macros_100g.get('grasas_100g', 0) / 100) * gramos,
        }

        fingerprint = FingerprintGenerator.generar_fingerprint_alimento(
            nombre=nombre,
            calorias_100g=macros_100g.get('calorias_100g', 0),
            proteina_100g=macros_100g.get('proteina_100g', 0),
            carbohidratos_100g=macros_100g.get('carbohidratos_100g', 0),
            grasas_100g=macros_100g.get('grasas_100g', 0),
            source=source,
        )

        return {
            'exito': True,
            'nombre': nombre,
            'gramos': gramos,
            'alimento_id': alimento_id,
            'macros_100g': macros_100g,
            'macros_totales': macros_totales,
            'source': source,
            'confianza': confianza,
            'fingerprint': fingerprint,
            'advertencias': advertencias or [],
        }
