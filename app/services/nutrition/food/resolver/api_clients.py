"""
Clientes para APIs externas (USDA, FatSecret).
"""
from typing import Optional, Dict, Any
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class FoodAPIClient(ABC):
    """Clase base para clientes de APIs de alimentos."""
    
    @abstractmethod
    def buscar_alimento(self, nombre: str) -> Optional[Dict[str, Any]]:
        """Busca alimento por nombre."""
        pass
    
    @abstractmethod
    def obtener_macros(self, food_id: str) -> Optional[Dict[str, float]]:
        """Obtiene macros del alimento."""
        pass


class USDAClient(FoodAPIClient):
    """
    Cliente para USDA FoodData Central API.
    
    Endpoint: https://fdc.nal.usda.gov/api/
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://fdc.nal.usda.gov/api/food/search"
        self.nombre = "USDA"
    
    def buscar_alimento(self, nombre: str) -> Optional[Dict[str, Any]]:
        """
        Busca alimento en USDA.
        
        Returns:
            {
                'food_id': str,
                'nombre': str,
                'macros': {calorias_100g, proteina_100g, ...}
            }
        """
        # TODO: Implementar con requests
        # Este es un stub para la estructura
        logger.info(f"Buscando en USDA: {nombre}")
        return None
    
    def obtener_macros(self, food_id: str) -> Optional[Dict[str, float]]:
        """Obtiene macros de USDA."""
        # TODO: Implementar
        return None


class FatSecretClient(FoodAPIClient):
    """
    Cliente para FatSecret API.
    
    Endpoint: https://platform.fatsecret.com/rest/
    """
    
    def __init__(self, consumer_key: Optional[str] = None, consumer_secret: Optional[str] = None):
        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self.base_url = "https://platform.fatsecret.com/rest"
        self.nombre = "FatSecret"
    
    def buscar_alimento(self, nombre: str) -> Optional[Dict[str, Any]]:
        """
        Busca alimento en FatSecret.
        
        Returns:
            {
                'food_id': str,
                'nombre': str,
                'macros': {calorias_100g, proteina_100g, ...}
            }
        """
        # TODO: Implementar con OAuth
        logger.info(f"Buscando en FatSecret: {nombre}")
        return None
    
    def obtener_macros(self, food_id: str) -> Optional[Dict[str, float]]:
        """Obtiene macros de FatSecret."""
        # TODO: Implementar
        return None
