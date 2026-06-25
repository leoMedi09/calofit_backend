"""
🍽️ Módulo de Cálculo Automático de Dietas Basado en Métricas Biométricas

Este módulo calcula recomendaciones de dieta automáticamente basándose en:
- Peso y altura (IMC)
- Edad y género
- Nivel de actividad
- Objetivo de salud
"""

from datetime import date
from typing import Dict, Optional
from dataclasses import dataclass

from app.core.macros_diarios import macros_desde_calorias_pct_clasico
from app.core.objetivo_utils import normalizar_objetivo, DEFICIT, SUPERAVIT

@dataclass
class RecomendacionDieta:
    """Estructura de recomendaciones de dieta automática"""
    calorias_diarias: float
    proteinas_g: float
    carbohidratos_g: float
    grasas_g: float
    imc: float
    categoria_imc: str
    gasto_metabolico_basal: float
    objetivo_recomendado: str
    alimentos_recomendados: list
    alimentos_a_evitar: list
    frecuencia_comidas: str
    notas: str


class CalculadorDietaAutomatica:
    """
    Calcula recomendaciones de dieta de forma automática basándose en métricas biométricas
    """
    
    @staticmethod
    def calcular_imc(peso: float, altura: float) -> tuple[float, str]:
        """
        Calcula el IMC y devuelve la categoría
        
        Args:
            peso: en kilogramos
            altura: en centímetros
            
        Returns:
            (imc: float, categoria: str)
        """
        altura_m = altura / 100
        imc = peso / (altura_m ** 2)
        
        if imc < 18.5:
            categoria = "Bajo peso"
        elif imc < 25:
            categoria = "Peso normal"
        elif imc < 30:
            categoria = "Sobrepeso"
        elif imc < 35:
            categoria = "Obesidad grado I"
        elif imc < 40:
            categoria = "Obesidad grado II"
        else:
            categoria = "Obesidad grado III"
        
        return round(imc, 1), categoria
    
    @staticmethod
    def calcular_gasto_metabolico_basal(peso: float, altura: float, edad: int, genero: str) -> float:
        """
        Calcula GMB usando la fórmula de Harris-Benedict revisada
        
        Args:
            peso: en kg
            altura: en cm
            edad: en años
            genero: 'M' (masculino) o 'F' (femenino)
            
        Returns:
            GMB en calorías
        """
        if genero.upper() == 'M':
            gmb = 88.362 + (13.397 * peso) + (4.799 * altura) - (5.677 * edad)
        else:
            gmb = 447.593 + (9.247 * peso) + (3.098 * altura) - (4.330 * edad)
        
        return round(gmb, 0)
    
    @staticmethod
    def get_factor_actividad(nivel_actividad: str) -> float:
        """
        Obtiene el factor multiplicador según nivel de actividad
        
        Niveles:
        - Sedentario: sin ejercicio
        - Leve: 1-3 días/semana
        - Moderado: 3-5 días/semana
        - Intenso: 6-7 días/semana
        - Muy intenso: entrenamiento profesional
        """
        factores = {
            'Sedentario': 1.20,
            'Ligero': 1.375,
            'Moderado': 1.55,
            'Activo': 1.725,
            'Muy activo': 1.90
        }
        return factores.get(nivel_actividad, 1.20)  # Sedentario por defecto
    
    @staticmethod
    def calcular_recomendacion_dieta(
        peso: float,
        altura: float,
        edad: int,
        genero: str,
        nivel_actividad: str = 'Moderado',
        objetivo: str = 'Mantener peso'
    ) -> RecomendacionDieta:
        """
        Calcula recomendación completa de dieta basada en métricas
        
        Args:
            peso: en kg
            altura: en cm
            edad: en años
            genero: 'M' o 'F'
            nivel_actividad: Sedentario/Leve/Moderado/Intenso/Muy intenso
            objetivo: Perder peso/Mantener peso/Ganar masa
            
        Returns:
            RecomendacionDieta con todos los cálculos
        """
        
        # 1. Calcular IMC
        imc, categoria_imc = CalculadorDietaAutomatica.calcular_imc(peso, altura)
        
        # 2. Calcular GMB
        gmb = CalculadorDietaAutomatica.calcular_gasto_metabolico_basal(
            peso, altura, edad, genero
        )
        
        # 3. Aplicar factor de actividad
        factor_actividad = CalculadorDietaAutomatica.get_factor_actividad(nivel_actividad)
        gasto_calorico_diario = gmb * factor_actividad
        
        # 4. Ajustar según objetivo usando normalización canónica
        _concepto = normalizar_objetivo(objetivo)
        if _concepto == DEFICIT:
            calorias = gasto_calorico_diario * 0.85
            ajuste_objetivo = "Déficit calórico (perder ~0.5kg/semana)"
        elif _concepto == SUPERAVIT:
            calorias = gasto_calorico_diario * 1.1
            ajuste_objetivo = "Superávit calórico (ganar ~0.5kg/semana)"
        else:
            calorias = gasto_calorico_diario
            ajuste_objetivo = "Mantenimiento de peso actual"
        
        # 5. Macronutrientes (misma regla % que parsear_macros solo-kcal / utils histórico)
        m_pct = macros_desde_calorias_pct_clasico(calorias, objetivo)
        proteinas_g = m_pct["proteinas_g"]
        carbohidratos_g = m_pct["carbohidratos_g"]
        grasas_g = m_pct["grasas_g"]

        # 6. Determinar alimentos recomendados según categoría IMC
        alimentos_recomendados = CalculadorDietaAutomatica.get_alimentos_recomendados(
            categoria_imc, objetivo
        )
        
        alimentos_a_evitar = CalculadorDietaAutomatica.get_alimentos_a_evitar(
            categoria_imc, objetivo
        )
        
        # 7. Determinar frecuencia de comidas
        frecuencia_comidas = CalculadorDietaAutomatica.get_frecuencia_comidas(objetivo)
        
        # 8. Generar notas personalizadas
        notas = CalculadorDietaAutomatica.generar_notas(
            imc, categoria_imc, objetivo, edad
        )
        
        return RecomendacionDieta(
            calorias_diarias=round(calorias, 0),
            proteinas_g=round(proteinas_g, 1),
            carbohidratos_g=round(carbohidratos_g, 1),
            grasas_g=round(grasas_g, 1),
            imc=imc,
            categoria_imc=categoria_imc,
            gasto_metabolico_basal=gmb,
            objetivo_recomendado=ajuste_objetivo,
            alimentos_recomendados=alimentos_recomendados,
            alimentos_a_evitar=alimentos_a_evitar,
            frecuencia_comidas=frecuencia_comidas,
            notas=notas
        )
    
    @staticmethod
    def get_alimentos_recomendados(categoria_imc: str, objetivo: str) -> list:
        """Obtiene lista de alimentos recomendados según categoría"""
        
        # Base común
        alimentos_base = [
            "Pollo sin piel",
            "Pescado (salmón, trucha)",
            "Huevos",
            "Legumbres (lentejas, garbanzos)",
            "Vegetales de hoja verde",
            "Frutas bajas en glucemia",
            "Arroz integral",
            "Avena",
            "Frutos secos (almendras, nueces)"
        ]
        
        _concepto_ali = normalizar_objetivo(objetivo)
        if _concepto_ali == SUPERAVIT:
            return alimentos_base + [
                "Carnes rojas magras",
                "Productos lácteos enteros",
                "Plátanos",
                "Pasta integral",
                "Aceite de oliva"
            ]
        elif _concepto_ali == DEFICIT:
            return alimentos_base + [
                "Verduras sin almidón",
                "Yogur griego bajo en grasa",
                "Té verde",
                "Agua",
                "Especias (canela, jengibre)"
            ]
        else:  # MANTENIMIENTO
            return alimentos_base + [
                "Carbohidratos complejos",
                "Grasas insaturadas",
                "Variedad de proteínas"
            ]
    
    @staticmethod
    def get_alimentos_a_evitar(categoria_imc: str, objetivo: str) -> list:
        """Obtiene lista de alimentos a evitar según categoría"""
        
        alimentos_evitar = [
            "Azúcares refinados",
            "Bebidas azucaradas",
            "Alimentos ultraprocesados",
            "Grasas trans",
            "Frituras",
            "Alcohol en exceso"
        ]
        
        _concepto_evit = normalizar_objetivo(objetivo)
        if _concepto_evit == DEFICIT or categoria_imc in ["Sobrepeso", "Obesidad grado I", "Obesidad grado II", "Obesidad grado III"]:
            return alimentos_evitar + [
                "Productos lácteos enteros",
                "Carnes grasas",
                "Productos de panadería",
                "Chocolate y dulces",
                "Salsas altas en calorías"
            ]
        
        return alimentos_evitar
    
    @staticmethod
    def get_frecuencia_comidas(objetivo: str) -> str:
        """Recomienda frecuencia de comidas según objetivo"""
        _concepto_frec = normalizar_objetivo(objetivo)
        if _concepto_frec == SUPERAVIT:
            return "5-6 comidas al día (3 principales + 2-3 meriendas)"
        elif _concepto_frec == DEFICIT:
            return "3 comidas principales + 2 meriendas (controlar tamaño de porciones)"
        else:
            return "3 comidas principales + 1-2 meriendas (flexible)"
    
    @staticmethod
    def generar_notas(imc: float, categoria_imc: str, objetivo: str, edad: int) -> str:
        """Genera notas personalizadas basadas en el perfil"""
        
        notas = []
        
        # Notas por categoría IMC
        if categoria_imc == "Bajo peso":
            notas.append("⚠️ Tu IMC indica bajo peso. Consulta con un nutricionista para un plan personalizado.")
        elif categoria_imc == "Peso normal":
            notas.append("✅ Tu IMC es normal. Mantén hábitos saludables.")
        elif categoria_imc == "Sobrepeso":
            notas.append("⚠️ Tu IMC indica sobrepeso. Se recomienda déficit calórico moderado.")
        elif "Obesidad" in categoria_imc:
            notas.append("🚨 Tu IMC indica obesidad. Busca ayuda profesional para un plan personalizado.")
        
        # Notas por objetivo
        _concepto_nota = normalizar_objetivo(objetivo)
        if _concepto_nota == DEFICIT:
            notas.append("💪 Para perder peso: Come despacio, bebe agua, aumenta actividad física.")
        elif _concepto_nota == SUPERAVIT:
            notas.append("🏋️ Para ganar masa: Come en superávit, entrena con pesas, aumenta proteína.")
        
        # Notas por edad
        if edad > 50:
            notas.append("📋 Mayor de 50: Aumenta ingesta de calcio y Vit. D. Consulta médico antes de cambios drásticos.")
        elif edad < 18:
            notas.append("📋 Menor de 18: Necesita más calorías para crecimiento. Nutricionista recomendado.")
        
        notas.append("💡 Esta es una recomendación automática. Consulta un nutricionista para un plan personalizado.")
        
        return " | ".join(notas)
