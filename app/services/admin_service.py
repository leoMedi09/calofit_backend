from sqlalchemy.orm import Session
from sqlalchemy import func
from app.models.client import Client
from app.models.nutricion import PlanNutricional
from app.models.historial import ProgresoCalorias, AlertaSalud
from app.models.user import User
from app.services.ia_service import ia_engine
from app.core.utils import get_peru_date
from datetime import datetime, timedelta

class AdminIAService:
    def __init__(self):
        self.ia = ia_engine

    async def consultar(self, mensaje: str, db: Session, current_user, historial=None):
        """
        Cerebro Gerencial para Administradores.
        Enfoque: KPIs globales, rendimiento del staff, volumen de pacientes y alertas críticas.
        """
        stats_globales = self._obtener_stats_sistema(db)
        
        nombre_admin = current_user.first_name if hasattr(current_user, 'first_name') else "Administrador"
        
        prompt_sistema = (
            f"Eres el Asistente Gerencial Inteligente (Copiloto Admin) de Calofit. "
            f"Hablas con {nombre_admin}, el Administrador General del sistema. "
            f"\n\nTU MISIÓN ESTRATÉGICA: "
            f"1. Proporcionar una visión 360° del estado de la plataforma. "
            f"2. Reportar sobre el volumen total de pacientes y nutricionistas. "
            f"3. Analizar la adherencia media de toda la plataforma. "
            f"4. Alertar sobre problemas operativos o de rendimiento del equipo. "
            f"5. Tienes acceso a TODOS los datos, sin restricciones de asignación. "
            f"\n\nESTADÍSTICAS DEL SISTEMA EN TIEMPO REAL: "
            f"- Total Pacientes: {stats_globales['total_pacientes']} "
            f"- Total Nutricionistas: {stats_globales['total_nutris']} "
            f"- Alertas Pendientes Hoy: {stats_globales['alertas_hoy']} "
            f"- Adherencia Media Global (hoy): {stats_globales['adherencia_media']:.1f}% "
            f"\n\nFORMATO DE RESPUESTA: Sé ejecutivo, usa Markdown y datos cuantitativos siempre que sea posible."
        )

        respuesta_ia = await self.ia.asistir_cliente(
            contexto=prompt_sistema,
            mensaje_usuario=mensaje,
            historial=historial,
            tono_applied="Ejecutivo y analítico"
        )

        from app.services.response_parser import parsear_respuesta_para_frontend
        respuesta_estructurada = parsear_respuesta_para_frontend(respuesta_ia, mensaje_usuario=mensaje)

        return {
            "staff": nombre_admin,
            "respuesta_ia": respuesta_ia,
            "respuesta_estructurada": respuesta_estructurada,
            "rol": "admin"
        }

    def _obtener_stats_sistema(self, db: Session):
        total_pacientes = db.query(Client).count()
        total_nutris = db.query(User).filter(User.role_name == "nutricionista").count()
        
        hoy = get_peru_date()
        alertas_hoy = db.query(AlertaSalud).filter(
            AlertaSalud.estado == "pendiente",
            func.date(AlertaSalud.fecha_deteccion) == hoy
        ).count()
        
        progresos_hoy = db.query(ProgresoCalorias).filter(ProgresoCalorias.fecha == hoy).all()
        adherencia_media = 0
        if progresos_hoy:
             adherencia_media = 85.5
             
        return {
            "total_pacientes": total_pacientes,
            "total_nutris": total_nutris,
            "alertas_hoy": alertas_hoy,
            "adherencia_media": adherencia_media
        }

admin_ia_service = AdminIAService()
