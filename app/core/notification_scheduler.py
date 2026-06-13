from datetime import timezone, timedelta

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from app.core.database import SessionLocal
from app.core.firebase import send_push_notification
from app.core.utils import get_peru_date
from app.core.logging_config import get_logger
from app.models.client import Client
from app.models.comida_registro import ComidaRegistro

logger = get_logger("notification_scheduler")

PERU_TZ = timezone(timedelta(hours=-5))


def revisar_clientes_sin_registro():
    """
    Job diario: a las 20:00 (hora de Perú) revisa qué clientes con notificaciones
    activas y token FCM registrado aún no registraron ninguna comida hoy,
    y les envía un recordatorio push (RF12).
    """
    db = SessionLocal()
    try:
        hoy = get_peru_date()

        clientes = (
            db.query(Client)
            .filter(Client.fcm_token.isnot(None))
            .filter(Client.notificaciones_activas.is_(True))
            .all()
        )

        for cliente in clientes:
            tiene_registro = (
                db.query(ComidaRegistro)
                .filter(ComidaRegistro.client_id == cliente.id)
                .filter(ComidaRegistro.fecha == hoy)
                .first()
            )
            if tiene_registro:
                continue

            enviado = send_push_notification(
                token=cliente.fcm_token,
                title="¿Ya registraste tu alimentación de hoy?",
                body="No olvides registrar tus comidas para mantener tu progreso en CaloFit 💪",
                data={"tipo": "recordatorio_diario"},
            )
            if enviado:
                logger.info("Recordatorio diario enviado a client_id=%s", cliente.id)
    except Exception as e:
        logger.error("Error en job de notificaciones diarias: %s", e)
    finally:
        db.close()


def iniciar_scheduler() -> BackgroundScheduler:
    scheduler = BackgroundScheduler(timezone=PERU_TZ)
    scheduler.add_job(
        revisar_clientes_sin_registro,
        trigger=CronTrigger(hour=20, minute=0),
        id="recordatorio_diario_comidas",
        replace_existing=True,
    )
    scheduler.start()
    logger.info("Scheduler de notificaciones iniciado (recordatorio diario 20:00 Perú).")
    return scheduler
