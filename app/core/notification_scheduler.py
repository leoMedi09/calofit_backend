import random
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

# Frases motivacionales — lista fija que rota al azar, sin depender del LLM
# (cero costo de tokens, cero riesgo de fallo por cupo/conexión de Groq).
_FRASES_MOTIVACIONALES = [
    "Cada comida que registras es un paso más cerca de tu meta. ¡Vamos! 💪",
    "No se trata de ser perfecto, se trata de ser constante. Hoy también cuenta.",
    "Tu cuerpo escucha todo lo que le dices. Hoy dile algo bueno con tu entrenamiento.",
    "El progreso no se ve de un día para otro, pero se construye día a día. Sigue así.",
    "Recuerda por qué empezaste. Hoy es un buen día para recordarlo con acción.",
    "La disciplina vence a la motivación cuando esta se acaba. Hoy elige disciplina.",
    "Un entrenamiento más, una comida registrada más — así se construyen los resultados.",
    "No compitas con nadie más que con la versión de ayer de ti mismo.",
    "El descanso también es parte del progreso. Escucha a tu cuerpo, pero no le creas todas sus excusas.",
    "Cada gota de sudor de hoy es una inversión en el tú del futuro.",
    "Lo que haces hoy en el gym y en tu plato, lo agradecerás en unas semanas.",
    "No necesitas motivación todos los días, necesitas el hábito. Hoy suma uno más.",
    "Tu meta no se mueve. Solo necesitas seguir dando pasos, aunque sean pequeños.",
    "El cambio real pasa en los días en que no tienes ganas pero igual lo haces.",
    "Eres más fuerte que la excusa que estás pensando en este momento.",
    "Cuida tu cuerpo, es el único lugar que tienes para vivir toda tu vida.",
    "Hoy es una nueva oportunidad para acercarte un poco más a la persona que quieres ser.",
    "Pequeños esfuerzos consistentes ganan siempre a grandes esfuerzos esporádicos.",
]


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


def enviar_motivacion_diaria():
    """
    Job de mañana/tarde/noche: envía una frase motivacional aleatoria (de
    _FRASES_MOTIVACIONALES, sin depender del LLM) a todos los clientes con
    notificaciones activas y token FCM registrado. Independiente del
    recordatorio de las 20:00 (que solo avisa si no registró comida).
    """
    db = SessionLocal()
    try:
        clientes = (
            db.query(Client)
            .filter(Client.fcm_token.isnot(None))
            .filter(Client.notificaciones_activas.is_(True))
            .all()
        )

        frase = random.choice(_FRASES_MOTIVACIONALES)
        for cliente in clientes:
            enviado = send_push_notification(
                token=cliente.fcm_token,
                title="CaloFit 💪",
                body=frase,
                data={"tipo": "motivacional"},
            )
            if enviado:
                logger.info("Mensaje motivacional enviado a client_id=%s", cliente.id)
    except Exception as e:
        logger.error("Error en job de motivación diaria: %s", e)
    finally:
        db.close()


def iniciar_scheduler() -> BackgroundScheduler:
    scheduler = BackgroundScheduler(timezone=PERU_TZ)
    scheduler.add_job(
        enviar_motivacion_diaria,
        trigger=CronTrigger(hour=7, minute=0),
        id="motivacion_mañana",
        replace_existing=True,
    )
    scheduler.add_job(
        enviar_motivacion_diaria,
        trigger=CronTrigger(hour=13, minute=0),
        id="motivacion_tarde",
        replace_existing=True,
    )
    scheduler.add_job(
        enviar_motivacion_diaria,
        trigger=CronTrigger(hour=18, minute=0),
        id="motivacion_noche",
        replace_existing=True,
    )
    scheduler.add_job(
        revisar_clientes_sin_registro,
        trigger=CronTrigger(hour=20, minute=0),
        id="recordatorio_diario_comidas",
        replace_existing=True,
    )
    scheduler.start()
    logger.info(
        "Scheduler de notificaciones iniciado (motivación 7:00/13:00/18:00, "
        "recordatorio de registro 20:00, hora Perú)."
    )
    return scheduler
