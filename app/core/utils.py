from datetime import datetime, date, timedelta, timezone

def get_peru_now() -> datetime:
    """Retorna la fecha y hora actual en zona horaria de Perú (UTC-5)"""
    # UTC now
    utc_now = datetime.now(timezone.utc)
    # Peru is UTC-5
    peru_time = utc_now - timedelta(hours=5)
    return peru_time

def get_peru_date() -> date:
    """Retorna la fecha actual en Perú"""
    return get_peru_now().date()
