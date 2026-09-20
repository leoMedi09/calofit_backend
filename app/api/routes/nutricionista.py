from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session, selectinload
from typing import List
from app.core.database import get_db
from app.models.client import Client
from app.models.user import User
from app.api.routes.auth import get_current_user
from app.services.ia_service import ia_service
from app.models.nutricion import PlanNutricional, PlanDiario
from app.models.historial import AlertaSalud
from app.schemas.nutricion import PlanNutricionalResponse, PlanNutricionalUpdate
from app.schemas.client import StrategicGuideUpdate
from datetime import datetime, timedelta
from app.core.utils import calcular_metabolismo_basal, obtener_macros_desglosados, get_peru_now
from app.schemas.client import StrategicGuideUpdate, ClientExpressCreate
from app.core.security import security

router = APIRouter()

_ROLES_PERMITIDOS = {
    "nutricionista",
    "nutritionist",
    "nutri",
    "admin",
    "administrador",
    "coach",
    "entrenador",
}


def check_is_nutri(current_user: User):
    role = str(getattr(current_user, "role_name", "")).lower()
    if role not in _ROLES_PERMITIDOS:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Operación permitida solo para Staff (Nutricionistas, Coaches o Admin)",
        )
    return current_user


def calcular_progreso_paciente(client: Client) -> float:
    """
    Calcula el progreso real (%) basado en la tendencia de peso y el objetivo.
    """
    if not client.historial_peso or len(client.historial_peso) < 1:
        return 50.0

    historial_ordenado = sorted(client.historial_peso, key=lambda x: x.fecha_registro)
    peso_inicial = historial_ordenado[0].peso_kg
    peso_actual = client.weight or peso_inicial
    objetivo = (client.goal or "Mantener peso").lower()

    if "perder" in objetivo:
        cambio = peso_inicial - peso_actual
        return min(100.0, max(0.0, 50.0 + (cambio * 2)))
    elif "ganar" in objetivo:
        cambio = peso_actual - peso_inicial
        return min(100.0, max(0.0, 50.0 + (cambio * 2)))
    else:
        variacion = abs(peso_actual - peso_inicial)
        return max(0.0, 100.0 - (variacion * 5))


@router.post("/clientes/express")
def create_express_patient(
    client_data: ClientExpressCreate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)
):
    """
    Crea un paciente de forma rápida ('Express') usando solo Correo y DNI.
    El DNI servirá como clave temporal y el usuario quedará Incompleto.
    """
    check_is_nutri(current_user)

    _ADMIN = {"admin", "administrador"}
    _NUTRI = {"nutricionista", "nutritionist", "nutri"}
    rol = str(getattr(current_user, "role_name", "")).lower()
    if rol not in _ADMIN | _NUTRI:
        raise HTTPException(status_code=403, detail="Solo nutricionistas o administradores pueden crear clientes.")

    nutri_id = client_data.assigned_nutri_id
    if nutri_id is None and rol in _NUTRI:
        nutri_id = current_user.id
    if nutri_id is not None:
        elegido = db.query(User).filter(User.id == nutri_id, User.is_active == True).first()
        if not elegido or str(getattr(elegido, "role_name", "")).lower() not in _NUTRI:
            raise HTTPException(status_code=400, detail="El nutricionista seleccionado no es válido.")

    if db.query(Client).filter(Client.email == client_data.email).first():
        raise HTTPException(status_code=400, detail="Este correo ya está registrado como paciente en el sistema.")
    if db.query(User).filter(User.email == client_data.email).first():
        raise HTTPException(status_code=400, detail="Este correo ya está registrado como staff en el sistema.")

    if db.query(Client).filter(Client.dni == client_data.dni).first():
        raise HTTPException(status_code=400, detail="Este DNI ya está registrado como paciente en el sistema.")

    try:
        from app.core.firebase import auth as firebase_admin_auth

        fb_user = firebase_admin_auth.create_user(
            email=client_data.email, password=client_data.dni, display_name="Paciente CaloFit"
        )
        flutter_uid = fb_user.uid
        print(f"✅ Usuario Firebase creado exitosamente para express: {flutter_uid}")

    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error al crear usuario en Firebase: {error_msg}")

        if "EMAIL_EXISTS" in error_msg:
            raise HTTPException(
                status_code=400, detail="Este correo electrónico ya está registrado en los servidores de CaloFit."
            )
        else:
            raise HTTPException(status_code=400, detail=f"Error validando la cuenta: {error_msg}")

    hashed_dni = security.hash_password(client_data.dni)

    nuevo_paciente = Client(
        email=client_data.email,
        dni=client_data.dni,
        first_name="",
        last_name_paternal="",
        last_name_maternal="",
        hashed_password=hashed_dni,
        flutter_uid=flutter_uid,
        is_profile_complete=False,
        assigned_nutri_id=nutri_id,
        assigned_coach_id=client_data.assigned_coach_id,
    )

    db.add(nuevo_paciente)
    db.commit()
    db.refresh(nuevo_paciente)

    from app.services.email_service import EmailService

    EmailService.send_welcome_credentials_brevo(
        email_to=client_data.email, dni=client_data.dni, nutricionista_name=current_user.first_name
    )

    return {"message": "Paciente Express creado exitosamente.", "client_id": nuevo_paciente.id}


@router.get("/clientes", response_model=List[dict])
def get_assigned_patients(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    check_is_nutri(current_user)

    query = db.query(Client)
    role = str(getattr(current_user, "role_name", "")).lower()

    if role in {"nutricionista", "nutritionist", "nutri"}:
        query = query.filter(Client.assigned_nutri_id == current_user.id)
    elif role in {"coach", "entrenador", "trainer"}:
        query = query.filter(
            Client.assigned_coach_id == current_user.id,
            Client.is_profile_complete == True,
        )

    clients = query.options(
        selectinload(Client.progreso_calorias),
        selectinload(Client.historial_peso),
        selectinload(Client.planes_nutricionales),
    ).all()

    now = get_peru_now()
    seven_days_ago = now - timedelta(days=7)

    result = []
    for c in clients:
        registros_recientes = [r for r in c.progreso_calorias if r.fecha >= seven_days_ago.date()]
        num_registros = len(registros_recientes)

        adherencia = round((num_registros / 7) * 100, 1)
        progreso = calcular_progreso_paciente(c)

        alerta_data = ia_service.generar_alerta_fuzzy(adherencia, progreso)

        now_naive = now.replace(tzinfo=None)
        created = c.created_at.replace(tzinfo=None) if c.created_at else None
        days_since_creation = (now_naive - created).days if created else 31
        is_new_user = days_since_creation < 30

        if is_new_user:
            hizo_checkin_peso = True
        else:
            last_record = max((r.fecha_registro for r in c.historial_peso), default=None)
            days_since = (now.date() - last_record).days if last_record else days_since_creation
            hizo_checkin_peso = days_since < 30

        result.append(
            {
                "id": c.id,
                "full_name": f"{c.first_name} {c.last_name_paternal} {c.last_name_maternal}",
                "email": c.email,
                "goal": c.goal,
                "weight": c.weight,
                "nutri_id": c.assigned_nutri_id,
                "coach_id": c.assigned_coach_id,
                "adherencia": adherencia,
                "alerta": alerta_data.get("mensaje", ""),
                "alerta_nivel": alerta_data.get("nivel", "Bajo"),
                "gender": c.gender,
                "is_validated": c.is_strategic_guide_validated,
                "is_profile_complete": c.is_profile_complete,
                "dni": c.dni,
                "hizo_checkin_peso": hizo_checkin_peso,
                "semana_status": _calcular_mes_status(c, db),
            }
        )

    return result


def _calcular_mes_status(c: Client, db: Session) -> str:
    """
    Determina el estado del plan del paciente:
    - 'validado': Último plan aprobado por nutricionista.
    - 'pendiente': Sin plan o plan sin validar.

    Nota de optimización: usa la relación `planes_nutricionales` ya cargada
    (selectinload) en memoria en vez de una query por cliente dentro de bucles.
    """
    planes = c.planes_nutricionales or []
    ultimo_plan = max(
        planes,
        key=lambda p: p.fecha_creacion,
        default=None,
    )

    if ultimo_plan and ultimo_plan.status == "validado":
        return "validado"
    return "pendiente"


@router.get("/cliente/{id}/progreso")
def get_patient_progress(id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    check_is_nutri(current_user)

    client = db.query(Client).filter(Client.id == id).first()
    if not client:
        raise HTTPException(status_code=404, detail="Cliente no encontrado")

    if (
        current_user.role_name.upper() in ["NUTRICIONISTA", "NUTRITIONIST"]
        and client.assigned_nutri_id != current_user.id
    ):
        raise HTTPException(status_code=403, detail="No tienes permiso para ver este paciente")

    historial_peso = [{"fecha": h.fecha_registro, "valor": h.peso_kg} for h in client.historial_peso]
    historial_imc = [{"fecha": h.fecha_registro, "valor": h.imc} for h in client.historial_imc]

    from app.models.historial import ProgresoCalorias
    from app.core.utils import get_peru_date

    hoy = get_peru_date()
    progreso_hoy = (
        db.query(ProgresoCalorias)
        .filter(
            ProgresoCalorias.client_id == id,
            ProgresoCalorias.fecha == hoy,
        )
        .first()
    )
    today_summary = {
        "calorias_consumidas": int(progreso_hoy.calorias_consumidas or 0) if progreso_hoy else 0,
        "calorias_quemadas": int(progreso_hoy.calorias_quemadas or 0) if progreso_hoy else 0,
        "proteinas": round(float(progreso_hoy.proteinas_consumidas or 0), 1) if progreso_hoy else 0.0,
        "carbos": round(float(progreso_hoy.carbohidratos_consumidos or 0), 1) if progreso_hoy else 0.0,
        "grasas": round(float(progreso_hoy.grasas_consumidas or 0), 1) if progreso_hoy else 0.0,
    }

    alertas = [
        {
            "id": a.id,
            "tipo": a.tipo,
            "descripcion": a.descripcion,
            "severidad": a.severidad,
            "estado": a.estado,
            "fecha": a.fecha_deteccion,
        }
        for a in client.alertas_salud
    ]

    tmb_estimada = calcular_metabolismo_basal(client)

    _GOAL_FACTOR = {
        "perder peso": 0.85,
        "perder_leve": 0.90,
        "ganar masa": 1.10,
        "ganar_leve": 1.05,
        "mantener peso": 1.0,
    }
    goal_key = (client.goal or "mantener peso").lower().strip()
    calorias_ajustadas = tmb_estimada * _GOAL_FACTOR.get(goal_key, 1.0)

    peso_cliente = float(client.weight or 70.0)
    recomendacion_ia = obtener_macros_desglosados(calorias_ajustadas, client.goal, peso_cliente)

    ultimo_plan_prog = (
        db.query(PlanNutricional)
        .filter(PlanNutricional.client_id == id)
        .order_by(PlanNutricional.fecha_creacion.desc())
        .first()
    )

    return {
        "id": client.id,
        "nombre": f"{client.first_name} {client.last_name_paternal} {client.last_name_maternal}",
        "objetivo": client.goal,
        "focus_objetivo": client.ai_strategic_focus,
        "semana_status": _calcular_mes_status(client, db),
        "plan_validated_at": str(ultimo_plan_prog.validated_at)
        if ultimo_plan_prog and ultimo_plan_prog.validated_at
        else None,
        "historial_peso": historial_peso,
        "historial_imc": historial_imc,
        "alertas_salud": alertas,
        "ai_strategic_focus": client.ai_strategic_focus,
        "is_strategic_guide_validated": client.is_strategic_guide_validated,
        "is_validated": client.is_strategic_guide_validated,
        "metabolismo_estimado": {
            "tmb": round(tmb_estimada),
            "calorias_objetivo": recomendacion_ia["calorias"],
            "proteinas_g": recomendacion_ia["proteinas_g"],
            "carbohidratos_g": recomendacion_ia["carbohidratos_g"],
            "grasas_g": recomendacion_ia["grasas_g"],
            "distribucion": recomendacion_ia["pct"],
        },
        "current_weight": client.weight,
        "current_height": client.height,
        "recommended_foods": client.recommended_foods,
        "forbidden_foods": client.forbidden_foods,
        "medical_conditions": client.medical_conditions,
        "coach_notes": client.coach_notes,
        "nutri_weekly_note": client.nutri_weekly_note,
        "today_summary": today_summary,
    }


@router.get("/cliente/{id}/registro-diario")
def get_patient_daily_log(
    id: int, fecha: str | None = None, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)
):
    """Devuelve el detalle de comidas y ejercicios registrados por el cliente en una fecha dada (default: hoy, hora Perú)."""
    check_is_nutri(current_user)

    client = db.query(Client).filter(Client.id == id).first()
    if not client:
        raise HTTPException(status_code=404, detail="Cliente no encontrado")

    if (
        current_user.role_name.upper() in ["NUTRICIONISTA", "NUTRITIONIST"]
        and client.assigned_nutri_id != current_user.id
    ):
        raise HTTPException(status_code=403, detail="No tienes permiso para ver este paciente")

    from app.models.comida_registro import ComidaRegistro
    from app.core.utils import get_peru_date
    from sqlalchemy import text as _sql

    fecha_consulta = datetime.strptime(fecha, "%Y-%m-%d").date() if fecha else get_peru_date()

    comidas = (
        db.query(ComidaRegistro)
        .filter(
            ComidaRegistro.client_id == id,
            ComidaRegistro.fecha == fecha_consulta,
        )
        .order_by(ComidaRegistro.created_at.asc())
        .all()
    )

    comidas_list = [
        {
            "id": c.id,
            "nombre": c.nombre_alimento,
            "momento": c.momento,
            "gramos": c.gramos,
            "kcal": c.kcal,
            "proteina_g": c.proteina_g,
            "carbohidratos_g": c.carbohidratos_g,
            "grasas_g": c.grasas_g,
            "hora": c.created_at,
        }
        for c in comidas
    ]

    ejercicios = db.execute(
        _sql(
            "SELECT ejercicio, series, reps, peso_kg, created_at, "
            "       calorias_quemadas, session_duration_min, intensity "
            "FROM workout_logs WHERE client_id = :cid AND DATE(created_at) = :fecha "
            "ORDER BY created_at ASC"
        ),
        {"cid": id, "fecha": fecha_consulta},
    ).fetchall()

    ejercicios_list = [
        {
            "ejercicio": e.ejercicio,
            "series": e.series,
            "reps": e.reps,
            "peso_kg": e.peso_kg,
            "calorias_quemadas": e.calorias_quemadas,
            "session_duration_min": e.session_duration_min,
            "intensity": e.intensity,
            "hora": e.created_at,
        }
        for e in ejercicios
    ]

    return {
        "fecha": fecha_consulta,
        "comidas": comidas_list,
        "ejercicios": ejercicios_list,
    }


@router.get("/cliente/{id}/sugerir-estrategia")
async def suggest_strategic_guide(
    id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)
):
    check_is_nutri(current_user)

    client = db.query(Client).filter(Client.id == id).first()
    if not client:
        raise HTTPException(status_code=404, detail="Cliente no encontrado")

    date_limit = datetime.utcnow() - timedelta(days=15)
    alertas_recent = [a for a in client.alertas_salud if a.fecha_deteccion >= date_limit]

    alertas_list = [{"tipo": a.tipo, "descripcion": a.descripcion, "severidad": a.severidad} for a in alertas_recent]

    imc = 0
    if client.weight and client.height:
        height_m = client.height / 100
        imc = round(client.weight / (height_m * height_m), 1)

    edad = 0
    if client.birth_date:
        today = datetime.now()
        edad = (
            today.year
            - client.birth_date.year
            - ((today.month, today.day) < (client.birth_date.month, client.birth_date.day))
        )

    historial_peso = [{"fecha": h.fecha_registro, "valor": h.peso_kg} for h in client.historial_peso]

    perfil = {
        "full_name": f"{client.first_name} {client.last_name_paternal}",
        "gender": "Hombre" if client.gender == "M" else "Mujer",
        "age": edad,
        "current_weight": client.weight,
        "current_height": client.height,
        "imc": imc,
        "activity_level": client.activity_level or "Moderado",
        "goal": client.goal,
        "medical_conditions": client.medical_conditions or [],
        "weight_history": historial_peso,
    }

    sugerencia = await ia_service.sugerir_guia_estrategica(perfil, alertas_list)
    return sugerencia


@router.post("/actualizar-guia-estrategica/{id}")
def update_strategic_guide(
    id: int, guide: StrategicGuideUpdate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)
):
    check_is_nutri(current_user)

    client = db.query(Client).filter(Client.id == id).first()
    if not client:
        raise HTTPException(status_code=404, detail="Cliente no encontrado")

    if (
        current_user.role_name.upper() in ["NUTRICIONISTA", "NUTRITIONIST"]
        and client.assigned_nutri_id != current_user.id
    ):
        raise HTTPException(status_code=403, detail="No tienes permiso para modificar este paciente")

    if guide.ai_strategic_focus is not None:
        client.ai_strategic_focus = guide.ai_strategic_focus
    if guide.recommended_foods is not None:
        client.recommended_foods = guide.recommended_foods
    if guide.forbidden_foods is not None:
        client.forbidden_foods = guide.forbidden_foods
    if guide.medical_conditions is not None:
        client.medical_conditions = guide.medical_conditions
    if guide.nutri_weekly_note is not None:
        client.nutri_weekly_note = guide.nutri_weekly_note
    if guide.workout_type is not None:
        client.workout_type = guide.workout_type
    if guide.session_duration is not None:
        client.session_duration = guide.session_duration

    plan_activo = (
        db.query(PlanNutricional)
        .filter(PlanNutricional.client_id == id)
        .order_by(PlanNutricional.fecha_creacion.desc())
        .first()
    )
    if plan_activo:
        plan_activo.validated_at = get_peru_now().replace(tzinfo=None)
        plan_activo.validated_by_id = current_user.id
        if plan_activo.status != "validado":
            plan_activo.status = "validado"

    db.commit()
    return {"status": "success", "message": "Guía estratégica actualizada para la IA"}


@router.post("/validar-plan/{id}")
def validate_plan(id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    check_is_nutri(current_user)

    client = db.query(Client).filter(Client.id == id).first()
    if not client:
        raise HTTPException(status_code=404, detail="Cliente no encontrado")

    plan = (
        db.query(PlanNutricional)
        .filter(PlanNutricional.client_id == id)
        .order_by(PlanNutricional.fecha_creacion.desc())
        .first()
    )
    if not plan:
        plan = PlanNutricional(
            client_id=id,
            genero=1 if (client.gender or "M") == "M" else 2,
            edad=25,
            peso=client.weight or 70.0,
            talla=client.height or 170.0,
            nivel_actividad=1.55,
            objetivo=client.goal or "Mantener peso",
            status="draft",
            calorias_ia_base=0,
        )
        db.add(plan)
        db.flush()

        for i in range(1, 8):
            db.add(
                PlanDiario(
                    plan_id=plan.id,
                    dia_numero=i,
                    calorias_dia=0,
                    proteinas_g=0,
                    carbohidratos_g=0,
                    grasas_g=0,
                    estado="sugerencia_ia",
                )
            )
        db.flush()

    plan.status = "validado"
    plan.validated_by_id = current_user.id
    plan.validated_at = get_peru_now().replace(tzinfo=None)
    db.commit()
    return {"message": f"Plan del cliente {id} validado correctamente por Nutricionista {current_user.first_name}"}


@router.get("/cliente/{id}/plan", response_model=PlanNutricionalResponse)
def get_client_plan(id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    check_is_nutri(current_user)
    client = db.query(Client).filter(Client.id == id).first()
    if not client:
        raise HTTPException(status_code=404, detail="Cliente no encontrado")

    if (
        current_user.role_name.upper() in ["NUTRICIONISTA", "NUTRITIONIST"]
        and client.assigned_nutri_id != current_user.id
    ):
        raise HTTPException(status_code=403, detail="No tienes permiso para ver este paciente")

    plan = (
        db.query(PlanNutricional)
        .filter(PlanNutricional.client_id == id)
        .order_by(PlanNutricional.fecha_creacion.desc())
        .first()
    )

    if not plan:
        print(f"📡 Generando vista previa de plan para cliente nuevo {id}")
        return {
            "id": 0,
            "client_id": id,
            "objetivo": client.goal or "Por definir",
            "status": "draft",
            "observaciones": "Plan inicial pendiente de configuración",
            "detalles_diarios": [
                {
                    "dia_numero": i,
                    "calorias_dia": 0,
                    "proteinas_g": 0,
                    "carbohidratos_g": 0,
                    "grasas_g": 0,
                    "estado": "pendiente",
                }
                for i in range(1, 8)
            ],
        }
    return plan


@router.put("/cliente/{id}/plan")
def update_client_plan(
    id: int,
    plan_update: PlanNutricionalUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    check_is_nutri(current_user)
    client = db.query(Client).filter(Client.id == id).first()
    if not client:
        raise HTTPException(status_code=404, detail="Cliente no encontrado")

    if (
        current_user.role_name.upper() in ["NUTRICIONISTA", "NUTRITIONIST"]
        and client.assigned_nutri_id != current_user.id
    ):
        raise HTTPException(status_code=403, detail="No tienes permiso")

    plan = (
        db.query(PlanNutricional)
        .filter(PlanNutricional.client_id == id)
        .order_by(PlanNutricional.fecha_creacion.desc())
        .first()
    )

    if not plan:
        print(f"✨ Creando primer plan para cliente {id} (Probablemente Express)")
        plan = PlanNutricional(
            client_id=id,
            genero=1 if (client.gender or "M") == "M" else 2,
            edad=25,
            peso=client.weight or 70.0,
            talla=client.height or 170.0,
            nivel_actividad=1.55,
            objetivo=client.goal or "Mantener peso",
            status="draft",
            calorias_ia_base=0,
        )
        db.add(plan)
        db.flush()
        for i in range(1, 8):
            dia = PlanDiario(
                plan_id=plan.id,
                dia_numero=i,
                calorias_dia=0,
                proteinas_g=0,
                carbohidratos_g=0,
                grasas_g=0,
                estado="sugerencia_ia",
            )
            db.add(dia)
        db.flush()

    if plan_update.objetivo:
        plan.objetivo = plan_update.objetivo
    if plan_update.observaciones:
        plan.observaciones = plan_update.observaciones
    if plan_update.status:
        plan.status = plan_update.status

    if plan_update.detalles_diarios:
        for i, daily_update in enumerate(plan_update.detalles_diarios):
            if i < len(plan.detalles_diarios):
                daily = plan.detalles_diarios[i]
                if daily_update.calorias_dia is not None:
                    daily.calorias_dia = daily_update.calorias_dia
                if daily_update.proteinas_g is not None:
                    daily.proteinas_g = daily_update.proteinas_g
                if daily_update.carbohidratos_g is not None:
                    daily.carbohidratos_g = daily_update.carbohidratos_g
                if daily_update.grasas_g is not None:
                    daily.grasas_g = daily_update.grasas_g
                daily.estado = daily_update.estado or "oficial"

    db.commit()
    return {"message": "Plan actualizado exitosamente"}


@router.get("/stats")
def get_nutri_stats(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    check_is_nutri(current_user)

    _ROLES_NUTRI = {"nutricionista", "nutritionist", "nutri"}
    _ROLES_COACH = {"coach", "entrenador", "trainer"}

    role = str(getattr(current_user, "role_name", "")).lower()
    query = db.query(Client)
    if role in _ROLES_NUTRI:
        query = query.filter(Client.assigned_nutri_id == current_user.id)
    elif role in _ROLES_COACH:
        query = query.filter(
            Client.assigned_coach_id == current_user.id,
            Client.is_profile_complete == True,
        )

    pacientes = query.options(
        selectinload(Client.progreso_calorias),
        selectinload(Client.historial_peso),
    ).all()
    total_pacientes = len(pacientes)
    paciente_ids = {c.id for c in pacientes}

    if total_pacientes == 0:
        return {
            "total_pacientes": 0,
            "validaciones_pendientes": 0,
            "alertas_criticas": 0,
            "adherencia_media": 0.0,
            "tendencia_adherencia": [0, 0, 0, 0, 0, 0, 0],
            "alertas_recientes": [],
        }

    validaciones_pendientes = (
        db.query(PlanNutricional)
        .join(Client)
        .filter(
            Client.id.in_(paciente_ids),
            PlanNutricional.status == "provisional_ia",
        )
        .count()
    )

    thirty_days_ago = datetime.now() - timedelta(days=30)
    seven_days_ago = datetime.now() - timedelta(days=7)

    alertas_db_query = (
        db.query(AlertaSalud)
        .filter(
            AlertaSalud.estado == "pendiente",
            AlertaSalud.fecha_deteccion >= thirty_days_ago,
        )
        .join(Client)
        .filter(Client.id.in_(paciente_ids))
        .options(selectinload(AlertaSalud.cliente))
    )
    alertas_db_count = alertas_db_query.count()

    pacientes_con_alerta_db = {a.cliente.id for a in alertas_db_query.all() if a.cliente}

    alertas_recientes_objs = alertas_db_query.order_by(AlertaSalud.fecha_deteccion.desc()).limit(5).all()
    alertas_formateadas = [
        {
            "id": a.id,
            "paciente": f"{a.cliente.first_name} {a.cliente.last_name_paternal}",
            "problema": a.descripcion,
            "urgency": a.severidad.capitalize(),
            "tipo": a.tipo,
        }
        for a in alertas_recientes_objs
    ]

    alertas_ia = 0
    for c in pacientes:
        if c.id in pacientes_con_alerta_db or not c.is_profile_complete:
            continue
        dias = len({r.fecha for r in c.progreso_calorias if r.fecha >= seven_days_ago.date()})
        if dias / 7 < 0.3:
            alertas_ia += 1
            alertas_formateadas.append(
                {
                    "id": 0,
                    "paciente": f"{c.first_name} {c.last_name_paternal}",
                    "problema": "Sin registros esta semana" if dias == 0 else f"Solo {dias} de 7 días con registro",
                    "urgency": "Alta" if dias == 0 else "Media",
                    "tipo": "progreso",
                    "_dias": dias,
                }
            )

    _rango = {"Alta": 0, "Media": 1, "Baja": 2}
    alertas_formateadas.sort(key=lambda x: (_rango.get(x["urgency"], 3), x.get("_dias", 0)))
    alertas_formateadas = [{k: v for k, v in x.items() if k != "_dias"} for x in alertas_formateadas[:30]]

    total_alertas = alertas_db_count + alertas_ia

    tendencia = []
    total_adh_sum = 0.0
    for i in range(6, -1, -1):
        target_date = (datetime.now() - timedelta(days=i)).date()
        conteo_dia = sum(1 for c in pacientes if any(r.fecha == target_date for r in c.progreso_calorias))
        adh_dia = round((conteo_dia / total_pacientes) * 100, 1)
        tendencia.append(adh_dia)
        total_adh_sum += adh_dia

    adherencia_media = round(total_adh_sum / 7, 1)

    return {
        "total_pacientes": total_pacientes,
        "validaciones_pendientes": validaciones_pendientes,
        "alertas_criticas": total_alertas,
        "alertas_recientes": alertas_formateadas,
        "adherencia_media": adherencia_media,
        "tendencia_adherencia": tendencia,
    }


@router.delete("/cliente/{id}")
def delete_client(id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """
    Elimina permanentemente a un cliente:
      1. Borra el usuario de Firebase Authentication (por flutter_uid).
      2. Elimina el registro de la BD (cascade borra historial, planes, etc.).
    Solo el Nutricionista asignado o un Admin pueden ejecutar esta acción.
    """
    check_is_nutri(current_user)

    client = db.query(Client).filter(Client.id == id).first()
    if not client:
        raise HTTPException(status_code=404, detail="Cliente no encontrado.")

    if current_user.role_name.upper() in ["NUTRICIONISTA", "NUTRITIONIST"]:
        if client.assigned_nutri_id != current_user.id:
            raise HTTPException(status_code=403, detail="No tienes permiso para eliminar este paciente.")

    flutter_uid = client.flutter_uid
    if flutter_uid:
        try:
            from firebase_admin import auth as firebase_auth

            firebase_auth.delete_user(flutter_uid)
            print(f"🔥 Firebase: Usuario {flutter_uid} eliminado correctamente.")
        except Exception as e:
            print(f"⚠️ Firebase: No se pudo eliminar el usuario ({e}). Continuando con la BD...")

    planes = db.query(PlanNutricional).filter(PlanNutricional.client_id == id).all()
    for plan in planes:
        db.query(PlanDiario).filter(PlanDiario.plan_id == plan.id).delete(synchronize_session=False)
    db.query(PlanNutricional).filter(PlanNutricional.client_id == id).delete(synchronize_session=False)
    db.flush()

    db.delete(client)
    db.commit()

    return {"status": "success", "message": f"Cliente ID {id} eliminado de la plataforma y Firebase."}


@router.put("/cliente/{id}/nota-entrenador")
def save_coach_note(
    id: int,
    payload: dict,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Guarda o actualiza la nota del entrenador sobre un cliente."""
    check_is_nutri(current_user)
    client = db.query(Client).filter(Client.id == id).first()
    if not client:
        raise HTTPException(status_code=404, detail="Cliente no encontrado.")
    client.coach_notes = payload.get("nota", "").strip() or None
    db.commit()
    return {"status": "ok"}


@router.get("/nutricionistas", response_model=List[dict])
def get_nutricionistas_list(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Nutricionistas activos, para elegir a quién asignar un cliente al crearlo."""
    check_is_nutri(current_user)

    _ROLES_NUTRI = {"nutricionista", "nutritionist", "nutri"}
    result = []
    for u in db.query(User).filter(User.is_active == True).all():
        if str(getattr(u, "role_name", "")).lower() not in _ROLES_NUTRI:
            continue
        full_name = f"{u.first_name or ''} {u.last_name_paternal or ''}".strip()
        result.append(
            {
                "id": u.id,
                "full_name": full_name or u.email,
                "email": u.email,
                "profile_picture_url": getattr(u, "profile_picture_url", None),
                "pacientes_count": len(u.clients_as_nutri),
            }
        )
    return result


@router.get("/coaches", response_model=List[dict])
def get_coaches_list(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Retorna la lista de entrenadores activos para que el nutri pueda asignarlos al crear un paciente."""
    check_is_nutri(current_user)

    _ROLES_COACH = {"coach", "entrenador", "trainer"}
    coaches = db.query(User).filter(User.is_active == True).all()

    result = []
    for u in coaches:
        role = str(getattr(u, "role_name", "")).lower()
        if role not in _ROLES_COACH:
            continue
        full_name = f"{u.first_name or ''} {u.last_name_paternal or ''}".strip()
        result.append(
            {
                "id": u.id,
                "full_name": full_name or u.email,
                "email": u.email,
                "profile_picture_url": getattr(u, "profile_picture_url", None),
                "pacientes_count": len(u.clients_as_coach),
            }
        )

    return result
