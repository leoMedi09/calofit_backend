from fastapi import APIRouter, Depends, HTTPException, File, UploadFile
from app.core.local_storage import local_storage
from sqlalchemy.orm import Session
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from app.core.database import get_db
from app.models.client import Client
from app.models.user import User
from app.schemas.client import ClientCreate, ClientUpdate, ClientResponse, ChangePassword, AdminCreateClient
from app.schemas.dieta import ClientResponseConDieta, RecomendacionDietaCompleta
from app.core.security import security
from app.api.routes.auth import get_current_user, get_current_staff
from app.services.calculador_dieta import CalculadorDietaAutomatica
from datetime import date
from app.services.email_service import EmailService
import random
from datetime import datetime, timedelta
from app.core.firebase import auth as firebase_admin_auth
from app.core.logging_config import get_logger

logger = get_logger("api.clientes")


router = APIRouter()


@router.post("/admin-crear", summary="Admin crea un cliente con credenciales mínimas")
def admin_crear_cliente(
    data: AdminCreateClient,
    db: Session = Depends(get_db),
    current_staff: User = Depends(get_current_staff),
):
    """
    Endpoint exclusivo para Admins.
    Crea un cliente solo con email + contraseña + firebase_uid.
    El cliente completará su perfil en el Onboarding al primer login.
    """
    if not (current_staff.role and current_staff.role.name.lower() in ['admin', 'superadmin', 'nutritionist', 'nutricionista']):
        raise HTTPException(status_code=403, detail="Solo los administradores o nutricionistas pueden crear clientes")

    existe = db.query(Client).filter(Client.email == data.email).first()
    if existe:
        raise HTTPException(status_code=400, detail="Este correo ya está registrado")

    nuevo = Client(
        first_name="",
        last_name_paternal="",
        last_name_maternal="",
        email=data.email,
        hashed_password=security.hash_password(data.password),
        flutter_uid=data.flutter_uid,
        gender="M",
        weight=0.0,
        height=0.0,
        activity_level="Sedentario",
        goal="Mantener peso",
        medical_conditions=[],
        assigned_nutri_id=data.assigned_nutri_id,
        assigned_coach_id=data.assigned_coach_id,
        is_profile_complete=False,
    )
    db.add(nuevo)
    db.commit()
    db.refresh(nuevo)
    return {"id": nuevo.id, "email": nuevo.email, "is_profile_complete": False}




@router.post("/registrar")
def registrar_cliente(cliente_data: ClientCreate, db: Session = Depends(get_db)):
    """Registra un nuevo cliente en el sistema"""
    
    existe = db.query(Client).filter(Client.email == cliente_data.email).first()
    if existe:
        raise HTTPException(status_code=400, detail="El email ya está registrado")
    
    if cliente_data.assigned_coach_id is not None:
        coach = db.query(User).filter(User.id == cliente_data.assigned_coach_id).first()
        if not coach:
            raise HTTPException(status_code=400, detail="El coach asignado no existe")
    
    if cliente_data.assigned_nutri_id is not None:
        nutri = db.query(User).filter(User.id == cliente_data.assigned_nutri_id).first()
        if not nutri:
            raise HTTPException(status_code=400, detail="El nutricionista asignado no existe")
    
    nuevo_cliente = Client(
        first_name=cliente_data.first_name,
        last_name_paternal=cliente_data.last_name_paternal,
        last_name_maternal=cliente_data.last_name_maternal,
        email=cliente_data.email,
        hashed_password=security.hash_password(cliente_data.password),
        birth_date=cliente_data.birth_date,
        weight=cliente_data.weight,
        height=cliente_data.height,
        gender=cliente_data.gender,
        medical_conditions=cliente_data.medical_conditions or [],
        activity_level=cliente_data.activity_level or 'Sedentario',
        goal=cliente_data.goal or 'Mantener peso',
        assigned_coach_id=cliente_data.assigned_coach_id,
        assigned_nutri_id=cliente_data.assigned_nutri_id,
        flutter_uid=cliente_data.flutter_uid
    )
    
    try:
        db.add(nuevo_cliente)
        db.commit()
        db.refresh(nuevo_cliente)
        
        print(f"🤖 Generando plan automático para {nuevo_cliente.email}...")
        from app.services.ia_service import ia_engine
        from app.models.nutricion import PlanNutricional, PlanDiario
        
        edad = (date.today() - nuevo_cliente.birth_date).days // 365 if nuevo_cliente.birth_date else 25
        
        plan_data = ia_engine.generar_plan_inicial_automatico({
            "genero": nuevo_cliente.gender,
            "edad": edad,
            "peso": nuevo_cliente.weight,
            "talla": nuevo_cliente.height,
            "nivel_actividad": nuevo_cliente.activity_level,
            "objetivo": nuevo_cliente.goal
        })
        
        if plan_data:
            plan_maestro = PlanNutricional(
                client_id=nuevo_cliente.id,
                genero=1 if nuevo_cliente.gender == "M" else 2,
                edad=edad,
                peso=nuevo_cliente.weight,
                talla=nuevo_cliente.height,
                nivel_actividad=1.55,
                objetivo=nuevo_cliente.goal,
                es_contingencia_ia=False,
                calorias_ia_base=plan_data["calorias_diarias"],
                status="draft_ia",
                validated_by_id=None,
                validated_at=None
            )
            db.add(plan_maestro)
            db.flush()
            
            for dia_info in plan_data["dias"]:
                plan_dia = PlanDiario(
                    plan_id=plan_maestro.id,
                    dia_numero=dia_info["dia_numero"],
                    calorias_dia=dia_info["calorias_dia"],
                    proteinas_g=dia_info["proteinas_g"],
                    carbohidratos_g=dia_info["carbohidratos_g"],
                    grasas_g=dia_info["grasas_g"],
                    sugerencia_entrenamiento_ia=dia_info["sugerencia_entrenamiento_ia"],
                    nota_asistente_ia=dia_info["nota_asistente_ia"],
                    validado_nutri=False,
                    estado="sugerencia_ia"
                )
                db.add(plan_dia)
            
            db.commit()
            print(f"✅ Plan automático creado (ID: {plan_maestro.id}) con {len(plan_data['dias'])} días")
            
            return {
                **nuevo_cliente.__dict__,
                "plan_generado": True,
                "plan_info": {
                    "id": plan_maestro.id,
                    "calorias_diarias": plan_data["calorias_diarias"],
                    "macros": plan_data["macros"],
                    "mensaje": "¡Tu plan nutricional ha sido generado automáticamente! 🎉"
                }
            }
        else:
            print("⚠️ No se pudo generar plan automático, pero registro exitoso")
            return nuevo_cliente
            
    except IntegrityError as e:
        db.rollback()
        logger.error("Error de integridad en registro de cliente: %s", e, exc_info=True)
        raise HTTPException(status_code=400, detail="Error de integridad en la base de datos al registrar el cliente")
    except Exception as e:
        db.rollback()
        logger.error("Error interno en registro de cliente: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Error interno del servidor")


@router.get("/perfil")
def obtener_perfil_cliente(
    current_user = Depends(get_current_user),
):
    """Obtiene el perfil del cliente autenticado"""
    
    print(f"🔍 GET /clientes/perfil llamado")
    print(f"🔍 Tipo de usuario: {type(current_user).__name__}")
    print(f"🔍 ID: {current_user.id}")
    print(f"🔍 Email: {current_user.email}")
    
    if not isinstance(current_user, Client):
        print(f"❌ Usuario no es Cliente, es {type(current_user).__name__}")
        raise HTTPException(
            status_code=403, 
            detail="Solo clientes pueden acceder a esta ruta"
        )
    
    print(f"✅ Cliente: {current_user.first_name} {current_user.last_name_paternal}")
    print(f"✅ Activity Level: {current_user.activity_level}")
    print(f"✅ Goal: {current_user.goal}")
    
    perfil_response = ClientResponse(
        id=current_user.id,
        first_name=current_user.first_name or "",
        last_name_paternal=current_user.last_name_paternal or "",
        last_name_maternal=current_user.last_name_maternal or "",
        email=current_user.email,
        flutter_uid=current_user.flutter_uid,
        birth_date=current_user.birth_date,
        weight=current_user.weight or 0.0,
        height=current_user.height or 0.0,
        gender=current_user.gender or "M",
        activity_level=current_user.activity_level or "Sedentario",
        goal=current_user.goal or "Mantener peso",
        workout_type=current_user.workout_type or "Cardio",
        session_duration=current_user.session_duration or 1.0,
        medical_conditions=current_user.medical_conditions or [],
        assigned_coach_id=current_user.assigned_coach_id,
        assigned_nutri_id=current_user.assigned_nutri_id,
        profile_picture_url=current_user.profile_picture_url,
        is_profile_complete=current_user.is_profile_complete
    )
    return perfil_response

@router.post("/perfil/foto")
async def subir_foto_perfil_cliente(
    file: UploadFile = File(...),
    current_user: Client = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Sube una foto de perfil localmente para el cliente y actualiza la URL en la base de datos"""
    if not isinstance(current_user, Client):
        raise HTTPException(status_code=403, detail="Solo clientes pueden usar este endpoint para su propio perfil")
        
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")
    
    file_bytes = await file.read()
    
    if current_user.profile_picture_url:
        local_storage.delete_file(current_user.profile_picture_url)
    
    relative_path = local_storage.save_file(file_bytes, file.filename)
    public_url = local_storage.get_public_url(relative_path)
    
    current_user.profile_picture_url = public_url
    db.commit()
    
    return {"message": "Foto de perfil actualizada exitosamente", "url": public_url}

@router.post("/forgot-password/request")
def solicitar_codigo(email: str, db: Session = Depends(get_db)):
    cliente = db.query(Client).filter(Client.email == email).first()
    if not cliente:
        raise HTTPException(status_code=404, detail="Email no encontrado")

    otp_code = f"{random.randint(100000, 999999)}"
    
    cliente.verification_code = otp_code
    cliente.code_expires_at = datetime.utcnow() + timedelta(minutes=15)
    db.commit()

    EmailService.send_otp_email(email, otp_code)

    return {"message": "Código enviado exitosamente"}

@router.post("/forgot-password/verify")
def verificar_y_cambiar(email: str, code: str, new_password: str, db: Session = Depends(get_db)):
    print(f"🔐 Iniciando verificación y sincronización para: {email}")
    
    cliente = db.query(Client).filter(
        Client.email == email,
        Client.verification_code == code,
        Client.code_expires_at > datetime.utcnow()
    ).first()

    if not cliente:
        print(f"❌ Código inválido o expirado para {email}")
        raise HTTPException(status_code=400, detail="Código inválido o expirado")

    try:
        try:
            fb_user = firebase_admin_auth.get_user_by_email(email)
            firebase_admin_auth.update_user(fb_user.uid, password=new_password)
            print(f"✅ Contraseña sincronizada en Firebase para UID: {fb_user.uid}")
        except Exception as fb_error:
            print(f"⚠️ Nota: No se pudo actualizar en Firebase (posiblemente no existe): {fb_error}")

        cliente.hashed_password = security.hash_password(new_password)
        cliente.verification_code = None
        cliente.code_expires_at = None
        
        db.commit()
        print(f"✅ Contraseña actualizada en PostgreSQL para: {email}")

        return {"success": True, "message": "Tu contraseña ha sido actualizada en todo el sistema"}

    except Exception as e:
        db.rollback()
        print(f"❌ Error crítico en el proceso: {e}")
        raise HTTPException(status_code=500, detail="Error interno al actualizar la contraseña")


@router.put("/perfil")
def actualizar_perfil_cliente(
    cliente_data: ClientUpdate,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Actualiza el perfil del cliente autenticado"""
    
    print(f"📝 PUT /clientes/perfil llamado")
    print(f"📝 Usuario: {current_user.email}")
    print(f"📝 Datos recibidos: {cliente_data.model_dump(exclude_unset=True)}")
    
    if not isinstance(current_user, Client):
        print(f"❌ Usuario no es Cliente")
        raise HTTPException(
            status_code=403, 
            detail="Solo clientes pueden actualizar su perfil"
        )
    
    cliente = db.query(Client).filter(Client.id == current_user.id).first()
    if not cliente:
        raise HTTPException(status_code=404, detail="Cliente no encontrado")
    
    if cliente_data.email and cliente_data.email != cliente.email:
        existe = db.query(Client).filter(Client.email == cliente_data.email).first()
        if existe:
            raise HTTPException(status_code=400, detail="El email ya está registrado")
    
    update_data = cliente_data.model_dump(exclude_unset=True)

    _PLAN_TRIGGER_FIELDS    = {"activity_level", "goal", "weight", "height"}
    _PLAN_INVALIDATE_FIELDS = {"activity_level", "goal", "weight", "height", "medical_conditions"}

    _old_plan_vals = {
        f: getattr(cliente, f)
        for f in _PLAN_INVALIDATE_FIELDS
        if hasattr(cliente, f)
    }

    for field, value in update_data.items():
        if hasattr(cliente, field):
            old_value = getattr(cliente, field)
            setattr(cliente, field, value)
            print(f"✅ {field}: {old_value} → {value}")

    _COMPLETION_FIELDS = {"first_name", "weight", "height", "birth_date", "gender"}
    if _COMPLETION_FIELDS & set(update_data.keys()):
        cliente.is_profile_complete = True

    try:
        db.commit()
        db.refresh(cliente)
        print(f"✅ Perfil actualizado para cliente ID {cliente.id}")

        plan_recalculado = False
        if _PLAN_TRIGGER_FIELDS & set(update_data.keys()):
            try:
                from app.models.nutricion import PlanNutricional, PlanDiario

                edad = 30
                if cliente.birth_date:
                    hoy = date.today()
                    edad = hoy.year - cliente.birth_date.year - (
                        (hoy.month, hoy.day) < (cliente.birth_date.month, cliente.birth_date.day)
                    )

                rec = CalculadorDietaAutomatica.calcular_recomendacion_dieta(
                    peso=float(cliente.weight or 70),
                    altura=float(cliente.height or 170),
                    edad=edad,
                    genero=cliente.gender or "M",
                    nivel_actividad=cliente.activity_level or "Moderado",
                    objetivo=cliente.goal or "Mantener peso",
                )

                plan_activo = (
                    db.query(PlanNutricional)
                    .filter(PlanNutricional.client_id == cliente.id)
                    .order_by(PlanNutricional.fecha_creacion.desc())
                    .first()
                )
                if plan_activo:
                    for pd in db.query(PlanDiario).filter(PlanDiario.plan_id == plan_activo.id).all():
                        pd.calorias_dia = round(rec.calorias_diarias)
                        pd.proteinas_g = rec.proteinas_g
                        pd.carbohidratos_g = rec.carbohidratos_g
                        pd.grasas_g = rec.grasas_g
                    db.commit()
                    plan_recalculado = True
                    print(f"✅ Plan recalculado: {round(rec.calorias_diarias)} kcal ({cliente.activity_level}, {cliente.goal})")
            except Exception as _e:
                print(f"⚠️ Recalculo de plan falló (no crítico): {_e}")

        try:
            from app.models.nutricion import PlanNutricional

            _campos_cambiados = [
                f for f in _PLAN_INVALIDATE_FIELDS
                if f in update_data and getattr(cliente, f) != _old_plan_vals.get(f)
            ]

            if _campos_cambiados:
                plan_activo = (
                    db.query(PlanNutricional)
                    .filter(PlanNutricional.client_id == cliente.id)
                    .order_by(PlanNutricional.fecha_creacion.desc())
                    .first()
                )
                if plan_activo and plan_activo.status == "validado":
                    plan_activo.status = "draft_ia"
                    plan_recalculado = plan_recalculado or bool(_PLAN_TRIGGER_FIELDS & set(_campos_cambiados))
                    db.commit()
                    print(
                        f"⚠️ Plan ID {plan_activo.id} → draft_ia "
                        f"(campos modificados: {_campos_cambiados})"
                    )
        except Exception as _e:
            print(f"⚠️ Reset plan falló (no crítico): {_e}")

        return {
            "message": "Perfil actualizado exitosamente",
            "cliente": cliente,
            "plan_recalculado": plan_recalculado,
            "plan_invalidado": bool(_campos_cambiados) if '_campos_cambiados' in dir() else False,
        }
    except IntegrityError as e:
        db.rollback()
        logger.error("Error de integridad actualizando cliente: %s", e, exc_info=True)
        raise HTTPException(status_code=400, detail="Error de integridad al actualizar el cliente")
    except Exception as e:
        db.rollback()
        logger.error("Error actualizando cliente: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Error interno al actualizar el cliente")


@router.put("/vincular-uid")
def vincular_uid_flutter(
    flutter_uid: str,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Asocia el UID de Firebase/Flutter con el perfil de salud del cliente.
    Este endpoint permite que la app móvil envíe su UID único para vincularse correctamente.
    
    Parámetros:
    - flutter_uid: UID único generado por Flutter/Firebase
    
    Ejemplo de uso desde Flutter:
    ```
    PUT /clientes/vincular-uid?flutter_uid=abc123def456
    Headers: Authorization: Bearer {token_jwt}
    ```
    """
    
    print(f"🔗 Vinculando UID de Flutter: {flutter_uid} al cliente ID: {current_user.id}")
    
    if not isinstance(current_user, Client):
        raise HTTPException(
            status_code=403,
            detail="Solo clientes pueden vincular UID de Flutter"
        )
    
    try:
        existing = db.query(Client).filter(
            Client.flutter_uid == flutter_uid,
            Client.id != current_user.id
        ).first()
        
        if existing:
            print(f"❌ UID ya está vinculado a otro usuario")
            raise HTTPException(
                status_code=400,
                detail="Este UID de Flutter ya está vinculado a otro usuario"
            )
        
        current_user.flutter_uid = flutter_uid
        db.commit()
        
        print(f"✅ UID vinculado exitosamente al cliente {current_user.first_name}")
        
        return {
            "message": "UID de Flutter vinculado exitosamente",
            "client_id": current_user.id,
            "flutter_uid": flutter_uid,
            "user": current_user.first_name + " " + current_user.last_name_paternal
        }
    
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error("Error vinculando UID: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Error al vincular el UID al perfil"
        )


@router.get("/por-uid/{flutter_uid}", response_model=ClientResponseConDieta)
def obtener_perfil_por_uid_con_dieta(
    flutter_uid: str,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Obtiene el perfil de salud completo CON RECOMENDACIÓN DE DIETA AUTOMÁTICA 
    usando el UID de Flutter e incluyendo condiciones médicas.
    
    🔒 REQUIERE AUTENTICACIÓN: Solo el dueño del perfil puede acceder.
    """
    
    print(f"🔍 Buscando perfil + dieta por UID de Flutter: {flutter_uid}")
    
    if isinstance(current_user, Client):
        if current_user.flutter_uid != flutter_uid:
            print(f"❌ Intento de acceso no autorizado: Usuario {current_user.email} intentó acceder a UID {flutter_uid}")
            raise HTTPException(
                status_code=403,
                detail="No tienes permiso para acceder a este perfil"
            )
    elif not (hasattr(current_user, 'role_name') and current_user.role_name in ['admin', 'nutritionist', 'coach']):
        raise HTTPException(
            status_code=403,
            detail="No autorizado para acceder a perfiles de clientes"
        )
    
    cliente = db.query(Client).filter(Client.flutter_uid == flutter_uid).first()
    
    if not cliente:
        print(f"❌ No se encontró cliente con UID: {flutter_uid}")
        raise HTTPException(
            status_code=404,
            detail="Perfil de salud no encontrado para este UID de Flutter"
        )
    
    print(f"✅ Perfil encontrado para {cliente.first_name}")
    
    edad = 30  
    if cliente.birth_date:
        today = date.today()
        edad = today.year - cliente.birth_date.year - (
            (today.month, today.day) < (cliente.birth_date.month, cliente.birth_date.day)
        )
    
    recomendacion = CalculadorDietaAutomatica.calcular_recomendacion_dieta(
        peso=cliente.weight or 70,
        altura=cliente.height or 170,
        edad=edad,
        genero=cliente.gender or 'M',
        nivel_actividad=cliente.activity_level or 'Moderado',
        objetivo=cliente.goal or 'Mantener peso'
    )
    
    dieta_schema = RecomendacionDietaCompleta(
        calorias_diarias=recomendacion.calorias_diarias,
        proteinas_g=recomendacion.proteinas_g,
        carbohidratos_g=recomendacion.carbohidratos_g,
        grasas_g=recomendacion.grasas_g,
        imc=recomendacion.imc,
        categoria_imc=recomendacion.categoria_imc,
        gasto_metabolico_basal=recomendacion.gasto_metabolico_basal,
        objetivo_recomendado=recomendacion.objetivo_recomendado,
        alimentos_recomendados=recomendacion.alimentos_recomendados,
        alimentos_a_evitar=recomendacion.alimentos_a_evitar,
        frecuencia_comidas=recomendacion.frecuencia_comidas,
        notas=recomendacion.notas
    )
    
    perfil_response = ClientResponseConDieta(
        id=cliente.id,
        first_name=cliente.first_name or "",
        last_name_paternal=cliente.last_name_paternal or "",
        last_name_maternal=cliente.last_name_maternal or "",
        email=cliente.email,
        flutter_uid=cliente.flutter_uid,
        birth_date=cliente.birth_date,
        weight=cliente.weight or 0.0,
        height=cliente.height or 0.0,
        gender=cliente.gender or "M",
        medical_conditions=cliente.medical_conditions or [],
        goal=cliente.goal,
        activity_level=cliente.activity_level,
        assigned_coach_id=cliente.assigned_coach_id,
        assigned_nutri_id=cliente.assigned_nutri_id,
        profile_picture_url=cliente.profile_picture_url,
        dieta_recomendada=dieta_schema
    )
    
    print(f"✅ Perfil completo enviado (Condiciones: {len(perfil_response.medical_conditions)})")
    
    return perfil_response


@router.get("/por-uid-simple/{flutter_uid}")
def obtener_perfil_por_uid(
    flutter_uid: str,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Obtiene el perfil de salud simple usando el UID de Flutter (SIN dieta automática).
    
    🔒 REQUIERE AUTENTICACIÓN: Solo el dueño del perfil puede acceder.
    
    Este es el endpoint anterior, mantenido para compatibilidad.
    Para obtener perfil CON dieta automática, usa: GET /clientes/por-uid/{flutter_uid}
    
    Parámetro:
    - flutter_uid: UID único de Firebase/Flutter
    
    Ejemplo de uso desde Flutter:
    ```
    GET /clientes/por-uid-simple/abc123def456
    Headers: Authorization: Bearer {token}
    ```
    """
    
    print(f"🔍 Buscando perfil simple por UID de Flutter: {flutter_uid}")
    
    if isinstance(current_user, Client):
        if current_user.flutter_uid != flutter_uid:
            print(f"❌ Intento de acceso no autorizado: Usuario {current_user.email} intentó acceder a UID {flutter_uid}")
            raise HTTPException(
                status_code=403,
                detail="No tienes permiso para acceder a este perfil"
            )
    elif not (hasattr(current_user, 'role_name') and current_user.role_name in ['admin', 'nutritionist', 'coach']):
        raise HTTPException(
            status_code=403,
            detail="No autorizado para acceder a perfiles de clientes"
        )
    
    cliente = db.query(Client).filter(Client.flutter_uid == flutter_uid).first()
    
    if not cliente:
        print(f"❌ No se encontró cliente con UID: {flutter_uid}")
        raise HTTPException(
            status_code=404,
            detail="Perfil de salud no encontrado para este UID de Flutter"
        )
    
    print(f"✅ Perfil encontrado para {cliente.first_name}")
    
    perfil_response = ClientResponse(
        id=cliente.id,
        first_name=cliente.first_name or "",
        last_name_paternal=cliente.last_name_paternal or "",
        last_name_maternal=cliente.last_name_maternal or "",
        email=cliente.email,
        flutter_uid=cliente.flutter_uid,
        birth_date=cliente.birth_date,
        weight=cliente.weight or 0.0,
        height=cliente.height or 0.0,
        gender=cliente.gender or "M",
        activity_level=cliente.activity_level or "Sedentario",
        goal=cliente.goal or "Mantener peso",
        workout_type=cliente.workout_type or "Cardio",
        session_duration=cliente.session_duration or 1.0,
        medical_conditions=cliente.medical_conditions or [],
        assigned_coach_id=cliente.assigned_coach_id,
        assigned_nutri_id=cliente.assigned_nutri_id,
        profile_picture_url=cliente.profile_picture_url,
        is_profile_complete=cliente.is_profile_complete
    )

    return perfil_response


@router.get("/checkin-status")
def check_checkin_status(
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Verifica si el usuario necesita hacer el check-in del mes"""
    from app.core.utils import get_peru_now
    from app.models.historial import HistorialPeso
    from app.models.client import Client as ClientModel

    if not isinstance(current_user, ClientModel):
        return {"needed": False, "already_done": True, "days_since": 0,
                "days_until_checkin": 30, "precision": 100, "is_new_user": False}

    now = get_peru_now()

    first_of_month = now.replace(day=1).date()

    now_naive = now.replace(tzinfo=None)
    created = current_user.created_at.replace(tzinfo=None) if current_user.created_at else None
    days_since_creation = (now_naive - created).days if created else 31
    is_new_user = days_since_creation < 30

    already_done = db.query(Client).filter(
        Client.id == current_user.id,
        Client.historial_peso.any(HistorialPeso.fecha_registro >= first_of_month)
    ).first()

    last_record = db.query(HistorialPeso).filter(
        HistorialPeso.client_id == current_user.id
    ).order_by(HistorialPeso.fecha_registro.desc()).first()

    days_since = 0
    if last_record:
        days_since = (now.date() - last_record.fecha_registro).days
    else:
        days_since = days_since_creation

    precision = 100
    if not is_new_user:
        if days_since > 15: precision = 70
        if days_since > 25: precision = 40
        if days_since > 35: precision = 15
    else:
        precision = 100

    days_until_checkin = max(0, 30 - days_since)
    
    needed = days_since >= 30

    from app.models.nutricion import PlanNutricional
    nutri_updates_pending = False
    last_update_date = None
    
    if current_user.is_strategic_guide_validated or current_user.nutri_weekly_note:
        nutri_updates_pending = True
    
    latest_plan = db.query(PlanNutricional).filter(
        PlanNutricional.client_id == current_user.id,
        PlanNutricional.status == "validado"
    ).order_by(PlanNutricional.validated_at.desc()).first()
    
    if latest_plan and latest_plan.validated_at:
        nutri_updates_pending = True
        last_update_date = latest_plan.validated_at.strftime("%d/%m/%Y")

    return {
        "needed": needed,
        "precision_score": precision,
        "days_since_update": days_since,
        "days_until_checkin": days_until_checkin,
        "nutri_updates_pending": nutri_updates_pending,
        "last_update_date": last_update_date,
        "first_of_month": first_of_month.strftime("%Y-%m-%d"),
        "message": "¡Calibración mensual pendiente!" if needed else ("¡Perfil al día!" if is_new_user else "Plan calibrado este mes")
    }

@router.post("/checkin")
def process_checkin(
    data: dict,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Procesa el check-in semanal y actualiza el historial"""
    cliente = db.query(Client).filter(Client.id == current_user.id).first()
    if not cliente:
        raise HTTPException(status_code=404, detail="Cliente no encontrado")
        
    old_weight = cliente.weight
    new_weight = data.get("weight")
    
    cliente.weight = new_weight
    if data.get("height"):
        cliente.height = data.get("height")
    if data.get("activity_level"):
        cliente.activity_level = data.get("activity_level")
        
    from app.models.historial import HistorialPeso
    from app.core.utils import get_peru_date
    nuevo_registro = HistorialPeso(
        client_id=cliente.id,
        peso_kg=new_weight,
        fecha_registro=get_peru_date()
    )
    db.add(nuevo_registro)
    
    alerta_staff = False
    if old_weight:
        diff_percent = abs(new_weight - old_weight) / old_weight * 100
        if diff_percent > 3:
            alerta_staff = True
            
    db.commit()
    return {
        "status": "success",
        "message": "Check-in completado. ¡Tu IA está calibrada!",
        "requires_staff_review": alerta_staff
    }


@router.put("/recalcular-dieta/{cliente_id}", response_model=ClientResponseConDieta)
def recalcular_dieta(
    cliente_id: int,
    objetivo: str = None,
    nivel_actividad: str = None,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Recalcula la dieta recomendada cuando el usuario cambia su objetivo o nivel de actividad.
    
    Parámetros opcionales (al menos uno debe proporcionarse):
    - objetivo: "Perder peso", "Mantener peso", "Ganar masa"
    - nivel_actividad: "Sedentario", "Ligero", "Moderado", "Intenso", "Muy intenso"
    
    Ejemplo desde Flutter:
    ```
    PUT /clientes/recalcular-dieta/3?objetivo=Perder+peso&nivel_actividad=Moderado
    ```
    """
    
    print(f"🔄 Recalculando dieta para cliente {cliente_id}")
    
    if current_user.type != 'staff' and current_user.user_id != cliente_id:
        raise HTTPException(
            status_code=403,
            detail="No tienes permiso para modificar este perfil"
        )
    
    cliente = db.query(Client).filter(Client.id == cliente_id).first()
    if not cliente:
        raise HTTPException(status_code=404, detail="Cliente no encontrado")
    
    if objetivo:
        cliente.goal = objetivo
    if nivel_actividad:
        cliente.activity_level = nivel_actividad
    
    db.commit()
    print(f"✅ Perfil actualizado: Objetivo={cliente.goal}, Actividad={cliente.activity_level}")
    
    edad = 30
    if cliente.birth_date:
        today = date.today()
        edad = today.year - cliente.birth_date.year - (
            (today.month, today.day) < (cliente.birth_date.month, cliente.birth_date.day)
        )
    
    print(f"🍽️  Recalculando dieta con nuevos parámetros...")
    
    recomendacion = CalculadorDietaAutomatica.calcular_recomendacion_dieta(
        peso=cliente.weight or 70,
        altura=cliente.height or 170,
        edad=edad,
        genero=cliente.gender or 'M',
        nivel_actividad=cliente.activity_level or 'Moderado',
        objetivo=cliente.goal or 'Mantener peso'
    )
    
    dieta_schema = RecomendacionDietaCompleta(
        calorias_diarias=recomendacion.calorias_diarias,
        proteinas_g=recomendacion.proteinas_g,
        carbohidratos_g=recomendacion.carbohidratos_g,
        grasas_g=recomendacion.grasas_g,
        imc=recomendacion.imc,
        categoria_imc=recomendacion.categoria_imc,
        gasto_metabolico_basal=recomendacion.gasto_metabolico_basal,
        objetivo_recomendado=recomendacion.objetivo_recomendado,
        alimentos_recomendados=recomendacion.alimentos_recomendados,
        alimentos_a_evitar=recomendacion.alimentos_a_evitar,
        frecuencia_comidas=recomendacion.frecuencia_comidas,
        notas=recomendacion.notas
    )
    
    perfil_response = ClientResponseConDieta(
        id=cliente.id,
        first_name=cliente.first_name or "",
        last_name_paternal=cliente.last_name_paternal or "",
        last_name_maternal=cliente.last_name_maternal or "",
        email=cliente.email,
        flutter_uid=cliente.flutter_uid,
        birth_date=cliente.birth_date,
        weight=cliente.weight or 0.0,
        height=cliente.height or 0.0,
        goal=cliente.goal,
        activity_level=cliente.activity_level,
        assigned_coach_id=cliente.assigned_coach_id,
        assigned_nutri_id=cliente.assigned_nutri_id,
        profile_picture_url=cliente.profile_picture_url,
        dieta_recomendada=dieta_schema
    )
    
    print(f"✅ Dieta recalculada: {recomendacion.calorias_diarias:.0f} kcal")
    
    return perfil_response


@router.put("/{cliente_id}/cambiar-contrasena")
def admin_cambiar_contrasena_cliente(
    cliente_id: int,
    nueva_contrasena: ChangePassword,
    db: Session = Depends(get_db),
    current_staff = Depends(get_current_staff)
):
    """
    SOLO ADMIN/STAFF: Cambia la contraseña de un cliente.
    
    El personal del gimnasio (Admin, Coach, Nutricionista) puede cambiar 
    la contraseña de cualquier cliente registrado.
    
    Parámetros:
    - cliente_id: ID del cliente cuya contraseña se cambiará
    - new_password: Nueva contraseña (mínimo 6 caracteres)
    - confirm_password: Confirmación de la contraseña
    
    Ejemplo:
    ```
    PUT /clientes/3/cambiar-contrasena
    Headers: Authorization: Bearer {token_admin}
    Body: {
        "new_password": "nuevaPassword123",
        "confirm_password": "nuevaPassword123"
    }
    ```
    """
    print(f"🔐 Admin {current_staff.email} intentando cambiar contraseña de cliente {cliente_id}")
    
    if nueva_contrasena.new_password != nueva_contrasena.confirm_password:
        raise HTTPException(
            status_code=400,
            detail="Las contraseñas no coinciden"
        )
    
    cliente = db.query(Client).filter(Client.id == cliente_id).first()
    if not cliente:
        raise HTTPException(status_code=404, detail="Cliente no encontrado")
    
    try:
        cliente.hashed_password = security.hash_password(nueva_contrasena.new_password)
        db.commit()
        
        print(f"✅ Contraseña de cliente {cliente.email} actualizada por admin {current_staff.email}")
        
        return {
            "message": "Contraseña del cliente actualizada exitosamente",
            "client_id": cliente.id,
            "client_email": cliente.email,
            "client_name": f"{cliente.first_name} {cliente.last_name_paternal}"
        }
    except Exception as e:
        db.rollback()
        logger.error("Error al cambiar contraseña: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Error al cambiar la contraseña"
        )


@router.put("/usuario/{usuario_id}/cambiar-contrasena")
def admin_cambiar_contrasena_usuario(
    usuario_id: int,
    nueva_contrasena: ChangePassword,
    db: Session = Depends(get_db),
    current_staff = Depends(get_current_staff)
):
    """
    SOLO ADMIN: Cambia la contraseña de un usuario (Coach, Nutricionista, Admin).
    
    Parámetros:
    - usuario_id: ID del usuario staff cuya contraseña se cambiará
    - new_password: Nueva contraseña (mínimo 6 caracteres)
    - confirm_password: Confirmación de la contraseña
    
    Ejemplo:
    ```
    PUT /clientes/usuario/5/cambiar-contrasena
    Headers: Authorization: Bearer {token_admin}
    Body: {
        "new_password": "nuevaPassword123",
        "confirm_password": "nuevaPassword123"
    }
    ```
    """
    print(f"🔐 Admin {current_staff.email} intentando cambiar contraseña de usuario {usuario_id}")
    
    if nueva_contrasena.new_password != nueva_contrasena.confirm_password:
        raise HTTPException(
            status_code=400,
            detail="Las contraseñas no coinciden"
        )
    
    usuario = db.query(User).filter(User.id == usuario_id).first()
    if not usuario:
        raise HTTPException(status_code=404, detail="Usuario staff no encontrado")
    
    try:
        usuario.hashed_password = security.hash_password(nueva_contrasena.new_password)
        db.commit()
        
        print(f"✅ Contraseña de usuario {usuario.email} actualizada por admin {current_staff.email}")
        
        return {
            "message": "Contraseña del usuario actualizada exitosamente",
            "user_id": usuario.id,
            "user_email": usuario.email,
            "user_name": f"{usuario.first_name} {usuario.last_name_paternal}",
            "user_role": usuario.role_name
        }
    except Exception as e:
        db.rollback()
        logger.error("Error al cambiar contraseña: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Error al cambiar la contraseña"
        )