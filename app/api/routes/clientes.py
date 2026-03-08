from fastapi import APIRouter, Depends, HTTPException, File, UploadFile
from app.core.local_storage import local_storage
from sqlalchemy.orm import Session
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from app.core.database import get_db
from app.models.client import Client
from app.models.user import User
from app.schemas.client import ClientCreate, ClientUpdate, ClientResponse, ChangePassword
from app.schemas.dieta import ClientResponseConDieta, RecomendacionDietaCompleta
from app.core.security import security
from app.api.routes.auth import get_current_user, get_current_staff
from app.services.calculador_dieta import CalculadorDietaAutomatica
from datetime import date
from app.services.email_service import EmailService
import random
from datetime import datetime, timedelta
from app.core.firebase import auth as firebase_admin_auth


router = APIRouter()


@router.post("/registrar")
def registrar_cliente(cliente_data: ClientCreate, db: Session = Depends(get_db)):
    """Registra un nuevo cliente en el sistema"""
    
    # Verificar si el email ya existe
    existe = db.query(Client).filter(Client.email == cliente_data.email).first()
    if existe:
        raise HTTPException(status_code=400, detail="El email ya está registrado")
    
    # Verificar que assigned_coach_id existe si se proporciona
    if cliente_data.assigned_coach_id is not None:
        coach = db.query(User).filter(User.id == cliente_data.assigned_coach_id).first()
        if not coach:
            raise HTTPException(status_code=400, detail="El coach asignado no existe")
    
    # Verificar que assigned_nutri_id existe si se proporciona
    if cliente_data.assigned_nutri_id is not None:
        nutri = db.query(User).filter(User.id == cliente_data.assigned_nutri_id).first()
        if not nutri:
            raise HTTPException(status_code=400, detail="El nutricionista asignado no existe")
    
    # Crear el nuevo cliente con valores proporcionados o por defecto
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
        
        # 🆕 GENERAR PLAN NUTRICIONAL AUTOMÁTICO
        print(f"🤖 Generando plan automático para {nuevo_cliente.email}...")
        from app.services.ia_service import ia_engine
        from app.models.nutricion import PlanNutricional, PlanDiario
        
        # Calcular edad
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
            # Crear plan maestro
            plan_maestro = PlanNutricional(
                client_id=nuevo_cliente.id,
                genero=1 if nuevo_cliente.gender == "M" else 2,
                edad=edad,
                peso=nuevo_cliente.weight,
                talla=nuevo_cliente.height,
                nivel_actividad=1.55,  # Moderado por defecto
                objetivo=nuevo_cliente.goal,
                es_contingencia_ia=False,  # Es plan inicial, no contingencia
                calorias_ia_base=plan_data["calorias_diarias"],
                status="draft_ia",  # Pendiente validación
                validated_by_id=None,
                validated_at=None
            )
            db.add(plan_maestro)
            db.flush()  # Obtener el ID sin hacer commit completo
            
            # Crear planes diarios
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
            
            # Retornar info completa al frontend
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
        raise HTTPException(status_code=400, detail=f"Error de integridad en la base de datos: {str(e)}")
    except Exception as e:
        db.rollback()
        print(f"❌ Error en registro: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")


@router.get("/perfil")
def obtener_perfil_cliente(
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Obtiene el perfil del cliente autenticado"""
    
    print(f"🔍 GET /clientes/perfil llamado")
    print(f"🔍 Tipo de usuario: {type(current_user).__name__}")
    print(f"🔍 ID: {current_user.id}")
    print(f"🔍 Email: {current_user.email}")
    
    # Verificar que sea un cliente
    if not isinstance(current_user, Client):
        print(f"❌ Usuario no es Cliente, es {type(current_user).__name__}")
        raise HTTPException(
            status_code=403, 
            detail="Solo clientes pueden acceder a esta ruta"
        )
    
    print(f"✅ Cliente: {current_user.first_name} {current_user.last_name_paternal}")
    print(f"✅ Activity Level: {current_user.activity_level}")
    print(f"✅ Goal: {current_user.goal}")
    
    # Crear respuesta manualmente manejando valores None
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
        medical_conditions=current_user.medical_conditions or [],
        assigned_coach_id=current_user.assigned_coach_id,
        assigned_nutri_id=current_user.assigned_nutri_id,
        profile_picture_url=current_user.profile_picture_url # ✅ Añadido
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
    
    # 0. Borrar foto anterior si existe
    if current_user.profile_picture_url:
        local_storage.delete_file(current_user.profile_picture_url)
    
    # Guardar Localmente
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

    # Generar código de 6 dígitos
    otp_code = f"{random.randint(100000, 999999)}"
    
    # Guardar en BD con expiración (15 mins)
    cliente.verification_code = otp_code
    cliente.code_expires_at = datetime.utcnow() + timedelta(minutes=15)
    db.commit()

    # Enviar por Resend
    EmailService.send_otp_email(email, otp_code)

    return {"message": "Código enviado exitosamente"}

@router.post("/forgot-password/verify")
def verificar_y_cambiar(email: str, code: str, new_password: str, db: Session = Depends(get_db)):
    print(f"🔐 Iniciando verificación y sincronización para: {email}")
    
    # 1. Buscar al cliente y validar el código
    cliente = db.query(Client).filter(
        Client.email == email,
        Client.verification_code == code,
        Client.code_expires_at > datetime.utcnow()
    ).first()

    if not cliente:
        print(f"❌ Código inválido o expirado para {email}")
        raise HTTPException(status_code=400, detail="Código inválido o expirado")

    try:
        # 2. 🔥 SINCRONIZACIÓN CON FIREBASE
        # Buscamos al usuario en Firebase por su email para obtener su UID
        try:
            fb_user = firebase_admin_auth.get_user_by_email(email)
            # Actualizamos la contraseña en Firebase
            firebase_admin_auth.update_user(fb_user.uid, password=new_password)
            print(f"✅ Contraseña sincronizada en Firebase para UID: {fb_user.uid}")
        except Exception as fb_error:
            # Si el usuario no existe en Firebase, solo imprimimos el error y seguimos
            print(f"⚠️ Nota: No se pudo actualizar en Firebase (posiblemente no existe): {fb_error}")

        # 3. Actualizar en PostgreSQL (BD Local)
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
    
    # Verificar que sea un cliente
    if not isinstance(current_user, Client):
        print(f"❌ Usuario no es Cliente")
        raise HTTPException(
            status_code=403, 
            detail="Solo clientes pueden actualizar su perfil"
        )
    
    # Obtener cliente de la BD
    cliente = db.query(Client).filter(Client.id == current_user.id).first()
    if not cliente:
        raise HTTPException(status_code=404, detail="Cliente no encontrado")
    
    # Verificar email único si se está cambiando
    if cliente_data.email and cliente_data.email != cliente.email:
        existe = db.query(Client).filter(Client.email == cliente_data.email).first()
        if existe:
            raise HTTPException(status_code=400, detail="El email ya está registrado")
    
    # Actualizar solo campos proporcionados
    update_data = cliente_data.model_dump(exclude_unset=True)
    
    for field, value in update_data.items():
        if hasattr(cliente, field):
            old_value = getattr(cliente, field)
            setattr(cliente, field, value)
            print(f"✅ {field}: {old_value} → {value}")
    
    try:
        db.commit()
        db.refresh(cliente)
        print(f"✅ Perfil actualizado para cliente ID {cliente.id}")
        
        return {
            "message": "Perfil actualizado exitosamente",
            "cliente": cliente
        }
    except IntegrityError as e:
        db.rollback()
        print(f"❌ Error de integridad: {e}")
        raise HTTPException(status_code=400, detail=f"Error de integridad: {str(e)}")
    except Exception as e:
        db.rollback()
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


# ✅ NUEVO ENDPOINT: Vincular UID de Flutter con perfil de salud
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
    
    # Verificar que sea un cliente
    if not isinstance(current_user, Client):
        raise HTTPException(
            status_code=403,
            detail="Solo clientes pueden vincular UID de Flutter"
        )
    
    try:
        # Verificar si el UID ya está vinculado a otro usuario
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
        
        # Actualizar el UID de Flutter del cliente actual
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
        print(f"❌ Error vinculando UID: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error al vincular UID: {str(e)}"
        )


# ✅ NUEVO ENDPOINT: Obtener perfil por UID de Flutter CON DIETA AUTOMÁTICA
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
    
    # 🔒 VALIDACIÓN DE SEGURIDAD: Verificar que el usuario sea el dueño del perfil
    if isinstance(current_user, Client):
        if current_user.flutter_uid != flutter_uid:
            print(f"❌ Intento de acceso no autorizado: Usuario {current_user.email} intentó acceder a UID {flutter_uid}")
            raise HTTPException(
                status_code=403,
                detail="No tienes permiso para acceder a este perfil"
            )
    elif not (hasattr(current_user, 'role_name') and current_user.role_name in ['admin', 'nutritionist', 'coach']):
        # Si no es cliente ni staff, denegar acceso
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
    
    # ✅ Calcular edad
    edad = 30  
    if cliente.birth_date:
        today = date.today()
        edad = today.year - cliente.birth_date.year - (
            (today.month, today.day) < (cliente.birth_date.month, cliente.birth_date.day)
        )
    
    # Calcular recomendación de dieta
    recomendacion = CalculadorDietaAutomatica.calcular_recomendacion_dieta(
        peso=cliente.weight or 70,
        altura=cliente.height or 170,
        edad=edad,
        genero=cliente.gender or 'M',
        nivel_actividad=cliente.activity_level or 'Moderado',
        objetivo=cliente.goal or 'Mantener peso'
    )
    
    # Convertir recomendación a schema
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
    
    # ✅ CREAR RESPUESTA INCLUYENDO medical_conditions
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
        # 🔥 AQUÍ ESTABA EL ERROR: Agregamos las condiciones médicas para que Flutter las vea
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


# ✅ MANTENER ENDPOINT ANTERIOR (sin dieta) para compatibilidad
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
    
    # 🔒 VALIDACIÓN DE SEGURIDAD: Verificar que el usuario sea el dueño del perfil
    if isinstance(current_user, Client):
        if current_user.flutter_uid != flutter_uid:
            print(f"❌ Intento de acceso no autorizado: Usuario {current_user.email} intentó acceder a UID {flutter_uid}")
            raise HTTPException(
                status_code=403,
                detail="No tienes permiso para acceder a este perfil"
            )
    elif not (hasattr(current_user, 'role_name') and current_user.role_name in ['admin', 'nutritionist', 'coach']):
        # Si no es cliente ni staff, denegar acceso
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
        assigned_coach_id=cliente.assigned_coach_id,
        assigned_nutri_id=cliente.assigned_nutri_id,
        profile_picture_url=cliente.profile_picture_url
    )
    
    return perfil_response


# ✅ NUEVO: Check-in Semanal (Sábados de Calibración)
@router.get("/checkin-status")
def check_checkin_status(
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Verifica si el usuario necesita hacer el check-in del sábado (Perseverante)"""
    from app.core.utils import get_peru_now
    from app.models.historial import HistorialPeso
    now = get_peru_now()
    
    # 🕒 Encontrar el último sábado transcurrido (o hoy si es sábado)
    # weekday(): 0=Mon, 5=Sat, 6=Sun
    days_since_sat = (now.weekday() - 5) % 7
    last_saturday = (now - timedelta(days=days_since_sat)).date()
    
    # 🔍 Verificar si ya registró peso desde el último sábado (inclusive)
    already_done = db.query(Client).filter(
        Client.id == current_user.id,
        Client.historial_peso.any(HistorialPeso.fecha_registro >= last_saturday)
    ).first()
    
    # Calcular precisión basada en la última actualización real (para el meter)
    last_record = db.query(HistorialPeso).filter(
        HistorialPeso.client_id == current_user.id
    ).order_by(HistorialPeso.fecha_registro.desc()).first()
    
    days_since = 30 # Default si no hay historial
    if last_record:
        days_since = (now.date() - last_record.fecha_registro).days
        
    precision = 100
    if days_since > 7: precision = 45
    if days_since > 14: precision = 15

    # El check-in es "necesario" si no se ha hecho desde el último sábado
    # y hoy es sábado, domingo o lunes (ventana de calibración)
    # O simplemente siempre que falte el del último sábado para máxima persistencia.
    needed = not already_done

    return {
        "needed": needed,
        "precision_score": precision,
        "days_since_update": days_since,
        "last_saturday": last_saturday.strftime("%Y-%m-%d"),
        "message": "¡Calibración pendiente!" if needed else "Plan calibrado"
    }

@router.post("/checkin")
def process_checkin(
    data: dict, # {"weight": 80.5, "height": 175, "activity_level": "Moderado"}
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Procesa el check-in semanal y actualiza el historial"""
    cliente = db.query(Client).filter(Client.id == current_user.id).first()
    if not cliente:
        raise HTTPException(status_code=404, detail="Cliente no encontrado")
        
    old_weight = cliente.weight
    new_weight = data.get("weight")
    
    # Actualizar perfil
    cliente.weight = new_weight
    if data.get("height"):
        cliente.height = data.get("height")
    if data.get("activity_level"):
        cliente.activity_level = data.get("activity_level")
        
    # Registrar en historial
    from app.models.historial import HistorialPeso
    from app.core.utils import get_peru_date
    nuevo_registro = HistorialPeso(
        client_id=cliente.id,
        peso_kg=new_weight,
        fecha_registro=get_peru_date()
    )
    db.add(nuevo_registro)
    
    # Lógica de alerta si el cambio es brusco (> 3% en una semana)
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


# ✅ NUEVO ENDPOINT: Recalcular dieta cuando cambia objetivo o actividad
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
    
    # Validar que el usuario sea propietario del perfil o sea personal de staff
    if current_user.type != 'staff' and current_user.user_id != cliente_id:
        raise HTTPException(
            status_code=403,
            detail="No tienes permiso para modificar este perfil"
        )
    
    # Obtener cliente
    cliente = db.query(Client).filter(Client.id == cliente_id).first()
    if not cliente:
        raise HTTPException(status_code=404, detail="Cliente no encontrado")
    
    # Actualizar objetivo y/o nivel de actividad si se proporcionan
    if objetivo:
        cliente.goal = objetivo
    if nivel_actividad:
        cliente.activity_level = nivel_actividad
    
    db.commit()
    print(f"✅ Perfil actualizado: Objetivo={cliente.goal}, Actividad={cliente.activity_level}")
    
    # Calcular edad
    edad = 30
    if cliente.birth_date:
        today = date.today()
        edad = today.year - cliente.birth_date.year - (
            (today.month, today.day) < (cliente.birth_date.month, cliente.birth_date.day)
        )
    
    # Recalcular dieta con los nuevos parámetros
    print(f"🍽️  Recalculando dieta con nuevos parámetros...")
    
    recomendacion = CalculadorDietaAutomatica.calcular_recomendacion_dieta(
        peso=cliente.weight or 70,
        altura=cliente.height or 170,
        edad=edad,
        genero=cliente.gender or 'M',  # ✅ USA EL GÉNERO REAL DEL CLIENTE
        nivel_actividad=cliente.activity_level or 'Moderado',
        objetivo=cliente.goal or 'Mantener peso'
    )
    
    # Convertir a schema
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
    
    # Retornar perfil con nueva dieta
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


# ✅ ENDPOINT: Admin cambia contraseña de un cliente
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
    
    # Verificar que las contraseñas coinciden
    if nueva_contrasena.new_password != nueva_contrasena.confirm_password:
        raise HTTPException(
            status_code=400,
            detail="Las contraseñas no coinciden"
        )
    
    # Obtener cliente
    cliente = db.query(Client).filter(Client.id == cliente_id).first()
    if not cliente:
        raise HTTPException(status_code=404, detail="Cliente no encontrado")
    
    try:
        # Actualizar contraseña
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
        print(f"❌ Error al cambiar contraseña: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error al cambiar contraseña: {str(e)}"
        )


# ✅ ENDPOINT: Admin cambia contraseña de un usuario (staff)
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
    
    # Verificar que las contraseñas coinciden
    if nueva_contrasena.new_password != nueva_contrasena.confirm_password:
        raise HTTPException(
            status_code=400,
            detail="Las contraseñas no coinciden"
        )
    
    # Obtener usuario (staff)
    usuario = db.query(User).filter(User.id == usuario_id).first()
    if not usuario:
        raise HTTPException(status_code=404, detail="Usuario staff no encontrado")
    
    try:
        # Actualizar contraseña
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
        print(f"❌ Error al cambiar contraseña: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error al cambiar contraseña: {str(e)}"
        )