import os
import sys
from sqlalchemy import create_engine, text
from passlib.context import CryptContext

# Configurar el contexto de hash compatible con el backend (Argon2)
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def main():
    # Obtener la URL de base de datos del entorno
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("❌ Error: La variable de entorno DATABASE_URL no está configurada.")
        sys.exit(1)

    print(f"🔌 Conectando a la base de datos...")
    
    # Crear engine de SQLAlchemy
    try:
        engine = create_engine(db_url, pool_pre_ping=True)
    except Exception as e:
        print(f"❌ Error al crear el motor de base de datos: {e}")
        sys.exit(1)

    admin_email = "leomedinaflores09@gmail.com"
    # Contraseña por defecto para restablecer/crear
    admin_password = "AdminWorldLight2026!"
    hashed_pwd = hash_password(admin_password)

    with engine.connect() as conn:
        trans = conn.begin()
        try:
            # 1. Asegurar que los roles básicos existan
            print("👤 Verificando roles básicos...")
            roles = [
                (1, "admin", "Acceso total al sistema"),
                (2, "coach", "Gestión de entrenamientos"),
                (3, "nutritionist", "Gestión de dietas y nutrición"),
                (4, "client", "Cliente del gimnasio")
            ]
            for r_id, r_name, r_desc in roles:
                # Verificar si el rol ya existe
                res = conn.execute(
                    text("SELECT id FROM roles WHERE id = :rid"),
                    {"rid": r_id}
                ).fetchone()
                if not res:
                    print(f"   ➕ Creando rol: {r_name} (ID: {r_id})")
                    conn.execute(
                        text("INSERT INTO roles (id, name, description) VALUES (:rid, :rname, :rdesc)"),
                        {"rid": r_id, "rname": r_name, "rdesc": r_desc}
                    )
            
            # 2. Crear o actualizar el usuario administrador
            print(f"🔍 Buscando usuario administrador '{admin_email}'...")
            user = conn.execute(
                text("SELECT id FROM users WHERE email = :email"),
                {"email": admin_email}
            ).fetchone()

            if user:
                print("   🔄 El usuario existe. Restableciendo contraseña y asegurando rol de administrador...")
                conn.execute(
                    text(
                        "UPDATE users SET "
                        "first_name = :fname, "
                        "last_name_paternal = :paternal, "
                        "last_name_maternal = :maternal, "
                        "hashed_password = :hash, "
                        "role_id = 1, "
                        "role_name = 'admin', "
                        "is_active = true "
                        "WHERE email = :email"
                    ),
                    {
                        "fname": "Leonardo",
                        "paternal": "Medina",
                        "maternal": "",
                        "hash": hashed_pwd,
                        "email": admin_email
                    }
                )
                print("   ✅ Contraseña del administrador restablecida con éxito.")
            else:
                print("   ➕ El usuario no existe. Creando nuevo usuario administrador...")
                conn.execute(
                    text(
                        "INSERT INTO users (first_name, last_name_paternal, last_name_maternal, email, hashed_password, role_id, role_name, is_active) "
                        "VALUES (:fname, :paternal, :maternal, :email, :hash, 1, 'admin', true)"
                    ),
                    {
                        "fname": "Leonardo",
                        "paternal": "Medina",
                        "maternal": "",
                        "email": admin_email,
                        "hash": hashed_pwd
                    }
                )
                print("   ✅ Usuario administrador creado con éxito.")

            trans.commit()
            print(f"\n🎉 OPERACIÓN COMPLETADA CON ÉXITO")
            print(f"📧 Email: {admin_email}")
            print(f"🔑 Contraseña establecida: {admin_password}")
            print(f"👉 Rol: admin (ID: 1)")
            print(f"⚠️ ¡Recuerda cambiar la contraseña después de loguearte por seguridad!")
            
        except Exception as e:
            trans.rollback()
            print(f"❌ Error durante la operación en la base de datos: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()
