import sys
import os

# Add the current directory to sys.path to import app
sys.path.append(os.getcwd())

from sqlalchemy.orm import Session
from app.core.database import SessionLocal, engine, Base
from app.models.role import Role
from app.models.user import User
from app.core.security import Security

def seed():
    db = SessionLocal()
    try:
        # 1. Create Roles
        roles_data = [
            {"name": "admin", "description": "Acceso total al sistema"},
            {"name": "coach", "description": "Gestión de entrenamientos"},
            {"name": "nutritionist", "description": "Gestión de dietas y nutrición"},
        ]
        
        roles = []
        for rd in roles_data:
            role = db.query(Role).filter(Role.name == rd["name"]).first()
            if not role:
                role = Role(name=rd["name"], description=rd["description"])
                db.add(role)
                db.commit()
                db.refresh(role)
                print(f"✅ Rol creado: {role.name}")
            else:
                print(f"ℹ️ Rol ya existe: {role.name}")
            roles.append(role)
            
        admin_role = next(r for r in roles if r.name == "admin")

        # 2. Create Default Admin User
        admin_email = "leomedinaflores09@gmail.com"
        admin_user = db.query(User).filter(User.email == admin_email).first()
        
        if not admin_user:
            admin_user = User(
                first_name="Leonardo",
                last_name_paternal="Medina",
                last_name_maternal="Flores",
                email=admin_email,
                hashed_password=Security.hash_password("alfa123"),
                role_id=admin_role.id,
                role_name=admin_role.name,
                is_active=True
            )
            db.add(admin_user)
            db.commit()
            print(f"✅ Usuario administrador creado: {admin_email}")
        else:
            print(f"ℹ️ Usuario administrador ya existe: {admin_email}")

        print("\n🚀 Inicialización completada con éxito!")

    except Exception as e:
        print(f"❌ Error durante la inicialización: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    seed()
