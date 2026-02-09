"""
Script Python para exportar BD local de PostgreSQL a Docker
Alternativa a pg_dump cuando no está en PATH
"""
import subprocess
import sys

print("=" * 70)
print("📦 EXPORTAR BD LOCAL A DOCKER")
print("=" * 70)

# Paso 1: Encontrar PostgreSQL
pg_paths = [
    r"C:\Program Files\PostgreSQL\16\bin\pg_dump.exe",
    r"C:\Program Files\PostgreSQL\15\bin\pg_dump.exe",
    r"C:\Program Files\PostgreSQL\14\bin\pg_dump.exe",
    r"C:\Program Files (x86)\PostgreSQL\16\bin\pg_dump.exe",
]

print("\n🔍 Buscando pg_dump...")
pg_dump_path = None
for path in pg_paths:
    try:
        result = subprocess.run([path, "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            pg_dump_path = path
            print(f"✅ Encontrado: {path}")
            break
    except:
        continue

if not pg_dump_path:
    print("❌ No se encontró pg_dump")
    print("\n💡 SOLUCIÓN ALTERNATIVA:")
    print("1. Abre pgAdmin 4")
    print("2. Click derecho en BD_Calofit → Backup...")
    print("3. Guarda como: backup_bd_calofit.sql")
    print("4. Luego ejecuta:")
    print("   docker exec -i calofit_db psql -U postgres -d BD_Calofit < backup_bd_calofit.sql")
    sys.exit(1)

# Paso 2: Exportar BD
print("\n📤 Exportando BD local...")
backup_file = r"D:\PROYECTO_TESIS\PROYECTO\calofit_backend\backup_bd_calofit.sql"

cmd = [
    pg_dump_path,
    "-U", "postgres",
    "-h", "localhost",
    "-d", "BD_Calofit",
    "-f", backup_file
]

try:
    result = subprocess.run(cmd, capture_output=True, text=True, input="leomeflo09\n")
    if result.returncode == 0:
        print(f"✅ Backup creado: {backup_file}")
        
        # Paso 3: Importar a Docker
        print("\n📥 Importando a Docker...")
        with open(backup_file, 'r', encoding='utf-8') as f:
            backup_data = f.read()
        
        import_cmd = [
            "docker", "exec", "-i", "calofit_db",
            "psql", "-U", "postgres", "-d", "BD_Calofit"
        ]
        
        import_result = subprocess.run(
            import_cmd,
            input=backup_data,
            capture_output=True,
            text=True
        )
        
        if import_result.returncode == 0:
            print("✅ BD importada exitosamente a Docker")
            
            # Paso 4: Verificar
            print("\n🔍 Verificando usuarios...")
            verify_cmd = [
                "docker", "exec", "calofit_db",
                "psql", "-U", "postgres", "-d", "BD_Calofit",
                "-c", "SELECT email, first_name FROM users LIMIT 5;"
            ]
            verify_result = subprocess.run(verify_cmd, capture_output=True, text=True)
            print(verify_result.stdout)
            
            print("\n🎉 ¡LISTO! Ahora puedes hacer login en Postman:")
            print("POST http://localhost:8000/auth/login")
            print('{')
            print('  "email": "leomedinaflores09@gmail.com",')
            print('  "password": "alfa123"')
            print('}')
        else:
            print(f"❌ Error al importar: {import_result.stderr}")
    else:
        print(f"❌ Error al exportar: {result.stderr}")
except Exception as e:
    print(f"❌ Error: {e}")
    print("\n💡 Prueba manualmente:")
    print(f"1. {pg_dump_path} -U postgres -h localhost -d BD_Calofit > backup_bd_calofit.sql")
    print("2. docker exec -i calofit_db psql -U postgres -d BD_Calofit < backup_bd_calofit.sql")
