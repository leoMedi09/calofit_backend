"""
Script de Prueba End-to-End para CaloFit
Prueba automáticamente todo el flujo del sistema:
1. Creación de usuarios (Admin, Nutri, Trainer, Cliente)
2. Asignación de personal
3. Generación y validación de planes
4. Registro NLP y detección de lesiones

Requisitos: pip install requests colorama
"""

import requests
import json
import time
from colorama import init, Fore, Style

# Inicializar colorama para colores en Windows
init()

# Configuración
BASE_URL = "http://localhost:8000"
VERIFY_SSL = False  # Para desarrollo local

class CaloFitTester:
    def __init__(self):
        self.tokens = {}
        self.user_ids = {}
        self.client_id = None
        self.plan_id = None
        
    def print_step(self, step_number, description):
        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"PASO {step_number}: {description}")
        print(f"{'='*60}{Style.RESET_ALL}\n")
    
    def print_success(self, message):
        print(f"{Fore.GREEN}✓ {message}{Style.RESET_ALL}")
    
    def print_error(self, message):
        print(f"{Fore.RED}✗ {message}{Style.RESET_ALL}")
    
    def print_info(self, message):
        print(f"{Fore.YELLOW}ℹ {message}{Style.RESET_ALL}")
    
    def make_request(self, method, endpoint, data=None, token=None, params=None):
        """Hacer request HTTP con manejo de errores"""
        url = f"{BASE_URL}{endpoint}"
        headers = {"Content-Type": "application/json"}
        
        if token:
            headers["Authorization"] = f"Bearer {token}"
        
        try:
            if method == "GET":
                response = requests.get(url, headers=headers, params=params, verify=VERIFY_SSL, timeout=10)
            elif method == "POST":
                response = requests.post(url, headers=headers, json=data, verify=VERIFY_SSL, timeout=10)
            elif method == "PUT":
                response = requests.put(url, headers=headers, json=data, params=params, verify=VERIFY_SSL, timeout=10)
            
            return response
        except requests.exceptions.Timeout:
            self.print_error(f"Timeout en {endpoint}")
            return None
        except requests.exceptions.ConnectionError:
            self.print_error(f"No se pudo conectar a {url}")
            self.print_error("¿Está el backend corriendo en http://localhost:8000?")
            return None
        except Exception as e:
            self.print_error(f"Error de conexión: {e}")
            return None
    
    def step1_create_admin(self):
        """Crear usuario administrador"""
        self.print_step(1, "Creando Usuario Administrador")
        
        data = {
            "first_name": "Carlos",
            "last_name_paternal": "Admin",
            "last_name_maternal": "Sistema",
            "email": "admin@worldlight.com",
            "password": "Admin123!",
            "role": "admin",
            "role_id": 1
        }
        
        response = self.make_request("POST", "/usuarios/registrar", data)
        
        if response and response.status_code == 201:
            user = response.json()
            self.user_ids['admin'] = user['id']
            self.print_success(f"Admin creado: {user['first_name']} (ID: {user['id']})")
            return True
        elif response and response.status_code == 400:
            self.print_info("Admin ya existe, continuando...")
            return True
        else:
            self.print_error(f"Error al crear admin: {response.text if response else 'Sin respuesta'}")
            return False
    
    def step2_login_admin(self):
        """Login como administrador"""
        self.print_step(2, "Login como Administrador")
        
        data = {
            "email": "admin@worldlight.com",
            "password": "Admin123!"
        }
        
        response = self.make_request("POST", "/auth/login", data)
        
        if response and response.status_code == 200:
            result = response.json()
            self.tokens['admin'] = result['access_token']
            self.print_success("Login exitoso")
            self.print_info(f"Token: {self.tokens['admin'][:20]}...")
            return True
        else:
            self.print_error(f"Error en login: {response.text if response else 'Sin respuesta'}")
            return False
    
    def step3_create_nutritionist(self):
        """Crear nutricionista"""
        self.print_step(3, "Creando Nutricionista")
        
        data = {
            "first_name": "María",
            "last_name_paternal": "González",
            "last_name_maternal": "López",
            "email": "maria.gonzalez@worldlight.com",
            "password": "Nutri123!",
            "role": "nutritionist",
            "role_id": 2
        }
        
        response = self.make_request("POST", "/admin/usuarios", data, self.tokens['admin'])
        
        if response and response.status_code == 201:
            user = response.json()
            self.user_ids['nutritionist'] = user['id']
            self.print_success(f"Nutricionista creada: {user['first_name']} (ID: {user['id']})")
            return True
        elif response and response.status_code == 400:
            self.print_info("Nutricionista ya existe")
            # Intentar obtener el ID de la base de datos
            self.user_ids['nutritionist'] = 2  # Asumimos ID 2
            return True
        else:
            self.print_error(f"Error: {response.text if response else 'Sin respuesta'}")
            return False
    
    def step4_create_trainer(self):
        """Crear entrenador"""
        self.print_step(4, "Creando Entrenador")
        
        data = {
            "first_name": "Pedro",
            "last_name_paternal": "Martínez",
            "last_name_maternal": "Ruiz",
            "email": "pedro.martinez@worldlight.com",
            "password": "Trainer123!",
            "role": "coach",
            "role_id": 3
        }
        
        response = self.make_request("POST", "/admin/usuarios", data, self.tokens['admin'])
        
        if response and response.status_code == 201:
            user = response.json()
            self.user_ids['trainer'] = user['id']
            self.print_success(f"Trainer creado: {user['first_name']} (ID: {user['id']})")
            return True
        elif response and response.status_code == 400:
            self.print_info("Trainer ya existe")
            self.user_ids['trainer'] = 3  # Asumimos ID 3
            return True
        else:
            self.print_error(f"Error: {response.text if response else 'Sin respuesta'}")
            return False
    
    def step5_create_client(self):
        """Crear cliente"""
        self.print_step(5, "Registrando Cliente")
        
        data = {
            "first_name": "Juan",
            "last_name_paternal": "Pérez",
            "last_name_maternal": "García",
            "email": "juan.perez@email.com",
            "flutter_uid": "TEST_FIREBASE_UID_001",
            "password": "Cliente123!",
            "birth_date": "1995-05-15",
            "weight": 75.5,
            "height": 175,
            "gender": "M",
            "activity_level": "Moderado",
            "goal": "Perder peso",
            "medical_conditions": []
        }
        
        response = self.make_request("POST", "/clientes/registrar", data)
        
        if response and response.status_code == 201:
            client = response.json()
            self.client_id = client['id']
            self.print_success(f"Cliente creado: {client['first_name']} (ID: {client['id']})")
            return True
        elif response and response.status_code == 400:
            self.print_info("Cliente ya existe")
            self.client_id = 1  # Asumimos ID 1
            return True
        else:
            self.print_error(f"Error: {response.text if response else 'Sin respuesta'}")
            return False
    
    def step6_assign_staff(self):
        """Asignar nutricionista y trainer al cliente"""
        self.print_step(6, "Asignando Personal al Cliente")
        
        params = {
            "nutri_id": self.user_ids.get('nutritionist', 2),
            "trainer_id": self.user_ids.get('trainer', 3)
        }
        
        response = self.make_request(
            "PUT", 
            f"/admin/clientes/{self.client_id}/asignar", 
            token=self.tokens['admin'],
            params=params
        )
        
        if response and response.status_code == 200:
            self.print_success("Personal asignado correctamente")
            self.print_info(f"Nutricionista ID: {params['nutri_id']}, Trainer ID: {params['trainer_id']}")
            return True
        else:
            self.print_error(f"Error: {response.text if response else 'Sin respuesta'}")
            return False
    
    def step7_login_nutritionist(self):
        """Login como nutricionista"""
        self.print_step(7, "Login como Nutricionista")
        
        data = {
            "email": "maria.gonzalez@worldlight.com",
            "password": "Nutri123!"
        }
        
        response = self.make_request("POST", "/auth/login", data)
        
        if response and response.status_code == 200:
            result = response.json()
            self.tokens['nutritionist'] = result['access_token']
            self.print_success("Login exitoso")
            return True
        else:
            self.print_error(f"Error: {response.text if response else 'Sin respuesta'}")
            return False
    
    def step8_generate_plan(self):
        """Generar plan nutricional"""
        self.print_step(8, "Generando Plan Nutricional (IA)")
        
        data = {
            "client_id": self.client_id,
            "edad": 29,
            "peso": 75.5,
            "talla": 175,
            "nivel_actividad": 1.55,
            "objetivo": "perder"
        }
        
        response = self.make_request("POST", "/nutricion/generar-plan", data, self.tokens['nutritionist'])
        
        if response and response.status_code == 200:
            plan = response.json()
            self.plan_id = plan.get('plan_maestro', {}).get('id')
            self.print_success(f"Plan generado (ID: {self.plan_id})")
            self.print_info(f"Estado inicial: {plan.get('plan_maestro', {}).get('status', 'draft_ia')}")
            return True
        else:
            self.print_error(f"Error: {response.text if response else 'Sin respuesta'}")
            return False
    
    def step9_validate_plan(self):
        """Validar el plan como nutricionista"""
        self.print_step(9, "Validando Plan (Nutricionista)")
        
        if not self.plan_id:
            self.print_error("No hay plan_id disponible")
            return False
        
        response = self.make_request(
            "PUT", 
            f"/nutricion/planes/{self.plan_id}/validar", 
            token=self.tokens['nutritionist']
        )
        
        if response and response.status_code == 200:
            result = response.json()
            self.print_success("Plan validado exitosamente")
            self.print_info(f"Validado por: {result.get('validado_por')}")
            self.print_info(f"Fecha: {result.get('fecha')}")
            return True
        else:
            self.print_error(f"Error: {response.text if response else 'Sin respuesta'}")
            return False
    
    def step10_login_client(self):
        """Login como cliente"""
        self.print_step(10, "Login como Cliente")
        
        data = {
            "email": "juan.perez@email.com",
            "password": "Cliente123!"
        }
        
        response = self.make_request("POST", "/auth/login", data)
        
        if response and response.status_code == 200:
            result = response.json()
            self.tokens['client'] = result['access_token']
            self.print_success("Login exitoso")
            return True
        else:
            self.print_error(f"Error: {response.text if response else 'Sin respuesta'}")
            return False
    
    def step11_test_nlp_logging(self):
        """Probar registro por NLP"""
        self.print_step(11, "Probando Registro NLP (Comida)")
        
        data = {
            "mensaje": "Hoy almorcé un plato grande de arroz con pollo y ensalada mixta"
        }
        
        response = self.make_request("POST", "/asistente/log-inteligente", data, self.tokens['client'])
        
        if response and response.status_code == 200:
            result = response.json()
            self.print_success("Registro NLP exitoso")
            self.print_info(f"Tipo detectado: {result.get('tipo_detectado')}")
            self.print_info(f"Alimentos: {result.get('alimentos')}")
            self.print_info(f"Calorías: {result.get('datos', {}).get('calorias')} kcal")
            self.print_info(f"Proteínas: {result.get('datos', {}).get('proteinas')} g")
            return True
        else:
            self.print_error(f"Error: {response.text if response else 'Sin respuesta'}")
            return False
    
    def step12_test_injury_detection(self):
        """Probar detección de lesiones"""
        self.print_step(12, "Probando Detección de Lesiones (IA)")
        
        data = {
            "mensaje": "Hoy me duele mucho la rodilla derecha después de correr"
        }
        
        response = self.make_request("POST", "/asistente/consultar", data, self.tokens['client'])
        
        if response and response.status_code == 200:
            result = response.json()
            self.print_success("Consulta procesada")
            self.print_info(f"Alerta detectada: {result.get('alerta_salud')}")
            self.print_info(f"Respuesta IA: {result.get('respuesta_ia')[:100]}...")
            return True
        else:
            self.print_error(f"Error: {response.text if response else 'Sin respuesta'}")
            return False
    
    def run_all_tests(self):
        """Ejecutar todas las pruebas"""
        print(f"\n{Fore.MAGENTA}{'='*60}")
        print("INICIANDO PRUEBAS END-TO-END DE CALOFIT")
        print(f"{'='*60}{Style.RESET_ALL}\n")
        
        time.sleep(1)
        
        tests = [
            self.step1_create_admin,
            self.step2_login_admin,
            self.step3_create_nutritionist,
            self.step4_create_trainer,
            self.step5_create_client,
            self.step6_assign_staff,
            self.step7_login_nutritionist,
            self.step8_generate_plan,
            self.step9_validate_plan,
            self.step10_login_client,
            self.step11_test_nlp_logging,
            self.step12_test_injury_detection
        ]
        
        for i, test in enumerate(tests, 1):
            if not test():
                self.print_error(f"\nPrueba falló en el paso {i}. Deteniendo...")
                return False
            time.sleep(0.5)  # Pausa breve entre pasos
        
        # Resumen final
        print(f"\n{Fore.GREEN}{'='*60}")
        print("✓ TODAS LAS PRUEBAS COMPLETADAS EXITOSAMENTE")
        print(f"{'='*60}{Style.RESET_ALL}\n")
        
        self.print_info("Verificar en la base de datos:")
        self.print_info("docker exec -i calofit_db psql -U postgres -d BD_Calofit -c \"SELECT * FROM alertas_salud;\"")
        
        return True

if __name__ == "__main__":
    try:
        tester = CaloFitTester()
        tester.run_all_tests()
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Pruebas interrumpidas por el usuario{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n{Fore.RED}Error inesperado: {e}{Style.RESET_ALL}")
