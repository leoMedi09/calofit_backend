import resend
import os
from dotenv import load_dotenv

load_dotenv()

resend.api_key = os.getenv("RESEND_API_KEY")

class EmailService:
    @staticmethod
    def send_otp_email(email_to: str, code: str):
        try:
            params = {
                "from": "CaloFit <onboarding@resend.dev>",
                "to": [email_to],
                "subject": f"{code} es tu código de seguridad CaloFit",
                "html": f"""
                <div style="font-family: sans-serif; max-width: 400px; margin: auto; border: 1px solid #eee; padding: 20px; border-radius: 10px;">
                    <h2 style="color: #4CAF50; text-align: center;">CaloFit</h2>
                    <p>Has solicitado restablecer tu contraseña. Usa el siguiente código:</p>
                    <div style="background: #f4f4f4; padding: 20px; text-align: center; font-size: 32px; font-weight: bold; letter-spacing: 10px; color: #333;">
                        {code}
                    </div>
                    <p style="font-size: 12px; color: #777; margin-top: 20px;">
                        Este código expirará en 15 minutos. Si no solicitaste este cambio, ignora este correo.
                    </p>
                </div>
                """
            }
            email = resend.Emails.send(params)
            return email
        except Exception as e:
            print(f"Error enviando correo con Resend: {e}")
            return None

    @staticmethod
    def send_welcome_credentials_email(email_to: str, dni: str, nutricionista_name: str):
        try:
            params = {
                "from": "CaloFit <onboarding@resend.dev>",
                "to": [email_to],
                "subject": f"¡Bienvenido a CaloFit! Tu nutricionista te ha registrado",
                "html": f"""
                <div style="font-family: sans-serif; max-width: 400px; margin: auto; border: 1px solid #eee; padding: 20px; border-radius: 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.05);">
                    <div style="text-align: center; margin-bottom: 20px;">
                        <h1 style="color: #1E88E5; margin: 0; font-size: 28px;">CaloFit</h1>
                        <p style="color: #666; margin-top: 5px; font-size: 14px;">Tu camino a una vida saludable</p>
                    </div>
                    
                    <p style="color: #333; font-size: 15px; line-height: 1.5;">¡Hola!</p>
                    <p style="color: #333; font-size: 15px; line-height: 1.5;">
                        Tu nutricionista <b>{nutricionista_name}</b> acaba de crear tu cuenta en nuestra plataforma premium.
                    </p>
                    
                    <div style="background: #F1F5F9; border-left: 4px solid #1E88E5; padding: 15px; margin: 20px 0; border-radius: 4px;">
                        <p style="margin: 0 0 10px 0; color: #555; font-size: 13px; text-transform: uppercase; font-weight: bold;">TUS CREDENCIALES DE ACCESO:</p>
                        <p style="margin: 0 0 5px 0;"><strong>Correo:</strong> {email_to}</p>
                        <p style="margin: 0;"><strong>Contraseña temporal:</strong> {dni}</p>
                    </div>
                    
                    <p style="color: #333; font-size: 15px; line-height: 1.5;">
                        Descarga la aplicación de CaloFit e ingresa con estos datos. Cuando inicies sesión por primera vez, <b>te pediremos completar tu perfil y que cambies tu contraseña</b> por motivos de seguridad.
                    </p>
                    <hr style="border: none; border-top: 1px solid #eee; margin: 25px 0;">
                    <p style="font-size: 12px; color: #999; text-align: center; margin: 0;">
                        Este es un mensaje automático del sistema CaloFit. Por favor no respondas a este correo.
                    </p>
                </div>
                """
            }
            email = resend.Emails.send(params)
            print(f"Correo de bienvenida enviado a {email_to}")
            return email
        except Exception as e:
            print(f"Error enviando correo de bienvenida con Resend: {e}")
            return None

    @staticmethod
    def send_welcome_credentials_gmail(email_to: str, dni: str, nutricionista_name: str):
        """
        Envía correos gratuitos y sin restricción de dominios usando el SMTP de Gmail
        (Requiere Contraseña de Aplicación de Google)
        """
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        import os
        
        gmail_user = os.getenv("GMAIL_SENDER")
        gmail_password = os.getenv("GMAIL_APP_PASSWORD")
        
        if not gmail_user or not gmail_password:
            print("Faltan credenciales GMAIL_SENDER o GMAIL_APP_PASSWORD en el archivo .env")
            return None

        # Construir el Mensaje HTML premium
        msg = MIMEMultipart("alternative")
        msg["Subject"] = "¡Bienvenido a CaloFit! Tu nutricionista te ha registrado"
        msg["From"] = f"CaloFit <{gmail_user}>"
        msg["To"] = email_to

        html_body = f"""
        <div style="font-family: sans-serif; max-width: 400px; margin: auto; border: 1px solid #eee; padding: 20px; border-radius: 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.05);">
            <div style="text-align: center; margin-bottom: 20px;">
                <h1 style="color: #1E88E5; margin: 0; font-size: 28px;">CaloFit</h1>
                <p style="color: #666; margin-top: 5px; font-size: 14px;">Tu camino a una vida saludable</p>
            </div>
            
            <p style="color: #333; font-size: 15px; line-height: 1.5;">¡Hola!</p>
            <p style="color: #333; font-size: 15px; line-height: 1.5;">
                Tu nutricionista <b>{nutricionista_name}</b> acaba de crear tu cuenta en nuestra plataforma premium.
            </p>
            
            <div style="background: #F1F5F9; border-left: 4px solid #1E88E5; padding: 15px; margin: 20px 0; border-radius: 4px;">
                <p style="margin: 0 0 10px 0; color: #555; font-size: 13px; text-transform: uppercase; font-weight: bold;">TUS CREDENCIALES DE ACCESO:</p>
                <p style="margin: 0 0 5px 0;"><strong>Correo:</strong> {email_to}</p>
                <p style="margin: 0;"><strong>Contraseña temporal:</strong> {dni}</p>
            </div>
            
            <p style="color: #333; font-size: 15px; line-height: 1.5;">
                Descarga la aplicación de CaloFit e ingresa con estos datos. Cuando inicies sesión por primera vez, <b>te pediremos completar tu perfil y que cambies tu contraseña</b> por motivos de seguridad.
            </p>
        </div>
        """
        
        parte_html = MIMEText(html_body, "html")
        msg.attach(parte_html)

        try:
            # Conectar a Gmail SMTP por el puerto 465 (Seguro SSL)
            server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
            server.login(gmail_user, gmail_password)
            server.sendmail(gmail_user, email_to, msg.as_string())
            server.quit()
            print(f"Correo de bienvenida enviado a {email_to} usando GMAIL SMTP")
            return True
        except Exception as e:
            print(f"Error crítico enviando correo vía Gmail: {e}")
            return None

    @staticmethod
    def send_welcome_credentials_brevo(email_to: str, dni: str, nutricionista_name: str):
        """
        Envía correos gratuitos y sin restricción usando la API V3 de Brevo.
        (Requiere BREVO_API_KEY y BREVO_SENDER en .env)
        """
        import requests
        import os
        
        api_key = os.getenv("BREVO_API_KEY")
        sender_email = os.getenv("BREVO_SENDER")
        
        if not api_key or not sender_email:
            print("Faltan credenciales BREVO_API_KEY o BREVO_SENDER en el archivo .env")
            return None

        url = "https://api.brevo.com/v3/smtp/email"
        
        headers = {
            "accept": "application/json",
            "api-key": api_key,
            "content-type": "application/json"
        }
        
        html_body = f"""
        <div style="font-family: sans-serif; max-width: 400px; margin: auto; border: 1px solid #eee; padding: 20px; border-radius: 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.05);">
            <div style="text-align: center; margin-bottom: 20px;">
                <h1 style="color: #1E88E5; margin: 0; font-size: 28px;">CaloFit</h1>
                <p style="color: #666; margin-top: 5px; font-size: 14px;">Tu camino a una vida saludable</p>
            </div>
            
            <p style="color: #333; font-size: 15px; line-height: 1.5;">¡Hola!</p>
            <p style="color: #333; font-size: 15px; line-height: 1.5;">
                Tu nutricionista <b>{nutricionista_name}</b> te registró en nuestra plataforma premium.
            </p>
            
            <div style="background: #F1F5F9; border-left: 4px solid #1E88E5; padding: 15px; margin: 20px 0; border-radius: 4px;">
                <p style="margin: 0 0 10px 0; color: #555; font-size: 13px; text-transform: uppercase; font-weight: bold;">TUS CREDENCIALES DE ACCESO:</p>
                <p style="margin: 0 0 5px 0;"><strong>Correo:</strong> {email_to}</p>
                <p style="margin: 0;"><strong>Contraseña temporal:</strong> {dni}</p>
            </div>
            
            <p style="color: #333; font-size: 15px; line-height: 1.5;">
                Ingresa con estos datos en la aplicación. Cuando inicies sesión por primera vez, <b>te pediremos completar tu perfil y que cambies tu contraseña</b> por motivos de seguridad.
            </p>
        </div>
        """

        payload = {
            "sender": {"name": "CaloFit", "email": sender_email},
            "to": [{"email": email_to}],
            "subject": f"¡Bienvenido a CaloFit! {nutricionista_name} te registró",
            "htmlContent": html_body
        }

        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            print(f"Correo de bienvenida enviado a {email_to} usando BREVO API")
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error crítico enviando correo vía Brevo: {e}")
            if e.response is not None:
                print(f"Detalle de Brevo: {e.response.text}")
            return None

    @staticmethod
    def send_password_reset_brevo(email_to: str, code: str):
        import requests
        import os
        
        api_key = os.getenv("BREVO_API_KEY")
        sender_email = os.getenv("BREVO_SENDER")
        
        if not api_key or not sender_email:
            print("Faltan credenciales BREVO_API_KEY o BREVO_SENDER en el archivo .env")
            return None

        url = "https://api.brevo.com/v3/smtp/email"
        
        headers = {
            "accept": "application/json",
            "api-key": api_key,
            "content-type": "application/json"
        }
        
        html_body = f"""
        <div style="font-family: sans-serif; max-width: 400px; margin: auto; border: 1px solid #eee; padding: 20px; border-radius: 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.05);">
            <div style="text-align: center; margin-bottom: 20px;">
                <h1 style="color: #1E88E5; margin: 0; font-size: 28px;">CaloFit</h1>
                <p style="color: #666; margin-top: 5px; font-size: 14px;">Recuperación de Contraseña</p>
            </div>
            
            <p style="color: #333; font-size: 15px; line-height: 1.5;">Hola,</p>
            <p style="color: #333; font-size: 15px; line-height: 1.5;">
                Has solicitado restablecer tu contraseña. Ingresa el siguiente código de 6 dígitos en la aplicación:
            </p>
            
            <div style="background: #F1F5F9; border-left: 4px solid #1E88E5; padding: 20px; margin: 20px 0; border-radius: 4px; text-align: center;">
                <p style="margin: 0; font-size: 32px; font-weight: bold; letter-spacing: 8px; color: #1E88E5;">{code}</p>
            </div>
            
            <p style="color: #666; font-size: 12px; line-height: 1.5;">
                Este código expirará en 15 minutos. Si no solicitaste este cambio, por favor ignora este correo.
            </p>
        </div>
        """

        payload = {
            "sender": {"name": "CaloFit Seguridad", "email": sender_email},
            "to": [{"email": email_to}],
            "subject": f"{code} es tu código de recuperación CaloFit",
            "htmlContent": html_body
        }

        import time
        last_error = None
        for attempt in range(3):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=15)
                response.raise_for_status()
                print(f"Código de recuperación enviado a {email_to} usando BREVO API")
                return response.json()
            except requests.exceptions.RequestException as e:
                last_error = e
                if attempt < 2:
                    time.sleep(2)
        print(f"Error crítico enviando código vía Brevo tras 3 intentos: {last_error}")
        return None

    @staticmethod
    def send_welcome_staff_brevo(email_to: str, password: str, staff_name: str, role_name: str, admin_name: str):
        """Correo de bienvenida al equipo para nuevo personal (nutri, coach, admin)."""
        import requests, os
        api_key = os.getenv("BREVO_API_KEY")
        sender_email = os.getenv("BREVO_SENDER")
        if not api_key or not sender_email:
            print("Faltan credenciales BREVO_API_KEY o BREVO_SENDER en .env")
            return None

        role_label = {
            "nutritionist": "Nutricionista", "nutricionista": "Nutricionista",
            "coach": "Entrenador", "entrenador": "Entrenador", "trainer": "Entrenador",
            "admin": "Administrador", "administrador": "Administrador",
        }.get(role_name.lower(), role_name.capitalize())

        html_body = f"""
        <div style="font-family:sans-serif;max-width:460px;margin:auto;border:1px solid #eee;padding:28px;border-radius:14px;box-shadow:0 4px 12px rgba(0,0,0,0.06)">
          <div style="text-align:center;margin-bottom:24px">
            <h1 style="color:#1E88E5;margin:0;font-size:28px">CaloFit</h1>
            <p style="color:#888;margin-top:6px;font-size:13px">Gimnasio World Light Lambayeque</p>
          </div>
          <p style="color:#333;font-size:15px;line-height:1.6">Hola, <strong>{staff_name}</strong> 👋</p>
          <p style="color:#333;font-size:15px;line-height:1.6">
            El administrador <strong>{admin_name}</strong> te ha dado acceso al sistema CaloFit
            como <strong>{role_label}</strong>. Ya puedes ingresar a la aplicación con las siguientes credenciales:
          </p>
          <div style="background:#F1F5F9;border-left:4px solid #1E88E5;padding:18px;margin:20px 0;border-radius:6px">
            <p style="margin:0 0 8px 0;color:#555;font-size:12px;text-transform:uppercase;font-weight:bold;letter-spacing:0.5px">Tus credenciales de acceso</p>
            <p style="margin:0 0 6px 0;font-size:15px"><strong>Correo:</strong> {email_to}</p>
            <p style="margin:0;font-size:15px"><strong>Contraseña:</strong> {password}</p>
          </div>
          <p style="color:#555;font-size:13px;line-height:1.5">
            Ingresa a la app CaloFit con estos datos. Te recomendamos cambiar tu contraseña desde tu perfil después del primer ingreso.
          </p>
          <hr style="border:none;border-top:1px solid #eee;margin:24px 0">
          <p style="font-size:11px;color:#aaa;text-align:center;margin:0">Mensaje automático del sistema CaloFit · No respondas a este correo.</p>
        </div>
        """
        payload = {
            "sender": {"name": "CaloFit", "email": sender_email},
            "to": [{"email": email_to}],
            "subject": f"¡Bienvenido al equipo CaloFit, {staff_name.split()[0]}!",
            "htmlContent": html_body,
        }
        try:
            r = requests.post("https://api.brevo.com/v3/smtp/email",
                              headers={"accept": "application/json", "api-key": api_key, "content-type": "application/json"},
                              json=payload)
            r.raise_for_status()
            print(f"Correo de bienvenida staff enviado a {email_to}")
            return r.json()
        except requests.exceptions.RequestException as e:
            print(f"Error enviando bienvenida staff vía Brevo: {e}")
            return None

    @staticmethod
    def send_password_updated_staff_brevo(email_to: str, staff_name: str, new_password: str, admin_name: str):
        """Notificación al staff cuando el admin cambia su contraseña."""
        import requests, os
        api_key = os.getenv("BREVO_API_KEY")
        sender_email = os.getenv("BREVO_SENDER")
        if not api_key or not sender_email:
            print("Faltan credenciales BREVO_API_KEY o BREVO_SENDER en .env")
            return None

        html_body = f"""
        <div style="font-family:sans-serif;max-width:460px;margin:auto;border:1px solid #eee;padding:28px;border-radius:14px;box-shadow:0 4px 12px rgba(0,0,0,0.06)">
          <div style="text-align:center;margin-bottom:24px">
            <h1 style="color:#1E88E5;margin:0;font-size:28px">CaloFit</h1>
            <p style="color:#888;margin-top:6px;font-size:13px">Aviso de Seguridad</p>
          </div>
          <p style="color:#333;font-size:15px;line-height:1.6">Hola, <strong>{staff_name}</strong></p>
          <p style="color:#333;font-size:15px;line-height:1.6">
            El administrador <strong>{admin_name}</strong> ha actualizado tu contraseña de acceso al sistema CaloFit.
          </p>
          <div style="background:#F1F5F9;border-left:4px solid #1E88E5;padding:18px;margin:20px 0;border-radius:6px">
            <p style="margin:0 0 8px 0;color:#555;font-size:12px;text-transform:uppercase;font-weight:bold;letter-spacing:0.5px">Tus nuevas credenciales</p>
            <p style="margin:0 0 6px 0;font-size:15px"><strong>Correo:</strong> {email_to}</p>
            <p style="margin:0;font-size:15px"><strong>Nueva contraseña:</strong> {new_password}</p>
          </div>
          <p style="color:#e53935;font-size:13px;line-height:1.5">
            Si no solicitaste este cambio, comunícate de inmediato con el administrador del gimnasio.
          </p>
          <hr style="border:none;border-top:1px solid #eee;margin:24px 0">
          <p style="font-size:11px;color:#aaa;text-align:center;margin:0">Mensaje automático del sistema CaloFit · No respondas a este correo.</p>
        </div>
        """
        payload = {
            "sender": {"name": "CaloFit Seguridad", "email": sender_email},
            "to": [{"email": email_to}],
            "subject": "Tu contraseña CaloFit fue actualizada",
            "htmlContent": html_body,
        }
        try:
            r = requests.post("https://api.brevo.com/v3/smtp/email",
                              headers={"accept": "application/json", "api-key": api_key, "content-type": "application/json"},
                              json=payload)
            r.raise_for_status()
            print(f"Notificación de cambio de contraseña enviada a {email_to}")
            return r.json()
        except requests.exceptions.RequestException as e:
            print(f"Error enviando notificación de contraseña vía Brevo: {e}")
            return None