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