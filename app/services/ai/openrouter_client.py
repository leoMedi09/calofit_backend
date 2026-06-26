import logging
import httpx
from types import SimpleNamespace

logger = logging.getLogger(__name__)

# Mapeo de modelos desde nombres de Groq/OpenAI a nombres de OpenRouter
MODEL_MAPPING = {
    "groq/compound-mini": "google/gemini-2.5-flash",
    "openai/gpt-oss-20b": "meta-llama/llama-3.3-70b-instruct",
    "openai/gpt-oss-120b": "meta-llama/llama-3.3-70b-instruct",
    "llama-3.3-70b-versatile": "meta-llama/llama-3.3-70b-instruct",
    "llama-3.1-8b-instant": "meta-llama/llama-3.1-8b-instruct",
}

class OpenRouterClient:
    """
    Clase cliente compatible con la interfaz mínima de AsyncGroq para OpenRouter.
    """
    def __init__(self, api_key: str, timeout: float = 180.0, **kwargs):
        self.api_key = api_key
        # En caso de que se pase un objeto httpx.Timeout
        if isinstance(timeout, httpx.Timeout):
            self.timeout = timeout.read or 180.0
        else:
            self.timeout = float(timeout)
        self.chat = OpenRouterChat(self)

class OpenRouterChat:
    def __init__(self, client: OpenRouterClient):
        self.client = client
        self.completions = OpenRouterCompletions(self.client)

class OpenRouterCompletions:
    def __init__(self, client: OpenRouterClient):
        self.client = client

    async def create(
        self,
        model: str,
        messages: list,
        max_tokens: int = None,
        temperature: float = None,
        **kwargs
    ):
        # Mapear modelo
        mapped_model = MODEL_MAPPING.get(model, model)
        logger.info(f"OpenRouter: mapeando modelo '{model}' a '{mapped_model}'")
        
        # Limitar max_tokens para evitar errores 402 en OpenRouter (limite de saldo de desarrollo)
        if max_tokens is None:
            max_tokens = 1000
            
        payload = {
            "model": mapped_model,
            "messages": messages,
            "max_tokens": max_tokens
        }
        if temperature is not None:
            payload["temperature"] = temperature

        headers = {
            "Authorization": f"Bearer {self.client.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/calofit/calofit",
            "X-Title": "CaloFit App",
        }

        async with httpx.AsyncClient() as client:
            try:
                resp = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.client.timeout
                )
                if resp.status_code != 200:
                    error_msg = f"Error en OpenRouter API ({resp.status_code}): {resp.text}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                
                # Reconstruir el objeto de respuesta esperado por el llamador
                message_obj = SimpleNamespace(content=content)
                choice_obj = SimpleNamespace(message=message_obj)
                response_obj = SimpleNamespace(choices=[choice_obj])
                return response_obj
                
            except Exception as e:
                logger.error(f"Excepción llamando a OpenRouter: {e}")
                raise e
