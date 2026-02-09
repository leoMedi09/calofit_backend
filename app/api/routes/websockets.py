from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import json
import asyncio

router = APIRouter()

# Conexiones activas (en producción usar Redis o similar)
active_connections = {}

@router.websocket("/ws/chat/{user_id}")
async def chat_websocket(websocket: WebSocket, user_id: int):
    """
    WebSocket para chat en tiempo real con streaming de IA.
    Permite conversación fluida con el asistente virtual.
    """
    await websocket.accept()
    active_connections[user_id] = websocket

    try:
        # Mensaje de bienvenida
        await websocket.send_json({
            "type": "welcome",
            "message": f"¡Hola! Soy tu asistente de CaloFit. ¿En qué puedo ayudarte hoy?",
            "user_id": user_id
        })

        while True:
            try:
                # Recibir mensaje del usuario
                data = await websocket.receive_text()
                message_data = json.loads(data)

                comando_texto = message_data.get("message", "")
                adherencia_pct = message_data.get("adherencia", 50)
                progreso_pct = message_data.get("progreso", 50)

                # Respuesta simple sin IA
                await websocket.send_json({
                    "type": "intent_detected",
                    "intent": "chat_general",
                    "entities": []
                })

                await websocket.send_json({
                    "type": "response_chunk",
                    "chunk": f"Recibí tu mensaje: '{comando_texto}'. Adherencia: {adherencia_pct}%, Progreso: {progreso_pct}%.",
                    "chunk_id": 0,
                    "total_chunks": 1
                })

                await websocket.send_json({
                    "type": "alerta",
                    "message": f"Tu adherencia está en {adherencia_pct}%. ¡Sigue así!"
                })

            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Mensaje JSON inválido"
                })
            except Exception as e:
                print(f"Error procesando mensaje: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": f"Error interno: {str(e)}"
                })

    except WebSocketDisconnect:
        print(f"Usuario {user_id} desconectado")
    except Exception as e:
        print(f"Error general en WebSocket chat: {e}")
    finally:
        if user_id in active_connections:
            del active_connections[user_id]

@router.websocket("/ws/training/{user_id}")
async def training_websocket(websocket: WebSocket, user_id: int):
    """
    WebSocket para monitoreo de entrenamiento en tiempo real.
    Recibe datos biométricos y envía actualizaciones de calorías.
    """
    await websocket.accept()
    active_connections[f"training_{user_id}"] = websocket

    try:
        await websocket.send_json({
            "type": "training_started",
            "message": "Monitoreo de entrenamiento iniciado. Envía datos de ejercicio."
        })

        while True:
            try:
                data = await websocket.receive_text()
                training_data = json.loads(data)

                # Extraer datos del ejercicio
                tipo_ejercicio = training_data.get("tipo_ejercicio", 1)
                duracion = training_data.get("duracion", 30)  # minutos
                intensidad = training_data.get("intensidad", 5)  # 1-10
                perfil_usuario = training_data.get("perfil", {})

                # Calcular calorías (valor fijo por ahora)
                calorias = 150  # valor por defecto

                # Enviar actualización
                await websocket.send_json({
                    "type": "calories_update",
                    "calories_burned": calorias,
                    "exercise_type": tipo_ejercicio,
                    "duration": duracion,
                    "intensity": intensidad,
                    "timestamp": training_data.get("timestamp")
                })

                # Verificar metas
                meta_calorias = training_data.get("meta_calorias", 300)
                if calorias and calorias >= meta_calorias:
                    await websocket.send_json({
                        "type": "achievement",
                        "message": f"¡Felicitaciones! Has alcanzado tu meta de {meta_calorias} calorías quemadas.",
                        "achievement": "meta_alcanzada"
                    })

            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Datos de entrenamiento JSON inválidos"
                })
            except Exception as e:
                print(f"Error procesando datos de entrenamiento: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": f"Error en monitoreo: {str(e)}"
                })

    except WebSocketDisconnect:
        print(f"Entrenamiento de usuario {user_id} terminado")
    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "message": f"Error en monitoreo: {str(e)}"
        })
    finally:
        if f"training_{user_id}" in active_connections:
            del active_connections[f"training_{user_id}"]

@router.websocket("/ws/notifications/{user_id}")
async def notifications_websocket(websocket: WebSocket, user_id: int):
    """
    WebSocket para notificaciones personalizadas con lógica difusa.
    """
    await websocket.accept()
    active_connections[f"notifications_{user_id}"] = websocket

    try:
        await websocket.send_json({
            "type": "notifications_connected",
            "message": "Sistema de notificaciones activado."
        })

        # Mantener conexión abierta para notificaciones push
        while True:
            try:
                # Esperar mensajes del cliente o mantener conexión viva
                data = await websocket.receive_text()
                # Procesar cualquier mensaje del cliente si es necesario
                print(f"Mensaje recibido en notificaciones: {data}")
            except Exception as e:
                # Si hay error recibiendo, continuar el bucle
                print(f"Error en recepción de notificaciones: {e}")
                await asyncio.sleep(1)  # Pequeña pausa para evitar bucles rápidos

    except WebSocketDisconnect:
        print(f"Notificaciones de usuario {user_id} desconectadas")
    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "message": f"Error en notificaciones: {str(e)}"
        })
    finally:
        if f"notifications_{user_id}" in active_connections:
            del active_connections[f"notifications_{user_id}"]

# Función para enviar notificaciones push (puedes llamarla desde otras partes del código)
async def send_notification_to_user(user_id: int, notification_data: dict):
    """
    Función para enviar notificaciones push a un usuario específico.
    """
    connection_key = f"notifications_{user_id}"
    if connection_key in active_connections:
        websocket = active_connections[connection_key]
        try:
            await websocket.send_json({
                "type": "push_notification",
                **notification_data
            })
        except Exception as e:
            print(f"Error enviando notificación a {user_id}: {e}")