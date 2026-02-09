from fastapi import APIRouter


router = APIRouter()

@router.get("/")
async def listar_ejercicios():
    return {"mensaje": "Lista de ejercicios del Gimnasio World Light"}