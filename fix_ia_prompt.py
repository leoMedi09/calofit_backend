import re
import os

path = r'd:\PROYECTO_TESIS\PROYECTO\calofit_backend\app\services\ia_service.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Fix prompt
old_prompt = r'- ⛔ PROHIBIDO: Escribir "Opción 1:" o "Opción 2:" dentro del \[CALOFIT_HEADER\]. SOLO pon el nombre del plato.'
new_prompt = r'- ⛔ PROHIBIDO: Escribir "Opción 1:", "Opción 2:" o "Opción 3:" al final del bloque [CALOFIT_INTENT: CHAT].\n   - ⛔ PROHIBIDO: Escribir "Opción 1:", "Opción 2:" o "Opción 3:" dentro del [CALOFIT_HEADER]. SOLO el nombre del plato.'

content = re.sub(old_prompt, new_prompt, content)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Done")
