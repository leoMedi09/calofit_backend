import os

def fix_python_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    fixed = []
    level = 0
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            fixed.append("\n")
            continue
            
        # These keywords decrease indentation level FOR THE CURRENT LINE
        if stripped.startswith(("elif ", "else:", "except ", "finally:")):
            temp_level = max(0, level - 4)
        else:
            temp_level = level
            
        fixed.append(" " * temp_level + stripped + "\n")
        
        # Adjust level for NEXT lines
        if stripped.endswith(":") and not stripped.startswith("#"):
            level += 4
        
        # Class and Def have fixed levels (assuming top level class)
        if stripped.startswith("class "):
            fixed[-1] = stripped + "\n" # Remove any indent
            level = 4
        if stripped.startswith("def "):
            fixed[-1] = "    " + stripped + "\n" # 4 spaces
            level = 8
            
        # Heuristic to decrease indent after return/raise
        if stripped.startswith(("return ", "raise ", "break", "continue")):
            # This is tricky without a full parser, but usually we go back 4
            level = max(8, level - 4) # Don't go below method level

    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(fixed)

if __name__ == "__main__":
    fix_python_file(r"d:\PROYECTO_TESIS\PROYECTO\calofit_backend\app\services\ia_service.py")
