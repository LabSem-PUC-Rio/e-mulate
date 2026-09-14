import os
import subprocess
import time

LLQW = 5
LQW = 2
LQB = 7
MQW = 2.5
RLQW = 1
RQW = 2
RQB = 7

simulation_folder = "C:/codes_jer/e-mulate/windows_executable/temp_database/"
os.makedirs(simulation_folder, exist_ok=True)

file_ed_exe = os.path.abspath('C:/codes_jer/e-mulate/windows_executable/photocurrent_sim_windows.exe')

LLQW_str_exe = "{:.0f}".format(LLQW)
LQW_str_exe  = "{:.1f}d0".format(LQW)
LQB_str_exe  = "{:.1f}d0".format(LQB)
MQW_str_exe  = "{:.1f}d0".format(MQW)
RLQW_str_exe = "{:.0f}".format(RLQW)
RQW_str_exe  = "{:.1f}d0".format(RQW)
RQB_str_exe  = "{:.1f}d0".format(RQB)

command = [
    file_ed_exe,
    LLQW_str_exe,
    LQW_str_exe,
    LQB_str_exe,
    MQW_str_exe,
    RLQW_str_exe,
    RQW_str_exe,
    RQB_str_exe,
    simulation_folder
]

print(f"Iniciando proceso: {file_ed_exe}")
print(f"Argumentos: {command[1:]}")

# Iniciar el proceso capturando salida y errores
process = subprocess.Popen(
    command,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

start_time = time.time()
# while True:
#     return_code = process.poll()
#     if return_code is None:
#         elapsed = time.time() - start_time
#         print(f"\r[Estado] El ejecutable continúa ejecutándose... ({elapsed:.1f}s)", end="", flush=True)
#         time.sleep(1)
#     else:
#         print() # Salto de línea después del contador
#         stdout, stderr = process.communicate()
#         if return_code == 0:
#             print(f"[OK] El ejecutable terminó exitosamente en {time.time() - start_time:.2f}s.")
#         else:
#             print(f"[ERROR] El ejecutable terminó con código de error: {return_code}")
        
#         if stdout.strip():
#             print("\n--- Salida estándar (stdout) ---")
#             print(stdout)
#         if stderr.strip():
#             print("\n--- Salida de error (stderr) ---")
#             print(stderr)
#         break
