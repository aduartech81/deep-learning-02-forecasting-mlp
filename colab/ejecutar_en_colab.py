# %% [markdown]
# # Ejecutar el proyecto 02 en Google Colab
#
# Corre los 9 capítulos con los datos reales de Binance (BTCUSDT diario) y **respalda todo en Google Drive
# después de cada capítulo y cada 10 minutos**. Si Colab se desconecta, vuelva a ejecutar todas las celdas:
# recupera lo guardado y continúa donde iba (la búsqueda de arquitectura se reanuda por configuración).
#
# **Antes de empezar (una sola vez):** *Entorno de ejecución → Cambiar tipo de entorno de ejecución*: **CPU** y
# **RAM alta** (High-RAM). No use GPU: estas redes son pequeñas y en CPU, con un proceso por núcleo, rinden más.
# No hace falta ningún secreto: la API de Binance es pública. Mantenga la pestaña abierta mientras corre.

# %%
# 1. Hardware
import os, subprocess, time, threading, glob
import psutil
print(f"RAM: {psutil.virtual_memory().total / 2**30:.1f} GB | núcleos: {os.cpu_count()}")

# %%
# 2. Google Drive como respaldo persistente
from google.colab import drive
drive.mount("/content/drive")
DRIVE = "/content/drive/MyDrive/deep-learning-02-forecasting-mlp"
os.makedirs(f"{DRIVE}/done", exist_ok=True)
REPO = "/content/deep-learning-02-forecasting-mlp"

# %%
# 3. Código del repositorio y restauración de lo ya guardado en Drive
if not os.path.exists(REPO):
    subprocess.run(["git", "clone", "https://github.com/aduartech81/deep-learning-02-forecasting-mlp.git", REPO], check=True)
else:
    subprocess.run(["git", "-C", REPO, "pull", "--ff-only"], check=False)
os.chdir(REPO)
for sub in ["results", "data", "cache", "app"]:
    if os.path.exists(f"{DRIVE}/{sub}"):
        os.makedirs(sub, exist_ok=True)
        subprocess.run(f'cp -r "{DRIVE}/{sub}/." "{sub}/"', shell=True, check=True)
        print("restaurado desde Drive:", sub)
# notebooks: solo los de capítulos YA terminados (una corrección subida a GitHub no queda tapada por una copia vieja)
done = sorted(os.listdir(f"{DRIVE}/done"))
for ch in done:
    for f in glob.glob(f"{DRIVE}/notebooks/{ch}_*.ipynb"):
        subprocess.run(["cp", f, "notebooks/"], check=True)
print("capítulos terminados:", done or "ninguno todavía")

# %%
# 4. Dependencias (TensorFlow ya viene en Colab)
subprocess.run("pip -q install -r requirements.txt", shell=True, check=True)
import tensorflow as tf, sklearn
print("TensorFlow", tf.__version__, "| scikit-learn", sklearn.__version__)

# %%
# 5. Ejecutar capítulo por capítulo, con respaldo en Drive
env = dict(os.environ, FC_PROFILE="full", SKIP_BOOK="1", TF_CPP_MIN_LOG_LEVEL="3")
CHAPTERS = ["01", "02", "03", "04", "05", "06", "07", "08", "09"]


def backup():
    for sub in ["results", "notebooks", "data", "cache", "app"]:
        if os.path.exists(sub):
            os.makedirs(f"{DRIVE}/{sub}", exist_ok=True)
            subprocess.run(f'cp -r "{sub}/." "{DRIVE}/{sub}/"', shell=True, check=False)


def _respaldo_periodico():
    while True:
        time.sleep(600)
        try:
            backup()
        except Exception:
            pass


threading.Thread(target=_respaldo_periodico, daemon=True).start()
for ch in CHAPTERS:
    if os.path.exists(f"{DRIVE}/done/{ch}"):
        print(f"{ch}: ya terminado"); continue
    t0 = time.time()
    print(f"=== {ch} ({time.strftime('%H:%M')})", flush=True)
    r = subprocess.run(["bash", "scripts/run_all.sh", ch], env=env)
    backup()
    if r.returncode != 0:
        raise SystemExit(f"El capítulo {ch} falló: revise logs/{ch}_*.log (quedó respaldo en Drive).")
    open(f"{DRIVE}/done/{ch}", "w").write(f"{time.time() - t0:.0f} s\n")
    print(f"{ch}: listo en {(time.time() - t0) / 60:.0f} min", flush=True)
print("Todos los capítulos terminados.")

# %% [markdown]
# ## 6. Subir los resultados a GitHub
#
# Requiere un token *fine-grained* de GitHub con acceso solo a este repositorio y permiso **Contents: Read and
# write**. Al ejecutar la celda aparece una casilla oculta: pegue ahí el token y pulse Enter (no queda guardado).

# %%
import getpass
check = subprocess.run("grep -l 'Perfil: smoke' notebooks/*.ipynb", shell=True, capture_output=True, text=True)
assert not check.stdout.strip(), f"Hay notebooks con salidas sintéticas: {check.stdout}"
t = getpass.getpass("Token de GitHub (github_pat_...): ").strip()
cmd = ("git config user.name 'Andrés Duarte Chaves' && "
       "git config user.email '200296813+aduartech81@users.noreply.github.com' && "
       "git add notebooks results app && "
       "git commit -q -m 'Resultados con BTCUSDT real (Colab): notebooks ejecutados, tablas, figuras y modelos' && "
       "git pull -q --rebase origin main && "
       f"git push -q https://x-access-token:{t}@github.com/aduartech81/deep-learning-02-forecasting-mlp.git HEAD:main && "
       "git log --oneline -3")
r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
print((r.stdout + r.stderr).replace(t, "***")[-3000:], "codigo", r.returncode)
del t
