#!/usr/bin/env bash
# Ejecuta los notebooks en orden (guardando salidas) y construye el Jupyter Book.
# Uso:
#   bash scripts/run_all.sh                  # perfil full (datos reales de Binance)
#   FC_PROFILE=smoke bash scripts/run_all.sh # prueba de humo con una serie sintética
#   bash scripts/run_all.sh 03 04            # solo los notebooks cuyo nombre empieza por 03 y 04
set -euo pipefail
cd "$(dirname "$0")/.."
export FC_PROFILE="${FC_PROFILE:-full}"
export TF_CPP_MIN_LOG_LEVEL=3
mkdir -p logs
python scripts/sync_notebooks.py

sel=("$@")
for nb in notebooks/[0-9][0-9]_*.ipynb; do
  name=$(basename "$nb")
  if [[ ${#sel[@]} -gt 0 ]]; then
    ok=0; for s in "${sel[@]}"; do [[ "$name" == "$s"* ]] && ok=1; done
    [[ $ok -eq 0 ]] && continue
  fi
  echo "=== $(date '+%H:%M:%S') ejecutando $name (perfil $FC_PROFILE)"
  (cd notebooks && jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3 "$name") 2>&1 | tee "logs/${name%.ipynb}.log"
done

if [[ "${SKIP_BOOK:-0}" != "1" ]]; then
  echo "=== construyendo el Jupyter Book"
  jupyter-book build . 2>&1 | tail -5
fi
