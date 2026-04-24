#!/usr/bin/env bash
# ============================================================
# run_prod.sh — roda o pipeline completo em modo prod
# ------------------------------------------------------------
# Usar em máquina pesada (Colab Pro, servidor próprio, cloud).
# Este script NÃO deve ser executado no laptop de desenvolvimento.
#
# Pré-requisitos:
#   - venv criado e dependências instaladas (`make install`)
#   - config.yaml com gcp.billing_project_id preenchido
#   - autenticação Google Cloud configurada (gcloud auth application-default login)
# ============================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Força mode=prod no config.yaml de forma idempotente
python - <<'PY'
import yaml
from pathlib import Path
cfg_path = Path("config.yaml")
cfg = yaml.safe_load(cfg_path.read_text())
if cfg.get("mode") != "prod":
    cfg["mode"] = "prod"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    print("[run_prod] config.yaml: mode -> prod")
else:
    print("[run_prod] config.yaml já em mode=prod")

if not cfg.get("gcp", {}).get("billing_project_id"):
    raise SystemExit("[run_prod] ERRO: preencha gcp.billing_project_id no config.yaml")
PY

PY_BIN="${PY_BIN:-python}"

echo "[run_prod] 01_ingest"       && $PY_BIN scripts/01_ingest.py
echo "[run_prod] 02_build_panel"  && $PY_BIN scripts/02_build_panel.py
echo "[run_prod] 03_features"     && $PY_BIN scripts/03_features.py
echo "[run_prod] 04_hypotheses"   && $PY_BIN scripts/04_hypotheses.py
echo "[run_prod] 05_train"        && $PY_BIN scripts/05_train.py
echo "[run_prod] 06_predict"      && $PY_BIN scripts/06_predict.py

echo "[run_prod] Pipeline prod concluído."
echo "  Modelos em:     models/"
echo "  Predições em:   data/processed/"
echo "  Relatórios em:  reports/"
