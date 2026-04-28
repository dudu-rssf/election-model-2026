"""
src.config — ponto único de configuração do projeto.

Carrega `config.yaml` na raiz do projeto e expõe constantes usadas por
todos os módulos. Nenhum outro arquivo deve ler o YAML diretamente.

Uso:
    from src.config import MODE, SEED, PATHS, MODE_CFG, CONFIG

    # Exemplo: escolher filtro de UFs/anos conforme modo (dev vs. prod)
    ufs = MODE_CFG["ufs"]
    anos = MODE_CFG["anos_presidencial"]
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# ------------------------------------------------------------
# Localização dos artefatos
# ------------------------------------------------------------
ROOT: Path = Path(__file__).resolve().parent.parent
CONFIG_PATH: Path = ROOT / "config.yaml"

if not CONFIG_PATH.exists():
    raise FileNotFoundError(
        f"config.yaml não encontrado em {CONFIG_PATH}. "
        "Rode os comandos a partir da raiz do projeto."
    )

CONFIG: dict[str, Any] = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

# ------------------------------------------------------------
# Modo e seed
# ------------------------------------------------------------
MODE: str = CONFIG["mode"]
if MODE not in {"dev", "prod"}:
    raise ValueError(f"config.yaml: mode deve ser 'dev' ou 'prod' (recebido: {MODE!r})")

SEED: int = int(CONFIG["seed"])

# Config específica do modo ativo (dev ou prod)
MODE_CFG: dict[str, Any] = CONFIG[MODE]

# ------------------------------------------------------------
# Paths absolutos derivados dos paths relativos do YAML
# ------------------------------------------------------------
PATHS: dict[str, Path] = {key: ROOT / rel for key, rel in CONFIG["paths"].items()}

# Garante que diretórios de saída existam. Não criamos data/raw
# aqui (quem baixa é src.ingestion), mas sim as pastas que scripts
# de qualquer fase podem querer escrever.
for key in ("data_interim", "data_processed", "reports", "models"):
    PATHS.setdefault(key, ROOT / key).mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# GCP (para ingestão via Base dos Dados)
# ------------------------------------------------------------
GCP_BILLING_PROJECT_ID: str | None = CONFIG.get("gcp", {}).get("billing_project_id")


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def set_global_seed(seed: int | None = None) -> int:
    """Fixa semente em random e numpy. Retorna a seed usada.

    Módulos que precisam de seeds específicos (lightgbm, pymc, etc.)
    continuam recebendo a seed via parâmetro — esta função cobre apenas
    os geradores globais.
    """
    s = int(seed if seed is not None else SEED)
    random.seed(s)
    np.random.seed(s)
    return s


def require_billing_project() -> str:
    """Exige que o billing_project_id do GCP esteja preenchido no config."""
    if not GCP_BILLING_PROJECT_ID:
        raise RuntimeError(
            "gcp.billing_project_id não está preenchido em config.yaml. "
            "Preencha antes de rodar a ingestão (Base dos Dados)."
        )
    return GCP_BILLING_PROJECT_ID


def summary() -> str:
    """Resumo textual útil para logs de scripts."""
    return (
        f"[config] mode={MODE} seed={SEED} "
        f"ufs={MODE_CFG.get('ufs')} "
        f"anos_presidencial={MODE_CFG.get('anos_presidencial')} "
        f"anos_municipal={MODE_CFG.get('anos_municipal')} "
        f"max_municipios={MODE_CFG.get('max_municipios')}"
    )


__all__ = [
    "ROOT",
    "CONFIG",
    "CONFIG_PATH",
    "MODE",
    "MODE_CFG",
    "SEED",
    "PATHS",
    "GCP_BILLING_PROJECT_ID",
    "set_global_seed",
    "require_billing_project",
    "summary",
]
