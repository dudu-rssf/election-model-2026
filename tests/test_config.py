"""Smoke test do módulo de configuração.

A Fase 0 só requer que `from src.config import ...` funcione e que
o modo padrão seja 'dev'. Testes mais substanciais entram nas fases
subsequentes.
"""
from __future__ import annotations

from pathlib import Path


def test_config_importa_e_expoe_constantes():
    from src import config

    assert config.MODE in {"dev", "prod"}
    assert isinstance(config.SEED, int)
    assert isinstance(config.PATHS, dict)
    for key in ("data_raw", "data_interim", "data_processed", "models", "reports"):
        assert key in config.PATHS, f"PATHS sem chave obrigatória: {key}"
        assert isinstance(config.PATHS[key], Path)


def test_modo_dev_padrao():
    """Cowork nunca deve mexer em prod; o default do repo é dev."""
    from src import config

    assert config.MODE == "dev", (
        "config.yaml deve estar em mode=dev no repositório. "
        "Trocar para prod é tarefa da máquina de execução pesada."
    )


def test_mode_cfg_tem_chaves_esperadas():
    from src import config

    for key in ("ufs", "anos_presidencial", "max_municipios"):
        assert key in config.MODE_CFG, f"MODE_CFG sem chave: {key}"


def test_set_global_seed_retorna_int():
    from src import config

    s = config.set_global_seed()
    assert s == config.SEED
