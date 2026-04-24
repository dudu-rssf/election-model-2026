#!/usr/bin/env python
"""
scripts/02_build_panel.py — Fase 2: painel mestre.

Lê `data/raw/` (usando sufixo dev quando aplicável), constrói:
  * `painel_mestre` — uma linha por (município, ano_presidencial) com prefeito vigente.
  * `presidencial_long` — tabela long de votação por candidato (alvo).

Saídas em `data/interim/`.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import MODE_CFG, set_global_seed, summary  # noqa: E402
from src.features import io as fio  # noqa: E402
from src.features import panel as panel_mod  # noqa: E402
from src.features import target as target_mod  # noqa: E402


log = logging.getLogger("02_build_panel")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fase 2 — construir painel mestre + tabela long presidencial")
    p.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args()


def configurar_logging(level: str) -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        level=getattr(logging, level),
    )


def main() -> int:
    args = parse_args()
    configurar_logging(args.log_level)
    set_global_seed()
    log.info(summary())

    # 1. Carrega insumos da Fase 1
    df_pres = fio.load_raw("resultados_presidenciais")
    df_pref = fio.load_raw("resultados_prefeito")
    df_cand_pres = fio.load_raw("candidatos_presidenciais")
    df_diretorio = fio.load_raw("diretorio_municipios")

    # Coligação do prefeito vem da tabela `partidos` (BD moveu em abr/2026).
    # Se ainda não foi ingerida, segue com NA + warning (não bloqueia o painel).
    try:
        df_partidos_pref = fio.load_raw("partidos_prefeito")
    except FileNotFoundError:
        log.warning(
            "partidos_prefeito.parquet não encontrado — "
            "rode `python scripts/01_ingest.py --only partidos_prefeito`. "
            "Seguindo com mayor_coligacao = NA."
        )
        df_partidos_pref = None

    log.info(
        "insumos: pres=%d, pref=%d, cand_pres=%d, partidos_pref=%s, dir=%d",
        len(df_pres), len(df_pref), len(df_cand_pres),
        len(df_partidos_pref) if df_partidos_pref is not None else "—",
        len(df_diretorio),
    )

    # 2. Painel mestre
    painel = panel_mod.construir_painel_mestre(
        diretorio=df_diretorio,
        df_prefeito=df_pref,
        df_partidos_prefeito=df_partidos_pref,
        anos_presidenciais=MODE_CFG["anos_presidencial"],
    )
    fio.save_interim(painel, "painel_mestre")

    # 3. Tabela long presidencial (alvo) — `nome_candidato` vem de candidatos_presidenciais,
    # porque a tabela resultados_candidato_municipio do BD não inclui mais esse campo.
    pres_long = target_mod.construir_presidencial_long(df_pres, df_cand_pres)
    fio.save_interim(pres_long, "presidencial_long")

    log.info("Fase 2 OK. painel=%d linhas; pres_long=%d linhas", len(painel), len(pres_long))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
