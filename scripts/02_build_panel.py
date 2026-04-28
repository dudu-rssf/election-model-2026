#!/usr/bin/env python
"""
scripts/02_build_panel.py — Fase 2: painel mestre.

Lê `data/raw/` (usando sufixo dev quando aplicável), constrói:

Eixo presidencial (Fase 2/3/4):
  * `painel_mestre` — uma linha por (município, ano_presidencial) com prefeito
    vigente (X-2).
  * `presidencial_long` — tabela long de votação por candidato (alvo).

Eixo municipal (Fase 4.5):
  * `painel_municipal` — uma linha por (município, ano_municipal) com prefeito
    vigente (X-4 = vencedor da eleição municipal anterior).
  * `prefeito_long` — tabela long de votação por candidato a prefeito (alvo
    do modelo 4.5).

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
from src.features import target_prefeito as target_pref_mod  # noqa: E402


log = logging.getLogger("02_build_panel")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fase 2 — construir painel mestre + tabela long presidencial e municipal")
    p.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    p.add_argument(
        "--skip-municipal",
        action="store_true",
        help="não construir painel_municipal/prefeito_long (Fase 4.5)",
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

    log.info("Fase 2 (presidencial) OK. painel=%d linhas; pres_long=%d linhas", len(painel), len(pres_long))

    # 4. Eixo municipal (Fase 4.5) — painel_municipal + prefeito_long
    if args.skip_municipal:
        log.info("--skip-municipal: pulando construção do eixo municipal")
        return 0

    try:
        df_cand_pref = fio.load_raw("candidatos_prefeito")
    except FileNotFoundError:
        log.warning(
            "candidatos_prefeito.parquet não encontrado — "
            "rode `python scripts/01_ingest.py --only candidatos_prefeito`. "
            "prefeito_long fica sem nome_candidato."
        )
        df_cand_pref = None

    painel_mun = panel_mod.construir_painel_mestre_municipal(
        diretorio=df_diretorio,
        df_prefeito=df_pref,
        df_partidos_prefeito=df_partidos_pref,
        anos_municipais=MODE_CFG["anos_municipal"],
    )
    fio.save_interim(painel_mun, "painel_municipal")

    pref_long = target_pref_mod.construir_prefeito_long(df_pref, df_cand_pref)
    fio.save_interim(pref_long, "prefeito_long")

    log.info(
        "Fase 2 (municipal) OK. painel_municipal=%d linhas; prefeito_long=%d linhas",
        len(painel_mun), len(pref_long),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
