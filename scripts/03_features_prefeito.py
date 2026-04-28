#!/usr/bin/env python
"""
scripts/03_features_prefeito.py — Fase 4.5: feature engineering (eixo municipal).

Análogo ao scripts/03_features.py, mas no eixo `ano_municipal`:
saída final é `data/processed/features_prefeito.parquet` com uma linha por
(ano_municipal × id_municipio × sigla_partido) — o long do prefeito enriquecido
com features.

Diferenças vs Fase 3 (presidencial):
  * Eixo: `ano_municipal` em vez de `ano_presidencial`.
  * Painel: `painel_municipal` (gap X-4 — o prefeito vigente no momento da
    próxima eleição municipal foi eleito 4 anos antes).
  * Continuidade: `MUNICIPAL_TO_MUNICIPAL_ANTERIOR` (X-4) em vez de
    `PRESIDENCIAL_TO_MUNICIPAL` (X-2).
  * Governador: vigente apenas (X-2). Não há concorrente — não tem eleição
    estadual no ano municipal.
  * Dep. federal: usa eleição federal mais recente, que coincide com o ano
    presidencial X-2 do ano municipal (mesmo `MUNICIPAL_TO_ESTADUAL_ANTERIOR`).

Entradas em `data/raw/`:
    resultados_prefeito
    resultados_governador
    resultados_deputado_federal
    partidos_prefeito       (opcional — coligação prefeito; NA p/ 2020 no BD)
    partidos_governador     (opcional — coligação governador)
    diretorio_municipios

Entradas em `data/interim/`:
    painel_municipal
    prefeito_long
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CONFIG, MODE_CFG, set_global_seed, summary  # noqa: E402
from src.features import io as fio  # noqa: E402
from src.features import (  # noqa: E402
    continuity,
    historical,
    local_power,
    structural,
    vertical,
)
from src.features.panel import MUNICIPAL_TO_MUNICIPAL_ANTERIOR  # noqa: E402
from src.features.vertical import MUNICIPAL_TO_ESTADUAL_ANTERIOR  # noqa: E402


log = logging.getLogger("03_features_prefeito")

ANO_COL = "ano_municipal"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fase 4.5 — feature engineering (eixo municipal)")
    p.add_argument("--log-level", type=str, default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def configurar_logging(level: str) -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        level=getattr(logging, level),
    )


def carregar_insumos() -> dict:
    """Lê tudo que Fase 4.5 precisa — interims do eixo municipal + raws compartilhados."""
    insumos = {
        "painel": fio.load_interim("painel_municipal"),
        "pref_long": fio.load_interim("prefeito_long"),
        "prefeito": fio.load_raw("resultados_prefeito"),
        "governador": fio.load_raw("resultados_governador"),
        "dep_federal": fio.load_raw("resultados_deputado_federal"),
        "diretorio": fio.load_raw("diretorio_municipios"),
    }
    for nome in ("partidos_prefeito", "partidos_governador"):
        try:
            insumos[nome] = fio.load_raw(nome)
        except FileNotFoundError:
            log.warning(
                "%s.parquet não encontrado — rode "
                "`python scripts/01_ingest.py --only %s`. "
                "Coligação correspondente fica NA.",
                nome, nome,
            )
            insumos[nome] = None
    for k, v in insumos.items():
        n = len(v) if v is not None else 0
        log.info("  %-24s %d linhas", k, n)
    return insumos


def computar_features(insumos: dict, anos_municipais) -> dict:
    """Calcula as 5 famílias de features no eixo municipal."""
    partidos = sorted(
        {str(p) for p in insumos["pref_long"]["sigla_partido"].dropna().unique()}
    )
    log.info("partidos cobertos: %d", len(partidos))

    struct = structural.features_structural(
        insumos["painel"], insumos["pref_long"], ano_col=ANO_COL,
    )
    log.info("structural: %d linhas", len(struct))

    lp_mun = local_power.features_local_mun_ano(
        insumos["painel"],
        ano_col=ANO_COL,
        ano_eleicao_anterior_col="ano_eleicao_municipal_anterior",
    )
    lp_part = local_power.alinhamento_partido_com_prefeito(
        insumos["painel"], partidos, ano_col=ANO_COL,
    )
    log.info("local_power: %d linhas (mun) + %d linhas (partido)", len(lp_mun), len(lp_part))

    sucessoes = CONFIG.get("partido_sucessao") or {}
    if sucessoes:
        log.info(
            "partido_sucessao: %d mapeamentos (siglas=%s)",
            sum(len(m) for m in sucessoes.values()),
            sorted(sucessoes.keys()),
        )
    hist = historical.features_historical(
        insumos["pref_long"], anos_municipais,
        sucessoes=sucessoes, ano_col=ANO_COL,
    )
    log.info("historical: %d linhas", len(hist))

    cont = continuity.features_continuity(
        insumos["prefeito"], insumos["partidos_prefeito"],
        anos_alvo=anos_municipais,
        ano_col=ANO_COL,
        map_ano_para_municipal=MUNICIPAL_TO_MUNICIPAL_ANTERIOR,
    )
    log.info("continuity: %d linhas", len(cont))

    align_gov = vertical.alinhamento_partido_com_governador(
        insumos["painel"],
        insumos["governador"],
        insumos["partidos_governador"],
        partidos,
        ano_col=ANO_COL,
        map_vigente=MUNICIPAL_TO_ESTADUAL_ANTERIOR,
        incluir_concorrente=False,
    )
    share_dep = vertical.share_dep_federal_por_partido(
        insumos["dep_federal"], anos_municipais,
        ano_col=ANO_COL,
        map_ano_para_federal=MUNICIPAL_TO_ESTADUAL_ANTERIOR,
    )
    log.info(
        "vertical: %d linhas (gov, vigente only) + %d linhas (dep. federal)",
        len(align_gov), len(share_dep),
    )

    return {
        "structural": struct,
        "local_power_mun": lp_mun,
        "local_power_partido": lp_part,
        "historical": hist,
        "continuity": cont,
        "align_gov": align_gov,
        "share_dep_federal": share_dep,
    }


def consolidar(insumos: dict, feats: dict) -> "pd.DataFrame":
    """Junta tudo no prefeito_long (1 linha por candidato × mun × ano_municipal)."""
    import pandas as pd  # local para evitar custo no carregamento inicial
    base = insumos["pref_long"].copy()
    base["id_municipio"] = base["id_municipio"].astype("string")

    # ----- (ano, mun) -----
    for nome, df in [
        ("structural", feats["structural"].drop(columns=["sigla_uf"], errors="ignore")),
        ("local_power_mun", feats["local_power_mun"]),
        ("continuity", feats["continuity"]),
    ]:
        base = base.merge(
            df, on=[ANO_COL, "id_municipio"], how="left",
            suffixes=("", f"_{nome}"),
        )

    # ----- (ano, mun, partido) -----
    for nome, df in [
        ("local_power_partido", feats["local_power_partido"]),
        ("historical", feats["historical"]),
        ("align_gov", feats["align_gov"]),
        ("share_dep_federal", feats["share_dep_federal"]),
    ]:
        base = base.merge(
            df, on=[ANO_COL, "id_municipio", "sigla_partido"], how="left",
            suffixes=("", f"_{nome}"),
        )

    log.info("features_prefeito consolidadas: %d linhas × %d colunas", *base.shape)
    return base


def main() -> int:
    args = parse_args()
    configurar_logging(args.log_level)
    set_global_seed()
    log.info(summary())

    insumos = carregar_insumos()
    anos = MODE_CFG["anos_municipal"]
    log.info("eixo=ano_municipal; anos-alvo=%s", anos)

    feats = computar_features(insumos, anos)
    df_features = consolidar(insumos, feats)
    fio.save_processed(df_features, "features_prefeito")

    log.info("Fase 4.5 (features) OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
