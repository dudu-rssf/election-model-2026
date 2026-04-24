#!/usr/bin/env python
"""
scripts/03_features.py — Fase 3: feature engineering.

Consolida features de 5 módulos em uma tabela final
`data/processed/features.parquet` com uma linha por
(ano_presidencial × id_municipio × sigla_partido).

Também gera `reports/top_continuidade_dev.md` para revisão humana — o
briefing pede parada se os top-20 não fizerem sentido político.

Entradas esperadas em `data/raw/` (Fases 1 + 1.5 + 3.5):
    resultados_presidenciais
    resultados_prefeito
    resultados_governador
    resultados_deputado_federal
    partidos_prefeito       (opcional — coligação prefeito; NA p/ 2020 no BD)
    partidos_governador     (opcional — coligação governador)
    diretorio_municipios

Entradas em `data/interim/` (Fase 2):
    painel_mestre
    presidencial_long
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CONFIG, MODE_CFG, PATHS, set_global_seed, summary  # noqa: E402
from src.features import io as fio  # noqa: E402
from src.features import (  # noqa: E402
    continuity,
    historical,
    local_power,
    structural,
    vertical,
)


log = logging.getLogger("03_features")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fase 3 — feature engineering")
    p.add_argument("--log-level", type=str, default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--skip-report-top", action="store_true",
                   help="não escrever reports/top_continuidade_dev.md")
    return p.parse_args()


def configurar_logging(level: str) -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        level=getattr(logging, level),
    )


def carregar_insumos() -> dict:
    """Lê tudo que Fase 3 precisa. Raws + interims já filtrados por dev sampling.

    `partidos_prefeito` e `partidos_governador` são opcionais: se não estiverem
    ingeridos, seguimos com `mayor_coligacao` / `gov_coligacao = NA` e as
    features de alinhamento por coligação saem NA.
    """
    insumos = {
        "painel": fio.load_interim("painel_mestre"),
        "pres_long": fio.load_interim("presidencial_long"),
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


def computar_features(insumos: dict, anos_presidenciais) -> dict:
    """Calcula as 5 famílias de features. Retorna dict de DataFrames."""
    partidos = sorted(
        {str(p) for p in insumos["pres_long"]["sigla_partido"].dropna().unique()}
    )
    log.info("partidos cobertos: %d", len(partidos))

    struct = structural.features_structural(insumos["painel"], insumos["pres_long"])
    log.info("structural: %d linhas", len(struct))

    lp_mun = local_power.features_local_mun_ano(insumos["painel"])
    lp_part = local_power.alinhamento_partido_com_prefeito(insumos["painel"], partidos)
    log.info("local_power: %d linhas (mun) + %d linhas (partido)", len(lp_mun), len(lp_part))

    sucessoes = CONFIG.get("partido_sucessao") or {}
    if sucessoes:
        log.info("partido_sucessao: %d mapeamentos (siglas=%s)",
                 sum(len(m) for m in sucessoes.values()),
                 sorted(sucessoes.keys()))
    hist = historical.features_historical(
        insumos["pres_long"], anos_presidenciais, sucessoes=sucessoes,
    )
    log.info("historical: %d linhas", len(hist))

    cont = continuity.features_continuity(
        insumos["prefeito"], insumos["partidos_prefeito"], anos_presidenciais
    )
    log.info("continuity: %d linhas", len(cont))

    align_gov = vertical.alinhamento_partido_com_governador(
        insumos["painel"],
        insumos["governador"],
        insumos["partidos_governador"],
        partidos,
    )
    share_dep = vertical.share_dep_federal_por_partido(
        insumos["dep_federal"], anos_presidenciais
    )
    log.info("vertical: %d linhas (gov) + %d linhas (dep. federal)",
             len(align_gov), len(share_dep))

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
    """Junta tudo no presidencial_long (1 linha por candidato × mun × ano).

    Estratégia de merge:
      * (ano, mun) covariates → left join sobre o long.
      * (ano, mun, partido) → left join via sigla_partido.
    """
    import pandas as pd  # local para evitar no carregamento inicial
    base = insumos["pres_long"].copy()
    base["id_municipio"] = base["id_municipio"].astype("string")

    # ----- (ano, mun) -----
    for nome, df in [
        ("structural", feats["structural"].drop(columns=["sigla_uf"], errors="ignore")),
        ("local_power_mun", feats["local_power_mun"]),
        ("continuity", feats["continuity"]),
    ]:
        base = base.merge(df, on=["ano_presidencial", "id_municipio"], how="left",
                          suffixes=("", f"_{nome}"))

    # ----- (ano, mun, partido) -----
    for nome, df in [
        ("local_power_partido", feats["local_power_partido"]),
        ("historical", feats["historical"]),
        ("align_gov", feats["align_gov"]),
        ("share_dep_federal", feats["share_dep_federal"]),
    ]:
        base = base.merge(
            df, on=["ano_presidencial", "id_municipio", "sigla_partido"], how="left",
            suffixes=("", f"_{nome}"),
        )

    log.info("features consolidadas: %d linhas × %d colunas", *base.shape)
    return base


def gerar_relatorio_top_continuidade(insumos: dict) -> Path | None:
    """Recomputa histórico completo e gera o markdown."""
    hist = continuity.calcular_historico_continuidade(
        insumos["prefeito"], insumos["partidos_prefeito"]
    )
    caminho = PATHS["reports"] / "top_continuidade_dev.md"
    continuity.salvar_relatorio_top_continuidade(
        hist, insumos["diretorio"], caminho, n=20
    )
    return caminho


def main() -> int:
    args = parse_args()
    configurar_logging(args.log_level)
    set_global_seed()
    log.info(summary())

    insumos = carregar_insumos()
    anos = MODE_CFG["anos_presidencial"]
    feats = computar_features(insumos, anos)
    df_features = consolidar(insumos, feats)
    fio.save_processed(df_features, "features")

    if not args.skip_report_top:
        path = gerar_relatorio_top_continuidade(insumos)
        log.info("Revise o top 20: %s", path)
        log.info(
            "Briefing manda PARAR aqui se os top-20 não fizerem sentido político."
        )

    log.info("Fase 3 OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
