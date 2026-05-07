#!/usr/bin/env python
"""
scripts/01_ingest.py — Fase 1: ingestão de dados brutos.

Baixa e cacheia em `data/raw/`:
  * resultados presidenciais por candidato/município (1º turno)
  * resultados de prefeito por candidato/município (eleição anterior a cada presidencial)
  * candidatos presidenciais (para coligações)
  * candidatos a prefeito (para coligação estadual)
  * diretório IBGE de municípios
  * geometrias municipais (via geobr)

Valida schemas/totais/IDs e gera `reports/ingestao_validacao_<mode>.md`.

Em modo dev (default), o filtro SQL já restringe UFs e anos; após o download,
aplica amostragem determinística de até `max_municipios` para manter a
estrutura realista.

Flags:
  --force         ignora cache, re-roda tudo
  --skip-geo      pula download de geometrias (útil pra iterar)
  --only <nome>   roda apenas uma tabela (nome de `queries.QUERIES`)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Permite rodar como `python scripts/01_ingest.py` a partir da raiz do projeto
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import MODE, MODE_CFG, PATHS, set_global_seed, summary  # noqa: E402
from src.ingestion import bd_client, geo, queries, sample, validate  # noqa: E402


log = logging.getLogger("01_ingest")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fase 1 — ingestão Base dos Dados + geobr")
    p.add_argument("--force", action="store_true", help="ignora cache Parquet local")
    p.add_argument("--skip-geo", action="store_true", help="pula geometrias (geobr)")
    p.add_argument(
        "--only",
        type=str,
        default=None,
        help=f"apenas uma tabela: {sorted(queries.QUERIES)}",
    )
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


def rodar_queries(force: bool, only: str | None) -> dict:
    """Executa as queries do registry e devolve dict nome->DataFrame."""
    frames: dict = {}
    alvo = {only: queries.QUERIES[only]} if only else queries.QUERIES

    for name, sql_fn in alvo.items():
        sql = sql_fn()
        log.info("→ %s", name)
        df = bd_client.download(name=name, sql=sql, force=force)
        log.info("  %s: %d linhas, %d colunas", name, len(df), df.shape[1])
        frames[name] = df
    return frames


# Tabelas e colunas-chave que NÃO podem ter null pro modelo. Nulls aqui são
# resíduos do BD em anos antigos: id_municipio null = voto de zona eleitoral
# sem município brasileiro (exterior, trânsito); numero_candidato null =
# registro de candidatura inválido. Drop é semanticamente correto.
# partidos_governador é exceção: id_municipio é null por design (cargo estadual).
_CHAVES_OBRIGATORIAS: dict[str, tuple[str, ...]] = {
    "resultados_presidenciais": ("id_municipio",),
    "resultados_prefeito": ("id_municipio",),
    "resultados_governador": ("id_municipio",),
    "resultados_deputado_federal": ("id_municipio",),
    "partidos_prefeito": ("id_municipio",),
    "diretorio_municipios": ("id_municipio",),
    "candidatos_deputado_federal": ("numero_candidato",),
}


def limpar_chaves_nulls(frames: dict) -> dict:
    """Drop linhas com chave null nas tabelas registradas em `_CHAVES_OBRIGATORIAS`.

    Logamos quantas linhas dropamos por (tabela, coluna).
    """
    for name, cols in _CHAVES_OBRIGATORIAS.items():
        df = frames.get(name)
        if df is None:
            continue
        for col in cols:
            if col not in df.columns:
                continue
            n_null = int(df[col].isna().sum())
            if n_null > 0:
                log.warning(
                    "  %s: dropando %d/%d linhas com %s null (%.2f%%)",
                    name, n_null, len(df), col, 100 * n_null / len(df),
                )
                df = df[df[col].notna()].copy()
                frames[name] = df
    return frames


def validar(frames: dict) -> validate.ValidationReport:
    report = validate.ValidationReport()
    if "resultados_presidenciais" in frames:
        validate.validate_resultados_presidenciais(frames["resultados_presidenciais"], report)
    if "resultados_prefeito" in frames:
        validate.validate_resultados_prefeito(frames["resultados_prefeito"], report)
    if "resultados_governador" in frames:
        validate.validate_resultados_governador(frames["resultados_governador"], report)
    if "resultados_deputado_federal" in frames:
        validate.validate_resultados_deputado_federal(frames["resultados_deputado_federal"], report)
    if "candidatos_presidenciais" in frames:
        validate.validate_candidatos(frames["candidatos_presidenciais"], report, "candidatos_presidenciais")
    if "candidatos_prefeito" in frames:
        validate.validate_candidatos(frames["candidatos_prefeito"], report, "candidatos_prefeito")
    if "candidatos_governador" in frames:
        validate.validate_candidatos(frames["candidatos_governador"], report, "candidatos_governador")
    if "candidatos_deputado_federal" in frames:
        validate.validate_candidatos(frames["candidatos_deputado_federal"], report, "candidatos_deputado_federal")
    if "partidos_prefeito" in frames:
        validate.validate_partidos_prefeito(frames["partidos_prefeito"], report)
    if "partidos_governador" in frames:
        validate.validate_partidos_governador(frames["partidos_governador"], report)
    if "diretorio_municipios" in frames:
        validate.validate_diretorio_municipios(frames["diretorio_municipios"], report)
    return report


def salvar_amostra_dev(frames: dict) -> dict:
    """Em dev, aplica amostragem e grava sufixo `.dev.parquet` separado.

    Mantém o Parquet bruto intocado para que o mesmo cache sirva prod
    eventualmente. Scripts seguintes (Fase 2+) leem os `.dev.parquet` em dev.
    """
    if MODE != "dev":
        return frames

    amostrados = sample.apply_dev_sampling(frames)
    raw_dir = PATHS["data_raw"]
    for name, df in amostrados.items():
        out = raw_dir / f"{name}.dev.parquet"
        df.to_parquet(out, index=False)
        log.info("  dev: salvo %s (%d linhas após amostragem)", out.name, len(df))
    return amostrados


def escrever_relatorio(report: validate.ValidationReport) -> Path:
    out = PATHS["reports"] / f"ingestao_validacao_{MODE}.md"
    out.write_text(report.to_markdown(), encoding="utf-8")
    log.info("relatório: %s", out)
    return out


def main() -> int:
    args = parse_args()
    configurar_logging(args.log_level)
    set_global_seed()
    log.info(summary())

    frames = rodar_queries(force=args.force, only=args.only)

    if not args.skip_geo:
        try:
            frames["geometrias"] = geo.download_geometrias(force=args.force)
        except Exception as e:  # geobr costuma ter glitches de rede
            log.warning("geometrias falharam (%s) — siga adiante, Fase 6 tentará de novo", e)

    frames = limpar_chaves_nulls(frames)

    report = validar(frames)
    escrever_relatorio(report)

    salvar_amostra_dev(frames)

    if not report.ok:
        log.error("validação encontrou %d error(s). Veja o relatório.", len(report.errors))
        return 2
    log.info("ingestão OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
