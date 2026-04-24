"""
src.ingestion.validate — sanity checks dos DataFrames ingeridos.

Objetivos:
  * Schema mínimo presente (colunas obrigatórias).
  * Nulls proibidos em chaves.
  * Totais de voto batendo com referência externa (tolerância 0.1%), quando
    fornecida (ex.: totais oficiais de presidencial por UF de https://dadosabertos.tse.jus.br).
  * Votos ≥ 0.
  * IDs IBGE de 7 dígitos.

Reporte produzido em `reports/ingestao_validacao_dev.md` pelo script.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd


TOLERANCIA_VOTOS = 0.001  # 0.1%


@dataclass
class ValidationIssue:
    tabela: str
    severidade: str  # "error" | "warning"
    mensagem: str


@dataclass
class ValidationReport:
    issues: list[ValidationIssue] = field(default_factory=list)

    def add(self, tabela: str, severidade: str, mensagem: str) -> None:
        self.issues.append(ValidationIssue(tabela, severidade, mensagem))

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severidade == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severidade == "warning"]

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0

    def to_markdown(self) -> str:
        lines = ["# Relatório de validação da ingestão", ""]
        if self.ok and not self.warnings:
            lines.append("Todas as validações passaram.")
            return "\n".join(lines)

        if self.errors:
            lines.append("## Errors")
            for i in self.errors:
                lines.append(f"- **{i.tabela}** — {i.mensagem}")
            lines.append("")
        if self.warnings:
            lines.append("## Warnings")
            for i in self.warnings:
                lines.append(f"- **{i.tabela}** — {i.mensagem}")
            lines.append("")
        return "\n".join(lines)


# ------------------------------------------------------------
# Checagens atômicas
# ------------------------------------------------------------
def check_columns(
    df: pd.DataFrame,
    required: Iterable[str],
    tabela: str,
    report: ValidationReport,
) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        report.add(tabela, "error", f"colunas faltando: {missing}")


def check_no_nulls(
    df: pd.DataFrame,
    columns: Iterable[str],
    tabela: str,
    report: ValidationReport,
) -> None:
    for col in columns:
        if col not in df.columns:
            continue
        n_null = int(df[col].isna().sum())
        if n_null > 0:
            report.add(tabela, "error", f"coluna {col!r} tem {n_null} null(s); proibido")


def check_non_negative(
    df: pd.DataFrame,
    columns: Iterable[str],
    tabela: str,
    report: ValidationReport,
) -> None:
    for col in columns:
        if col not in df.columns:
            continue
        n_neg = int((df[col] < 0).sum())
        if n_neg > 0:
            report.add(tabela, "error", f"coluna {col!r} tem {n_neg} valor(es) < 0")


def check_id_municipio_7_digitos(
    df: pd.DataFrame,
    col: str,
    tabela: str,
    report: ValidationReport,
) -> None:
    if col not in df.columns:
        return
    serie = df[col].astype("string").str.strip()
    bad = (~serie.str.match(r"^\d{7}$")).sum()
    if bad > 0:
        report.add(tabela, "error", f"{col}: {bad} valor(es) não são IBGE de 7 dígitos")


def check_totais_vs_oficial(
    df_votos: pd.DataFrame,
    oficial_por_uf: dict[tuple[int, str], int] | None,
    tabela: str,
    report: ValidationReport,
    tolerancia: float = TOLERANCIA_VOTOS,
) -> None:
    """Valida totais de voto por (ano, UF) contra referência oficial.

    Args:
        df_votos: DataFrame com colunas ano, sigla_uf, votos.
        oficial_por_uf: dict {(ano, uf): total_oficial}; se None, pula check.
        tolerancia: fração máxima de divergência aceita.
    """
    if not oficial_por_uf:
        report.add(tabela, "warning", "totais oficiais não fornecidos — check pulado")
        return

    agg = df_votos.groupby(["ano", "sigla_uf"], as_index=False)["votos"].sum()
    for _, row in agg.iterrows():
        chave = (int(row["ano"]), str(row["sigla_uf"]))
        if chave not in oficial_por_uf:
            continue
        oficial = oficial_por_uf[chave]
        if oficial <= 0:
            report.add(
                tabela, "warning", f"total oficial <= 0 para {chave}, pulado"
            )
            continue
        diff = abs(int(row["votos"]) - oficial) / oficial
        if diff > tolerancia:
            report.add(
                tabela,
                "error",
                f"total {chave}: {int(row['votos']):,} vs oficial {oficial:,} "
                f"(diff {diff:.2%} > {tolerancia:.2%})",
            )


# ------------------------------------------------------------
# Validações por tabela
# ------------------------------------------------------------
def validate_resultados_presidenciais(
    df: pd.DataFrame,
    report: ValidationReport,
    oficial_por_uf: dict[tuple[int, str], int] | None = None,
) -> None:
    tabela = "resultados_presidenciais"
    required = [
        "ano",
        "sigla_uf",
        "id_municipio",
        "cargo",
        "numero_candidato",
        "turno",
        "votos",
    ]
    check_columns(df, required, tabela, report)
    check_no_nulls(df, ["ano", "sigla_uf", "id_municipio", "numero_candidato", "votos"], tabela, report)
    check_non_negative(df, ["votos"], tabela, report)
    check_id_municipio_7_digitos(df, "id_municipio", tabela, report)
    check_totais_vs_oficial(df, oficial_por_uf, tabela, report)


def validate_resultados_prefeito(df: pd.DataFrame, report: ValidationReport) -> None:
    tabela = "resultados_prefeito"
    required = ["ano", "sigla_uf", "id_municipio", "cargo", "sigla_partido", "votos"]
    check_columns(df, required, tabela, report)
    check_no_nulls(df, ["ano", "sigla_uf", "id_municipio"], tabela, report)
    check_non_negative(df, ["votos"], tabela, report)
    check_id_municipio_7_digitos(df, "id_municipio", tabela, report)


def validate_resultados_governador(df: pd.DataFrame, report: ValidationReport) -> None:
    tabela = "resultados_governador"
    required = ["ano", "sigla_uf", "id_municipio", "cargo", "sigla_partido", "votos"]
    check_columns(df, required, tabela, report)
    check_no_nulls(df, ["ano", "sigla_uf", "id_municipio"], tabela, report)
    check_non_negative(df, ["votos"], tabela, report)
    check_id_municipio_7_digitos(df, "id_municipio", tabela, report)


def validate_resultados_deputado_federal(df: pd.DataFrame, report: ValidationReport) -> None:
    tabela = "resultados_deputado_federal"
    required = ["ano", "sigla_uf", "id_municipio", "cargo", "sigla_partido", "votos"]
    check_columns(df, required, tabela, report)
    check_no_nulls(df, ["ano", "sigla_uf", "id_municipio"], tabela, report)
    check_non_negative(df, ["votos"], tabela, report)
    check_id_municipio_7_digitos(df, "id_municipio", tabela, report)


def validate_candidatos(df: pd.DataFrame, report: ValidationReport, tabela: str) -> None:
    required = ["ano", "numero_candidato", "nome_candidato", "sigla_partido", "cargo"]
    check_columns(df, required, tabela, report)
    check_no_nulls(df, ["ano", "numero_candidato", "sigla_partido"], tabela, report)


def validate_partidos_prefeito(df: pd.DataFrame, report: ValidationReport) -> None:
    """Valida `partidos_prefeito` (br_tse_eleicoes.partidos / cargo='prefeito').

    `composicao_coligacao` PODE vir 100% null (ano 2020 — buraco da BD); só
    avisamos. Chaves (ano, sigla_uf, id_municipio, sigla_partido) não podem
    ser null.
    """
    tabela = "partidos_prefeito"
    required = ["ano", "sigla_uf", "id_municipio", "sigla_partido", "composicao_coligacao"]
    check_columns(df, required, tabela, report)
    check_no_nulls(df, ["ano", "sigla_uf", "id_municipio", "sigla_partido"], tabela, report)
    check_id_municipio_7_digitos(df, "id_municipio", tabela, report)

    if "composicao_coligacao" in df.columns and "ano" in df.columns:
        cobertura = (
            df.assign(_ok=df["composicao_coligacao"].notna())
            .groupby("ano")["_ok"]
            .mean()
        )
        for ano, pct in cobertura.items():
            if pct < 0.5:
                report.add(
                    tabela,
                    "warning",
                    f"ano {int(ano)}: composicao_coligacao populada em {pct:.1%} "
                    "das linhas (<50%); features de coligação ficam degradadas neste ano",
                )


def validate_partidos_governador(df: pd.DataFrame, report: ValidationReport) -> None:
    """Valida `partidos_governador`. id_municipio é NULL por design (cargo estadual)."""
    tabela = "partidos_governador"
    required = ["ano", "sigla_uf", "sigla_partido", "composicao_coligacao"]
    check_columns(df, required, tabela, report)
    check_no_nulls(df, ["ano", "sigla_uf", "sigla_partido"], tabela, report)

    if "composicao_coligacao" in df.columns and "ano" in df.columns:
        cobertura = (
            df.assign(_ok=df["composicao_coligacao"].notna())
            .groupby("ano")["_ok"]
            .mean()
        )
        for ano, pct in cobertura.items():
            if pct < 0.5:
                report.add(
                    tabela,
                    "warning",
                    f"ano {int(ano)}: composicao_coligacao populada em {pct:.1%}; "
                    "features de alinhamento gov-coligação ficam degradadas",
                )


def validate_diretorio_municipios(df: pd.DataFrame, report: ValidationReport) -> None:
    tabela = "diretorio_municipios"
    required = ["id_municipio", "sigla_uf", "nome"]
    check_columns(df, required, tabela, report)
    check_no_nulls(df, ["id_municipio", "sigla_uf", "nome"], tabela, report)
    check_id_municipio_7_digitos(df, "id_municipio", tabela, report)


__all__ = [
    "TOLERANCIA_VOTOS",
    "ValidationIssue",
    "ValidationReport",
    "check_columns",
    "check_no_nulls",
    "check_non_negative",
    "check_id_municipio_7_digitos",
    "check_totais_vs_oficial",
    "validate_resultados_presidenciais",
    "validate_resultados_prefeito",
    "validate_resultados_governador",
    "validate_resultados_deputado_federal",
    "validate_candidatos",
    "validate_partidos_prefeito",
    "validate_partidos_governador",
    "validate_diretorio_municipios",
]
