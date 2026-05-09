"""Agregação: município → UF → nacional."""
from src.aggregation.aggregate import (
    SomaUnitariaResult,
    agregar_municipal_para_uf,
    agregar_uf_para_nacional,
    cobertura_agregada,
    verificar_soma_unitaria,
)

__all__ = [
    "agregar_municipal_para_uf",
    "agregar_uf_para_nacional",
    "verificar_soma_unitaria",
    "cobertura_agregada",
    "SomaUnitariaResult",
]
