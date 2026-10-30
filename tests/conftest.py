"""Fixtures compartidas.

La bandeja sintética reproduce en miniatura la estructura real, incluido el
incidente de corrección retroactiva. Las pruebas no tocan `landing/` del
proyecto: si dependieran de los datos de Kaggle no podrían correr en el CI,
donde esos CSV no existen porque no se versionan en Git.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

FAMILIAS = ["GROCERY I", "PRODUCE"]


def _ventas(fecha: str, tiendas=(1, 44), ventas: float | None = None) -> pd.DataFrame:
    """Un día de ventas para unas tiendas. `ventas=0.0` simula el POS caído."""
    filas = [
        {
            "date": fecha,
            "store_nbr": tienda,
            "family": familia,
            "sales": ventas if ventas is not None else 10.0 * tienda,
            "onpromotion": 0,
        }
        for tienda in tiendas
        for familia in FAMILIAS
    ]
    return pd.DataFrame(filas)


def _escribir(df: pd.DataFrame, destino: Path) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destino, index=False, date_format="%Y-%m-%d")


@pytest.fixture
def bandeja(tmp_path: Path) -> Path:
    """Bandeja mínima con el incidente de corrección retroactiva.

    Calendario:
      histórico           -> hasta 2017-06-30
      llega el 2017-07-06 -> ventas del 2017-07-05, tienda 44 EN CEROS
      llega el 2017-07-07 -> ventas del 2017-07-06, normales
      llega el 2017-07-20 -> ventas del 2017-07-05 CORREGIDAS
    """
    landing = tmp_path / "landing"

    historico = pd.concat([_ventas("2017-06-29"), _ventas("2017-06-30")], ignore_index=True)
    _escribir(historico, landing / "historico" / "sales.csv")

    # Entrega original degradada: el POS de la tienda 44 reportó ceros.
    original = pd.concat(
        [_ventas("2017-07-05", tiendas=(1,)), _ventas("2017-07-05", tiendas=(44,), ventas=0.0)],
        ignore_index=True,
    )
    _escribir(original, landing / "arrival_date=2017-07-06" / "sales" / "sales_2017-07-05.csv")

    _escribir(
        _ventas("2017-07-06"),
        landing / "arrival_date=2017-07-07" / "sales" / "sales_2017-07-06.csv",
    )

    # La corrección: mismo archivo, llegada posterior, valores verdaderos.
    _escribir(
        _ventas("2017-07-05"),
        landing / "arrival_date=2017-07-20" / "sales" / "sales_2017-07-05.csv",
    )

    return landing
