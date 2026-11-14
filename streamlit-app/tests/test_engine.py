import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import engine  # noqa: E402


def _df():
    return pd.DataFrame({
        "datetime": pd.to_datetime(["2025-01-01 10:00", "2025-01-03 10:00", "2025-01-05 00:00", "2025-01-06 10:00"]),
        "isin": ["X", "Y", "X", "Y"],
        "action": ["BUY", "BUY", "DIVIDEND", "SELL"],
        "quantity": [10.0, 5.0, 0.0, 5.0],
        "net_amount": [100.0, 50.0, 3.0, 60.0],
    })


def test_summary():
    s = engine.summary(_df())
    assert s["bought"] == 150 and s["sold"] == 60 and s["dividends"] == 3
    assert s["net_invested"] == 90
    assert s["open_positions"] == 1  # Y fully sold, X still held


def test_cumulative_cashflows_is_daily_and_flat_between_events():
    f = engine.cumulative_cashflows(_df())
    assert len(f) == 6  # 1 Jan .. 6 Jan
    assert f.loc["2025-01-02", "net_invested"] == 100
    assert f.loc["2025-01-04", "net_invested"] == 150
    assert f.iloc[-1]["net_invested"] == 90
    assert f.iloc[-1]["dividends_cum"] == 3