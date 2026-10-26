"""Predictions freeze at kickoff: started games keep the saved pre-kickoff row."""
import pandas as pd

from predict import apply_freeze

NOW = pd.Timestamp("2026-10-25T18:00:00Z")
KO = {"early": pd.Timestamp("2026-10-25T17:00:00Z"), "late": pd.Timestamp("2026-10-25T20:25:00Z")}


def frame(p_early, p_late):
    return pd.DataFrame({"game_id": ["early", "late"], "p_home": [p_early, p_late], "confidence": [0.6, 0.7]})


def test_started_game_keeps_saved_prediction():
    out = apply_freeze(frame(0.9, 0.2), frame(0.55, 0.45), KO, NOW).set_index("game_id")
    assert out.loc["early", "p_home"] == 0.55   # kicked off: frozen
    assert out.loc["late", "p_home"] == 0.2     # not yet: refreshed


def test_no_saved_file_uses_fresh_predictions():
    out = apply_freeze(frame(0.9, 0.2), None, KO, NOW).set_index("game_id")
    assert out.loc["early", "p_home"] == 0.9


def test_nothing_started_changes_nothing():
    early = pd.Timestamp("2026-10-25T12:00:00Z")
    out = apply_freeze(frame(0.9, 0.2), frame(0.55, 0.45), KO, early).set_index("game_id")
    assert list(out["p_home"]) == [0.9, 0.2] or set(out["p_home"]) == {0.9, 0.2}
