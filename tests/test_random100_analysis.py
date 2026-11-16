"""Counts from saved reaches, including multiple violations and missing data."""
import json
from types import SimpleNamespace as NS

import numpy as np

from experiments.random100 import analyze


def record(status, q):
    return NS(metadata={"status": status}, ticks={"q": np.asarray(q)})


def test_counts_reaches_not_samples_or_joints_and_keeps_arrival_separate():
    q = np.zeros((3, 7))
    q[:2, 1] = -1.00002
    q[2, 5] = 1.00003
    records = [record("arrived", q), record("timeout", np.zeros((1, 7))),
               record("arrived", np.full((1, 7), 1.000005))]
    counts = analyze.summarize(NS(cases=[None]*3), records, -np.ones(7), np.ones(7))
    assert counts == dict(cases=3, arrived=2, joint_limit_violations=1, checked=3, unchecked=0)


def test_missing_empty_and_nonfinite_positions_are_not_clean_passes():
    records = [record("error", []), record("fault", [[np.nan]*7]),
               record("interrupted", [[1.1]*7]), record("error", [[0.]*6])]
    counts = analyze.summarize(NS(cases=[None]*5), records, -np.ones(7), np.ones(7))
    assert counts == dict(cases=5, arrived=0, joint_limit_violations=1, checked=1, unchecked=4)


def test_main_prints_totals_and_saves_json(tmp_path, monkeypatch, capsys):
    study = NS(cases=[None]*2)
    records = [record("arrived", np.zeros((1, 7)))]
    monkeypatch.setattr(analyze, "load", lambda path: (study, records))
    monkeypatch.setattr(analyze.tm, "limits", lambda: NS(lower=-np.ones(7), upper=np.ones(7)))
    assert analyze.main([str(tmp_path)]) == 0
    assert json.loads((tmp_path / "analysis.json").read_text()) == dict(
        cases=2, arrived=1, joint_limit_violations=0, checked=1, unchecked=1)
    output = capsys.readouterr().out
    assert "Arrived: 1/2" in output
    assert "Joint-position limit violations: 0/1 checked reaches" in output
    assert "Unchecked reaches (missing or invalid positions): 1" in output
