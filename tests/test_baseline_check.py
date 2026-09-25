import copy
import json

import pytest

from src.retrieving.baseline_check import main

BASELINE = {
    "total_queries": 38,
    "reproducibility_fingerprint": {
        "dataset_md5": "abc",
        "index_count": 2284,
        "configuration": {"top_k": 5, "use_hybrid": False, "use_sparse": False, "use_reranker": False},
    },
    "overall_metrics": {"recall_at_1": 0.974, "recall_at_3": 1.0, "recall_at_5": 1.0, "mrr": 0.98},
}


def run(tmp_path, candidate, *extra):
    base_file, cand_file = tmp_path / "base.json", tmp_path / "cand.json"
    base_file.write_text(json.dumps(BASELINE))
    cand_file.write_text(json.dumps(candidate))
    return main([str(base_file), str(cand_file), *extra])


def variant(**overrides):
    candidate = copy.deepcopy(BASELINE)
    for path, value in overrides.items():
        target = candidate
        *parents, leaf = path.split("__")
        for parent in parents:
            target = target[parent]
        target[leaf] = value
    return candidate


def test_identical_run_passes(tmp_path):
    assert run(tmp_path, BASELINE) == 0


def test_improvement_passes(tmp_path):
    assert run(tmp_path, variant(overall_metrics__recall_at_1=1.0)) == 0


def test_any_drop_fails_by_default(tmp_path, capsys):
    assert run(tmp_path, variant(overall_metrics__recall_at_5=0.974)) == 1
    assert "recall_at_5" in capsys.readouterr().out


def test_drop_within_the_allowance_passes(tmp_path):
    assert run(tmp_path, variant(overall_metrics__recall_at_5=0.974), "--max-drop", "0.03") == 0


@pytest.mark.parametrize(
    "candidate",
    [
        variant(reproducibility_fingerprint__dataset_md5="different"),
        variant(total_queries=6),
    ],
)
def test_runs_on_different_data_are_not_comparable(tmp_path, candidate):
    assert run(tmp_path, candidate) == 2


def test_runs_against_a_different_index_size_are_not_comparable(tmp_path, capsys):
    assert run(tmp_path, variant(reproducibility_fingerprint__index_count=3000)) == 2
    assert "refresh the baseline" in capsys.readouterr().out


def test_runs_with_different_configuration_are_not_comparable(tmp_path):
    candidate = copy.deepcopy(BASELINE)
    candidate["reproducibility_fingerprint"]["configuration"]["use_reranker"] = True
    assert run(tmp_path, candidate) == 2
