import json
import shutil
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pytest

import evaluate


result_dict = {"metric": 1.0, "model_name": "x"}

SAVE_EXTRA_KEYS = ["_timestamp", "_git_commit_hash", "_evaluate_version", "_python_version", "_interpreter_path"]


class TestSave(TestCase):
    def setUp(self):
        self.save_path = Path(tempfile.mkdtemp())
        git_commit = patch("evaluate.saving._git_commit_hash", return_value="test-commit")
        git_commit.start()
        self.addCleanup(git_commit.stop)

    def tearDown(self):
        shutil.rmtree(self.save_path)

    def test_save_to_folder(self):
        file_path = evaluate.save(self.save_path, **result_dict)
        with open(file_path, "r") as f:
            loaded_result_dict = json.load(f)
        for key in SAVE_EXTRA_KEYS:
            _ = loaded_result_dict.pop(key)
        self.assertDictEqual(result_dict, loaded_result_dict)

    def test_save_to_folder_nested(self):
        file_path = evaluate.save(self.save_path / "sub_dir1/sub_dir2", **result_dict)
        with open(file_path, "r") as f:
            loaded_result_dict = json.load(f)
        for key in SAVE_EXTRA_KEYS:
            _ = loaded_result_dict.pop(key)
        self.assertDictEqual(result_dict, loaded_result_dict)

    def test_save_to_file(self):
        _ = evaluate.save(self.save_path / "test.json", **result_dict)
        with open(self.save_path / "test.json", "r") as f:
            loaded_result_dict = json.load(f)
        for key in SAVE_EXTRA_KEYS:
            _ = loaded_result_dict.pop(key)
        self.assertDictEqual(result_dict, loaded_result_dict)


@pytest.fixture
def save_numpy_result(tmp_path, monkeypatch):
    monkeypatch.setattr("evaluate.saving._git_commit_hash", lambda: "test-commit")

    def save_result(value):
        path = evaluate.save(tmp_path / "numpy.json", result=value)
        assert isinstance(path, Path)
        with path.open() as f:
            result = json.load(f)
        assert result["_git_commit_hash"] == "test-commit"
        assert all(key in result for key in SAVE_EXTRA_KEYS)
        return result["result"]

    return save_result


@pytest.mark.parametrize(
    "value, expected",
    [
        (np.array([[2, 1], [0, 1]], dtype=np.int64), [[2, 1], [0, 1]]),
        (np.array([0.25, 0.75], dtype=np.float32), [0.25, 0.75]),
        (np.array([0.5], dtype=np.longdouble), [0.5]),
        (np.array([2**64 - 1], dtype=np.uint64), [2**64 - 1]),
        (np.array([True, False]), [True, False]),
        (np.array(["cat", "dog"]), ["cat", "dog"]),
        (np.array(3), 3),
        (np.array(0.5, dtype=np.float32), 0.5),
        (np.array(None, dtype=object), None),
        (np.array([[[1]]]), [[[1]]]),
        (np.array([], dtype=np.float64), []),
        (np.empty((2, 0), dtype=np.int64), [[], []]),
        (np.arange(6).reshape(2, 3).T, [[0, 3], [1, 4], [2, 5]]),
        (np.arange(6)[::2], [0, 2, 4]),
    ],
)
def test_save_numpy_arrays(save_numpy_result, value, expected):
    assert save_numpy_result(value) == expected


@pytest.mark.parametrize(
    "scalar_type, value",
    [(np.bool_, True)]
    + [(dtype, 3) for dtype in (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)]
    + [(dtype, 0.5) for dtype in (np.float16, np.float32, np.float64, np.longdouble)]
    + [(np.str_, "label")],
)
def test_save_numpy_scalars(save_numpy_result, scalar_type, value):
    saved = save_numpy_result(scalar_type(value))
    assert saved == value
    assert type(saved) is type(value)


def test_save_nested_numpy_results(save_numpy_result):
    matrix = np.array([[2, 1], [0, 1]])
    result = {
        "metrics": {"matrix": matrix, "accuracy": np.float32(0.75)},
        "runs": [{"count": np.int64(4)}, (np.bool_(True), np.array([0.5]))],
        "objects": np.array([{"score": np.float32(0.5)}, np.array([1, 2]), None], dtype=object),
    }
    assert save_numpy_result(result) == {
        "metrics": {"matrix": [[2, 1], [0, 1]], "accuracy": 0.75},
        "runs": [{"count": 4}, [True, [0.5]]],
        "objects": [{"score": 0.5}, [1, 2], None],
    }
    np.testing.assert_array_equal(matrix, [[2, 1], [0, 1]])
    assert isinstance(result["metrics"]["accuracy"], np.float32)


@pytest.mark.parametrize(
    "value",
    [
        object(),
        {1, 2},
        1 + 2j,
        np.complex64(1 + 2j),
        np.array([1 + 2j]),
        np.array([], dtype=complex),
        np.datetime64("2026-11-12"),
        np.array(["2026-11-12"], dtype="datetime64[D]"),
        np.timedelta64(1, "D"),
        np.array([1], dtype="timedelta64[D]"),
        np.bytes_(b"label"),
        np.array([b"label"]),
        np.array([(1,)], dtype=[("count", "i4")]),
        np.array([object()], dtype=object),
        {"nested": [np.array([1]), object()]},
    ],
)
def test_save_rejects_unsupported_values(save_numpy_result, value):
    with pytest.raises(TypeError, match="not JSON serializable"):
        save_numpy_result(value)


def test_save_python_values_unchanged(save_numpy_result):
    value = {"text": "label", "count": 3, "score": 0.5, "ok": True, "missing": None, "items": [1, "x"]}
    assert save_numpy_result(value) == value


def test_save_nonfinite_values_unchanged(save_numpy_result):
    values = save_numpy_result([float("nan"), float("inf"), np.float32("nan"), np.float32("-inf")])
    assert np.isnan(values[0]) and np.isnan(values[2])
    assert values[1] == float("inf") and values[3] == float("-inf")


def test_save_numpy_keys_remain_unsupported(save_numpy_result):
    with pytest.raises(TypeError, match="keys must be"):
        save_numpy_result({np.int64(1): "value"})
