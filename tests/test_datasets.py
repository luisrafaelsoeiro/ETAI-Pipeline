"""
Week 9: the pipeline must run on datasets other than COMPAS, and stop early -- with a clear
message -- on a config that doesn't fit its data. Each test builds a small synthetic dataset
(nothing from data/ is changed), writes a config for it in a temporary folder and runs the
whole pipeline (`main.run`) or the config checks on it.

Run from the project root:
    python tests/test_datasets.py          (or: python -m pytest tests, if pytest is installed)

Covered:
  1. yes/no target with TEXT labels, a semicolon-separated file, placeholders and spelling
     variants, a numeric column stored as text, rows without a target, repeated ids, a column
     listed nowhere, new data without an id column;
  2. a THREE-class target, with XGBoost (needs 0..k-1 labels), tuning + nested CV + feature
     selection, multiclass metrics, the locked-test evaluation and per-class probabilities;
  3. numeric features only, no sensitive attribute, a tab-separated file, a 0/1 target;
  4. configs that must be rejected before anything is fitted (wrong names, overlaps, binary-only
     metrics on three classes, a continuous target, text in a numeric feature, ...);
  5. the project's own configs (COMPAS, Adult) pass every check.
"""
import os
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

import main  # noqa: E402
from src.config import ConfigError, check_dataset, load_config  # noqa: E402
from src.data import load_data  # noqa: E402
from src.preprocessing import clean_dataset, encode_target  # noqa: E402

TOKENS = ["?", "n/a", "", "nan"]


def _base_config(tmp: Path, data_path: Path, target: str, numeric: list, categorical: list, **overrides) -> dict:
    config = {
        "data": {"path": str(data_path), "target": target, "drop_columns": ["id"]},
        "test_set": {"size": 0.2, "random_state": 0},
        "cv": {"n_splits": 3, "shuffle": True, "random_state": 0,
               "metrics": ["accuracy", "f1", "roc_auc"], "selection_metric": "roc_auc"},
        "diagnostics": {"id_column": "id", "placeholder_tokens": TOKENS},
        "preprocessing": {"encoder": "onehot", "scaler": "standard",
                          "numeric_features": numeric, "categorical_features": categorical, "random_state": 0},
        "model": {"type": "logistic_regression", "params": {"max_iter": 500}},
        "output": {"results_dir": str(tmp / "results"), "scoreboard": str(tmp / "results" / "runs.csv")},
    }
    for key, value in overrides.items():
        section, name = key.split("__")
        config.setdefault(section, {})[name] = value
    return config


def _write_config(tmp: Path, config: dict, name: str = "config_test.yaml") -> str:
    path = tmp / name
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return str(path)


def _binary_text_data(n: int, rng, with_target: bool = True, with_id: bool = True) -> pd.DataFrame:
    x1 = rng.normal(size=n)
    x2 = rng.integers(0, 50, size=n)
    color = rng.choice(["red", "Red ", "RED", "blue", "green", "?"], size=n)
    shape = rng.choice(["circle", "square", "n/a"], size=n)
    score = x1 + (np.char.lower(np.char.strip(color.astype(str))) == "red") + rng.normal(scale=0.5, size=n)
    df = pd.DataFrame({
        "x1_text": np.where(rng.random(n) < 0.05, "?", np.round(x1, 3).astype(str)),   # numeric, stored as text
        "x2": x2, "color": color, "shape": shape,
        "group": rng.choice(["A", "B"], size=n),
        "unused_note": rng.choice(["foo", "bar"], size=n),                                # listed nowhere
    })
    if with_id:
        df.insert(0, "id", np.arange(n))
    if with_target:
        df["label"] = np.where(score > 0.3, "yes", "no").astype(object)
        df.loc[df.sample(5, random_state=1).index, "label"] = np.nan                    # no target
    return df


def test_binary_text_labels_semicolon_file_and_new_data_without_id():
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as folder:
        tmp = Path(folder)
        train = _binary_text_data(400, rng)
        train = pd.concat([train, train.head(5)], ignore_index=True)                    # repeated ids
        train.to_csv(tmp / "train.csv", sep=";", index=False)
        _binary_text_data(60, rng, with_target=False, with_id=False).to_csv(tmp / "new.csv", sep=";", index=False)

        config = _base_config(
            tmp, tmp / "train.csv", "label", ["x1_text", "x2"], ["color", "shape"],
            data__sensitive_attr="group", data__read_options={"sep": ";"},
            diagnostics__numeric_text_columns=["x1_text"],
            diagnostics__canonical_categories={"color": {"red": "red", "blue": "blue", "green": "green"}},
            preprocessing__mnar_indicator_sources=["color"],
            feature_selection__enabled=True, feature_selection__min_votes=1,
            predict__path=str(tmp / "new.csv"),
        )
        config["predict"]["output"] = str(tmp / "predictions.csv")
        result = main.run(_write_config(tmp, config))

        assert result["classes"] == ["no", "yes"], result["classes"]
        predictions = pd.read_csv(result["predictions"])
        assert len(predictions) == 60 and list(predictions.columns) == ["row", "probability", "prediction"]
        assert set(predictions["prediction"]) <= {"yes", "no"}, "predictions come back in the data's own labels"
        report = Path(result["report"]).read_text(encoding="utf-8")
        assert "unused_note" in report, "a column listed nowhere is reported"
        assert "5 training rows have no target" in report
        assert "true no" in report and "False positive rate by group" in report
        assert len(pd.read_csv(config["output"]["scoreboard"])) == 1


def _three_class_data(n: int, rng, with_target: bool = True) -> pd.DataFrame:
    a, b = rng.normal(size=n), rng.normal(size=n)
    region = rng.choice(["north", "south", "east"], size=n)
    score = a + 0.5 * b + (region == "north") + rng.normal(scale=0.4, size=n)
    df = pd.DataFrame({"id": np.arange(n), "a": a, "b": b, "region": region, "sex": rng.choice(["F", "M"], size=n)})
    if with_target:
        df["tier"] = pd.cut(score, [-np.inf, -0.3, 0.8, np.inf], labels=["Low", "Medium", "High"]).astype(str)
    return df


def test_three_classes_with_xgboost_tuning_selection_and_locked_test():
    rng = np.random.default_rng(1)
    with tempfile.TemporaryDirectory() as folder:
        tmp = Path(folder)
        _three_class_data(450, rng).to_csv(tmp / "train.csv", index=False)
        _three_class_data(40, rng, with_target=False).to_csv(tmp / "new.csv", index=False)

        config = _base_config(
            tmp, tmp / "train.csv", "tier", ["a", "b"], ["region"],
            data__sensitive_attr="sex",
            cv__metrics=["accuracy", "f1_macro", "roc_auc_ovr", "fbeta_2_macro"], cv__selection_metric="roc_auc_ovr",
            preprocessing__encoder="target",
            model__type="xgboost", model__params={"n_estimators": 30, "max_depth": 2, "random_state": 0, "n_jobs": 1},
            feature_selection__enabled=True,
            feature_selection__search_space={"select__min_votes": {"type": "int", "low": 1, "high": 2}},
            tuning__enabled=True, tuning__n_trials=2, tuning__n_splits=2, tuning__random_state=0,
            tuning__search_spaces={"xgboost": {"model__max_depth": {"type": "int", "low": 1, "high": 3}}},
            test_set__evaluate=True,
            predict__path=str(tmp / "new.csv"),
        )
        config["predict"]["output"] = str(tmp / "predictions.csv")
        result = main.run(_write_config(tmp, config))

        assert result["classes"] == ["High", "Low", "Medium"], result["classes"]
        assert 0.5 < result["scores"]["roc_auc_ovr"].mean() <= 1
        predictions = pd.read_csv(result["predictions"])
        assert len(predictions) == 40
        assert {"probability_High", "probability_Low", "probability_Medium"} <= set(predictions.columns)
        assert np.allclose(predictions.filter(like="probability_").sum(axis=1), 1)
        assert set(predictions["prediction"]) <= {"High", "Low", "Medium"}
        report = Path(result["report"]).read_text(encoding="utf-8")
        assert "Locked test set" in report and "roc_auc_ovr" in report
        assert "False positive rate per class by sex" in report


def test_numeric_only_no_sensitive_attribute_tab_separated():
    rng = np.random.default_rng(2)
    with tempfile.TemporaryDirectory() as folder:
        tmp = Path(folder)
        n = 300
        x = rng.normal(size=(n, 3))
        df = pd.DataFrame(x, columns=["f1", "f2", "f3"])
        df["outcome"] = (x[:, 0] - x[:, 1] + rng.normal(scale=0.5, size=n) > 0).astype(int)
        df.to_csv(tmp / "train.tsv", sep="\t", index=False)

        config = _base_config(tmp, tmp / "train.tsv", "outcome", ["f1", "f2", "f3"], [],
                              data__drop_columns=[], diagnostics__id_column=None,
                              model__type="random_forest", model__params={"n_estimators": 30, "random_state": 0},
                              test_set__evaluate=True)
        result = main.run(_write_config(tmp, config))
        assert result["classes"] == [0, 1]
        report = Path(result["report"]).read_text(encoding="utf-8")
        assert "Fairness report: skipped" in report and "Locked test set" in report


def _expect_config_error(config: dict, tmp: Path, *fragments) -> None:
    try:
        main.run(_write_config(tmp, config, "config_bad.yaml"))
    except ConfigError as error:
        message = str(error)
        for fragment in fragments:
            assert fragment in message, f"expected '{fragment}' in:\n{message}"
        return
    raise AssertionError(f"no ConfigError; expected one mentioning {fragments}")


def test_configs_that_must_be_rejected_before_fitting():
    rng = np.random.default_rng(3)
    with tempfile.TemporaryDirectory() as folder:
        tmp = Path(folder)
        _three_class_data(200, rng).to_csv(tmp / "three.csv", index=False)
        binary = _binary_text_data(200, rng)
        binary.to_csv(tmp / "binary.csv", index=False)

        def config(**overrides):
            return _base_config(tmp, tmp / "binary.csv", "label", ["x2"], ["color"], **overrides)

        # names and lists, checked without the data
        bad = config(model__type="xgboosted", preprocessing__encoder="one_hot")
        _expect_config_error(bad, tmp, "unknown `model.type` 'xgboosted'", "unknown `preprocessing.encoder` 'one_hot'")
        bad = config(preprocessing__categorical_features=["color", "x2"], cv__selection_metric="auc")
        _expect_config_error(bad, tmp, "listed twice", "must be one of `cv.metrics`")
        bad = config(preprocessing__numeric_features=["x2", "label"])
        _expect_config_error(bad, tmp, "`label` is listed as a feature but is the target")
        bad = config(tuning__enabled=True,
                     tuning__search_spaces={"logistic_regression": {"model__CC": {"type": "float", "low": 0.1, "high": 1}}})
        _expect_config_error(bad, tmp, "`model__CC` doesn't exist in the pipeline", "model__C")
        bad = config(tuning__enabled=True, tuning__search_spaces={})
        _expect_config_error(bad, tmp, "no search space for `logistic_regression`")

        # the config against the data
        bad = config(preprocessing__numeric_features=["x2", "x1_text"])
        _expect_config_error(bad, tmp, "numeric features that hold text", "numeric_text_columns")
        bad = config(preprocessing__categorical_features=["colour"])
        _expect_config_error(bad, tmp, "lists columns that aren't in the data: ['colour']")
        bad = _base_config(tmp, tmp / "three.csv", "tier", ["a", "b"], ["region"])     # metrics f1, roc_auc
        _expect_config_error(bad, tmp, "only exist for two classes", "f1_macro")
        bad = _base_config(tmp, tmp / "three.csv", "a", ["b"], ["region"])             # continuous target
        _expect_config_error(bad, tmp, "looks continuous")
        rare = _three_class_data(200, rng)
        rare.loc[:1, "tier"] = "Extreme"                                                # a class with 2 rows
        rare.to_csv(tmp / "rare.csv", index=False)
        bad = _base_config(tmp, tmp / "rare.csv", "tier", ["a"], ["region"],
                           cv__metrics=["accuracy"], cv__selection_metric="accuracy")
        _expect_config_error(bad, tmp, "class 'Extreme'", "too few")
        bad = config(data__sensitive_attr="gender")
        _expect_config_error(bad, tmp, "`data.sensitive_attr` `gender` not found")


def test_target_labels_and_positive_class():
    y = pd.Series(["no", "yes", "yes", None, "no"]).dropna()
    encoded, classes = encode_target(y)
    assert classes == ["no", "yes"] and encoded.tolist() == [0, 1, 1, 0]
    encoded, classes = encode_target(y, positive_class="no")
    assert classes == ["yes", "no"] and encoded.tolist() == [1, 0, 0, 1]
    encoded, classes = encode_target(pd.Series([0, 1, 1, 0]))
    assert classes == [0, 1] and encoded.tolist() == [0, 1, 1, 0], "a 0/1 target keeps its values"


def test_project_configs_pass_every_check():
    for path in ["config.yaml", "config_adult.yaml"]:
        config = load_config(path)
        df = clean_dataset(load_data(config["data"]["path"], **config["data"]["read_options"]), config["diagnostics"])
        assert check_dataset(df, config, path) == [], path


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    tests = [(name, f) for name, f in globals().items() if name.startswith("test_") and callable(f)]
    failed = 0
    for name, test in tests:
        try:
            test()
            print(f"PASS  {name}")
        except Exception:                                  # report every failure, then exit non-zero
            failed += 1
            print(f"FAIL  {name}\n{traceback.format_exc()}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
