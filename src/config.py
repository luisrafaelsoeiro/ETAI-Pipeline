"""
Loading and checking a run's configuration (week 9: any dataset, not only COMPAS).

Two kinds of checks run before anything is fitted, so a wrong setting stops the run at the
start with a clear message instead of failing (or silently doing the wrong thing) minutes later:

  - `load_config()` -> `validate_config()`: the config file on its own -- required keys, known
    option names (encoder, scaler, imputer, model, metrics), feature lists that don't overlap or
    contain the target, tuning search spaces whose parameter names exist in the pipeline, ...
  - `check_dataset()`: the config against the loaded data -- the target and every listed column
    exist, the target is a classification target with enough rows per class for the stratified
    splits, numeric features really are numeric, metrics suit the number of classes, ...

Every problem found is collected and reported together (`ConfigError`), so one run lists them all.
"""
import difflib
import os

import numpy as np
import pandas as pd
import yaml

# Defaults for optional settings, filled in by `load_config()` (a key present in the file wins).
_DEFAULTS = {
    "data": {"drop_columns": [], "sensitive_attr": None, "reference_score": None,
             "positive_class": None, "read_options": {}},
    "test_set": {"evaluate": False},
    "cv": {"shuffle": True, "n_jobs": 1},
    "diagnostics": {"id_column": None, "numeric_text_columns": [], "placeholder_tokens": [],
                    "validity_rules": {}, "canonical_categories": {}, "redundant_columns": []},
    "preprocessing": {"mnar_indicator_sources": [], "imputation": {}, "random_state": None},
    "feature_selection": {"enabled": False},
    "tuning": {"enabled": False},
    "output": {"results_dir": "results", "scoreboard": "results/runs.csv"},
}
_REQUIRED = {
    "data": ["path", "target"],
    "test_set": ["size", "random_state"],
    "cv": ["n_splits", "metrics", "selection_metric"],
    "preprocessing": ["encoder", "scaler", "numeric_features", "categorical_features"],
    "model": ["type"],
}
_NUMERIC_IMPUTERS = {"median", "mean", "most_frequent", "constant", "knn"}
_CATEGORICAL_IMPUTERS = {"most_frequent", "constant"}
# scikit-learn metrics that only exist for a yes/no target (their multiclass versions take a
# suffix: f1_macro, precision_weighted, roc_auc_ovr, ...)
BINARY_ONLY_METRICS = {"precision", "recall", "f1", "roc_auc", "average_precision", "jaccard"}
# a numeric target with more distinct values than this is treated as continuous (regression)
_MAX_CLASSES = 20


class ConfigError(ValueError):
    """A configuration (or configuration-vs-data) problem, with every problem found listed."""

    def __init__(self, source: str, problems: list):
        self.problems = list(problems)
        super().__init__(f"{source}: {len(self.problems)} problem(s)\n  - " + "\n  - ".join(self.problems))


def load_config(path: str = "config.yaml", validate: bool = True) -> dict:
    """Reads the YAML config at `path`, fills in the defaults of optional settings and (by
    default) checks it with `validate_config()`. Raises `ConfigError` listing every problem."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ConfigError(path, ["the file is empty or not a YAML mapping (key: value)"])

    for section, defaults in _DEFAULTS.items():
        if config.get(section) is None:
            config[section] = {}
        if not isinstance(config[section], dict):
            raise ConfigError(path, [f"`{section}` must be a mapping (key: value), got {type(config[section]).__name__}"])
        for key, value in defaults.items():
            if config[section].get(key) is None:
                config[section][key] = value.copy() if isinstance(value, (dict, list)) else value
    if validate:
        validate_config(config, path)
    return config


def _is_list_of_str(value) -> bool:
    return isinstance(value, list) and all(isinstance(v, str) for v in value)


def search_space_for(config: dict) -> dict:
    """The tuning search space of the configured model (plus the selection's, when it is on)."""
    tuning = config.get("tuning") or {}
    spaces = tuning.get("search_spaces") or {}
    space = dict(spaces.get(config["model"]["type"]) or tuning.get("search_space") or {})
    selection = config.get("feature_selection") or {}
    if selection.get("enabled"):
        space.update(selection.get("search_space") or {})
    return space


def validate_config(config: dict, source: str = "config") -> None:
    """Checks a loaded config on its own (no data needed). Raises `ConfigError` listing every
    problem; returns None when the config is usable."""
    # imported here: these modules import nothing from this one, but keep startup light
    from src.evaluate import build_scorers
    from src.model import _MODELS, build_model, build_pipeline
    from src.preprocessing import _ENCODERS, _SCALERS

    problems = []
    for section, keys in _REQUIRED.items():
        if not isinstance(config.get(section), dict):
            problems.append(f"missing section `{section}`")
            continue
        problems += [f"missing `{section}.{key}`" for key in keys if config[section].get(key) is None]
    if problems:                                          # the checks below need these keys
        raise ConfigError(source, problems)

    data, cv, prep, model = config["data"], config["cv"], config["preprocessing"], config["model"]

    # --- data ---
    for key in ("path", "target"):
        if not isinstance(data[key], str):
            problems.append(f"`data.{key}` must be text, got {data[key]!r}")
    if data.get("sensitive_attr") is not None and not isinstance(data["sensitive_attr"], str):
        problems.append("`data.sensitive_attr` must be a column name or null")
    if not _is_list_of_str(data.get("drop_columns")):
        problems.append("`data.drop_columns` must be a list of column names")
    reference = data.get("reference_score")
    if reference is not None and not (isinstance(reference, dict) and "column" in reference and "negative" in reference):
        problems.append("`data.reference_score` must be null or {column: ..., negative: ..., label: ...}")
    if not isinstance(data.get("read_options"), dict):
        problems.append("`data.read_options` must be a mapping of pandas reader options (e.g. {sep: ';'})")

    # --- locked test set ---
    size = config["test_set"]["size"]
    if not ((isinstance(size, float) and 0 < size < 1) or (isinstance(size, int) and not isinstance(size, bool) and size >= 1)):
        problems.append(f"`test_set.size` must be a share between 0 and 1 (e.g. 0.2) or a number of rows, got {size!r}")

    # --- cross-validation and metrics ---
    if not (isinstance(cv["n_splits"], int) and cv["n_splits"] >= 2):
        problems.append(f"`cv.n_splits` must be an integer >= 2, got {cv['n_splits']!r}")
    metrics = cv["metrics"]
    if not (_is_list_of_str(metrics) and metrics):
        problems.append("`cv.metrics` must be a non-empty list of metric names")
    else:
        if len(set(metrics)) != len(metrics):
            problems.append(f"`cv.metrics` lists a metric twice: {metrics}")
        for name in metrics:
            try:
                build_scorers([name])
            except (ValueError, KeyError) as error:
                problems.append(f"unknown metric `{name}` in `cv.metrics` ({str(error).splitlines()[0][:120]})")
        if cv["selection_metric"] not in metrics:
            problems.append(f"`cv.selection_metric` '{cv['selection_metric']}' must be one of `cv.metrics`: {metrics}")

    # --- preprocessing ---
    if prep["encoder"] not in _ENCODERS:
        problems.append(f"unknown `preprocessing.encoder` '{prep['encoder']}'. Options: {list(_ENCODERS)}")
    if prep["scaler"] not in _SCALERS:
        problems.append(f"unknown `preprocessing.scaler` '{prep['scaler']}'. Options: {list(_SCALERS)}")
    numeric, categorical = prep["numeric_features"], prep["categorical_features"]
    mnar = prep.get("mnar_indicator_sources")
    for name, value in [("numeric_features", numeric), ("categorical_features", categorical),
                        ("mnar_indicator_sources", mnar)]:
        if not _is_list_of_str(value):
            problems.append(f"`preprocessing.{name}` must be a list of column names (use [] for none)")
    if _is_list_of_str(numeric) and _is_list_of_str(categorical):
        features = numeric + categorical
        if not features:
            problems.append("no features: `preprocessing.numeric_features` and `categorical_features` are both empty")
        repeated = sorted({f for f in features if features.count(f) > 1})
        if repeated:
            problems.append(f"features listed twice (or as both numeric and categorical): {repeated}")
        not_features = {data.get("target"): "the target", data.get("sensitive_attr"): "the sensitive attribute"}
        not_features.update({c: "in data.drop_columns" for c in data.get("drop_columns") or []})
        for column in features:
            if column in not_features:
                problems.append(f"`{column}` is listed as a feature but is {not_features[column]}")
    imputation = prep.get("imputation") or {}
    if imputation.get("numeric_strategy", "median") not in _NUMERIC_IMPUTERS:
        problems.append(f"unknown `imputation.numeric_strategy` '{imputation['numeric_strategy']}'. "
                        f"Options: {sorted(_NUMERIC_IMPUTERS)}")
    if imputation.get("categorical_strategy", "most_frequent") not in _CATEGORICAL_IMPUTERS:
        problems.append(f"unknown `imputation.categorical_strategy` '{imputation['categorical_strategy']}'. "
                        f"Options: {sorted(_CATEGORICAL_IMPUTERS)}")

    # --- diagnostics ---
    diagnostics = config["diagnostics"]
    for name in ("numeric_text_columns", "placeholder_tokens", "redundant_columns"):
        if not isinstance(diagnostics.get(name), list):
            problems.append(f"`diagnostics.{name}` must be a list (use [] for none)")
    rules = diagnostics.get("validity_rules")
    if not isinstance(rules, dict):
        problems.append("`diagnostics.validity_rules` must be a mapping {column: {min: ..., max: ...}}")
    else:
        for column, bounds in rules.items():
            if not (isinstance(bounds, dict) and bounds and set(bounds) <= {"min", "max"}):
                problems.append(f"`diagnostics.validity_rules.{column}` must be {{min: ..., max: ...}} (either optional)")
            elif "min" in bounds and "max" in bounds and bounds["min"] > bounds["max"]:
                problems.append(f"`diagnostics.validity_rules.{column}`: min {bounds['min']} > max {bounds['max']}")
    maps = diagnostics.get("canonical_categories")
    if not (isinstance(maps, dict) and all(isinstance(m, dict) for m in maps.values())):
        problems.append("`diagnostics.canonical_categories` must be {column: {variant: canonical value}}")
    if diagnostics.get("id_column") and _is_list_of_str(numeric) and _is_list_of_str(categorical) \
            and diagnostics["id_column"] in numeric + categorical:
        problems.append(f"the id column `{diagnostics['id_column']}` is listed as a feature")

    # --- model, feature selection, tuning ---
    if model["type"] not in _MODELS:
        problems.append(f"unknown `model.type` '{model['type']}'. Options: {list(_MODELS)}")
    else:
        try:
            build_model(model)
        except TypeError as error:
            problems.append(f"`model.params` don't fit `{model['type']}`: {error}")
    selection = config["feature_selection"]
    if selection.get("enabled") and selection.get("min_votes", 2) not in (1, 2, 3):
        problems.append(f"`feature_selection.min_votes` must be 1, 2 or 3, got {selection.get('min_votes')!r}")
    tuning = config["tuning"]
    if tuning.get("enabled") and model["type"] in _MODELS:
        space = search_space_for(config)
        if not search_space_for({**config, "feature_selection": {}}):
            problems.append(f"tuning is on, but there is no search space for `{model['type']}`: add "
                            f"`tuning.search_spaces.{model['type']}` or set `tuning.enabled: false`")
        # parameter names don't depend on the encoder/scaler choice: fall back to valid ones so
        # a typo elsewhere doesn't hide a search-space problem
        safe_prep = {**prep, "encoder": prep["encoder"] if prep["encoder"] in _ENCODERS else "onehot",
                     "scaler": prep["scaler"] if prep["scaler"] in _SCALERS else "none"}
        try:
            valid = set(build_pipeline(safe_prep, model, selection).get_params(deep=True))
        except (TypeError, ValueError, KeyError):
            valid = None                                  # reported above (model params)
        if valid is not None:
            for name, spec in space.items():
                if name not in valid:
                    hint = difflib.get_close_matches(name, sorted(valid), n=1)
                    problems.append(f"search-space parameter `{name}` doesn't exist in the pipeline"
                                    + (f" (did you mean {hint[0]}?)" if hint else ""))
                if not isinstance(spec, dict) or spec.get("type") not in ("int", "float", "categorical"):
                    problems.append(f"search space `{name}`: `type` must be int, float or categorical")
                elif spec["type"] == "categorical" and not spec.get("choices"):
                    problems.append(f"search space `{name}`: a categorical entry needs `choices`")
                elif spec["type"] != "categorical" and not ("low" in spec and "high" in spec and spec["low"] < spec["high"]):
                    problems.append(f"search space `{name}`: needs `low` < `high`")
        for key in ("n_trials", "n_splits"):
            if key in tuning and not (isinstance(tuning[key], int) and tuning[key] >= (2 if key == "n_splits" else 1)):
                problems.append(f"`tuning.{key}` must be an integer >= {2 if key == 'n_splits' else 1}")

    # --- prediction of new rows, output ---
    predict = config.get("predict")
    if predict:
        if not isinstance(predict, dict) or not predict.get("path") or not predict.get("output"):
            problems.append("`predict` needs `path` (the new data) and `output` (the predictions file)")
        elif not 0 < predict.get("threshold", 0.5) < 1:
            problems.append("`predict.threshold` must be between 0 and 1")
    output = config["output"]
    if not (isinstance(output.get("results_dir"), str) and isinstance(output.get("scoreboard"), str)):
        problems.append("`output.results_dir` and `output.scoreboard` must be paths")

    if problems:
        raise ConfigError(source, problems)


def check_dataset(df: pd.DataFrame, config: dict, source: str = "config vs data") -> list:
    """
    Checks the config against the cleaned training data, before the split. Raises `ConfigError`
    listing every problem; returns a list of warnings (things to know, not errors).
    """
    data, prep, cv = config["data"], config["preprocessing"], config["cv"]
    diagnostics, tuning = config["diagnostics"], config["tuning"]
    problems, warnings = [], []

    target = data["target"]
    if target not in df.columns:
        raise ConfigError(source, [f"target column `{target}` not found. Columns: {list(df.columns)}"])

    # --- the target: a classification target with enough rows per class ---
    y = df[target].dropna()
    classes = y.value_counts()
    if len(classes) < 2:
        problems.append(f"the target `{target}` has {len(classes)} class(es) after dropping missing values: "
                        "a classifier needs at least 2")
    fractional = pd.api.types.is_float_dtype(y) and not np.allclose(y, np.round(y))
    many_numbers = pd.api.types.is_numeric_dtype(y) and len(classes) > _MAX_CLASSES
    if fractional or many_numbers:
        problems.append(f"the target `{target}` looks continuous ({len(classes)} distinct values): this pipeline "
                        "is for classification (stratified folds, classifiers, class probabilities)")
    if data.get("positive_class") is not None and data["positive_class"] not in set(classes.index):
        problems.append(f"`data.positive_class` {data['positive_class']!r} is not a value of `{target}`: "
                        f"{sorted(classes.index.tolist(), key=str)}")
    if len(classes) >= 2:
        smallest, needed = int(classes.min()), cv["n_splits"]
        test_size = config["test_set"]["size"]
        dev_share = 1 - test_size if isinstance(test_size, float) else 1 - test_size / len(y)
        if tuning.get("enabled"):
            # the inner folds split the training part of an outer fold
            inner = tuning.get("n_splits", cv["n_splits"])
            needed = max(needed, int(np.ceil(inner / (1 - 1 / cv["n_splits"]))))
        if smallest * dev_share < needed or smallest < 2:
            problems.append(f"class {classes.idxmin()!r} of `{target}` has only {smallest} rows: too few for the "
                            f"locked test split and {needed}-fold stratified CV. Lower `cv.n_splits` / "
                            "`tuning.n_splits`, or merge rare classes")
        if len(classes) > 2:
            binary_only = [m for m in cv["metrics"] if m in BINARY_ONLY_METRICS
                           or (m.startswith("fbeta_") and m.count("_") == 1)]
            if binary_only:
                problems.append(f"`{target}` has {len(classes)} classes, but these metrics only exist for two "
                                f"classes: {binary_only}. Use e.g. f1_macro, precision_macro, recall_macro, "
                                "roc_auc_ovr, fbeta_2_macro")
            if prep["encoder"] == "target":
                warnings.append(f"target encoding with {len(classes)} classes adds one column per category column "
                                "per class")
            if data.get("reference_score"):
                warnings.append("`data.reference_score` compares a yes/no score: skipped for a multiclass target")
    missing_target = int(df[target].isna().sum())
    if missing_target:
        warnings.append(f"{missing_target} training rows have no target: dropped before the split")

    # --- every listed column exists, with a usable type ---
    columns = set(df.columns)
    listed = {
        "preprocessing.numeric_features": prep["numeric_features"],
        "preprocessing.categorical_features": prep["categorical_features"],
        "preprocessing.mnar_indicator_sources": prep["mnar_indicator_sources"],
    }
    for name, values in listed.items():
        absent = [c for c in values if c not in columns]
        if absent:
            problems.append(f"`{name}` lists columns that aren't in the data: {absent}")
    for name in ("sensitive_attr",):
        if data.get(name) and data[name] not in columns:
            problems.append(f"`data.{name}` `{data[name]}` not found in the data")
    reference = data.get("reference_score")
    if reference and reference["column"] not in columns:
        problems.append(f"`data.reference_score.column` `{reference['column']}` not found in the data")
    not_numeric = [c for c in prep["numeric_features"] if c in columns and not pd.api.types.is_numeric_dtype(df[c])]
    if not_numeric:
        examples = {c: df[c].dropna().astype(str).head(3).tolist() for c in not_numeric}
        problems.append(f"numeric features that hold text after cleaning: {examples}. Add them to "
                        "`diagnostics.numeric_text_columns` (converted, non-numbers -> NaN) or list them "
                        "as categorical")
    empty = [c for c in prep["numeric_features"] + prep["categorical_features"] if c in columns and df[c].isna().all()]
    if empty:
        problems.append(f"features that are entirely missing: {empty}")
    if diagnostics.get("id_column") and diagnostics["id_column"] not in columns:
        warnings.append(f"id column `{diagnostics['id_column']}` not found: duplicates are checked on whole rows "
                        "only, and predictions are numbered by row")
    unlisted = [c for c in df.columns if c not in set(prep["numeric_features"]) | set(prep["categorical_features"])
                | set(data.get("drop_columns") or []) | {target, data.get("sensitive_attr"), diagnostics.get("id_column")}
                and not (reference and c == reference["column"])]
    if unlisted:
        warnings.append(f"columns not listed as features, nor dropped (left out of the model): {unlisted}")

    if problems:
        raise ConfigError(source, problems)
    return warnings
