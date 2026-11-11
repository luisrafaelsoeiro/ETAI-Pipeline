"""
Preprocessing -- raw data in, model-ready data out: category cleanup, domain-rule/
placeholder -> NaN conversion, de-duplication, mechanism-matched imputation, a
leak-safe/deployable encoder/scaler pipeline, and the split that sets the final test
set aside. One file, one obvious place to look for "how does raw data become model-ready."

See Practical/W4/notebooks/02_preprocessing.ipynb for the full walkthrough, and
03_cross_validation.ipynb for why the split below now creates a *locked* test set.

Two things every function here respects, on purpose:
  - leak-safe: `clean_dataset` and `split_features_target` learn nothing from the data
    (no means, no category lists, no target), so they're safe to run on the whole
    dataset. Everything that *is* learned from data -- imputation, encoding, scaling --
    lives inside `build_preprocessor`'s ColumnTransformer, which sits inside the model's
    sklearn Pipeline. That means it gets re-fit on the training part of every CV fold,
    and never sees the validation fold or the locked test set.
  - deployable from day one: nothing before the split needs the target column --
    `y` comes back as `None` on label-free inference data, and nothing breaks.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import (
    OneHotEncoder, OrdinalEncoder, TargetEncoder, StandardScaler, MinMaxScaler, RobustScaler,
)
from category_encoders import CountEncoder


def flag_invalid_values(df: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """
    Applies a dict of {column: {"min": ..., "max": ...}} domain rules (either bound is
    optional) and converts violations to NaN **in place** on `df`. An "impossible but
    not missing" value (an age of -3, a COMPAS decile score of 15) counts as missing
    once this runs -- `.isna()` alone would never have caught it.

    Returns a small report: how many violations were found per column.
    """
    report_rows = []
    for column, bounds in rules.items():
        if column not in df.columns:
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        lower_ok = numeric >= bounds["min"] if "min" in bounds else pd.Series(True, index=numeric.index)
        upper_ok = numeric <= bounds["max"] if "max" in bounds else pd.Series(True, index=numeric.index)
        violations = numeric.notna() & ~(lower_ok & upper_ok)
        report_rows.append({"column": column, "rule": bounds, "violations": int(violations.sum())})
        df.loc[violations, column] = np.nan
    return pd.DataFrame(report_rows)


def _canonicalize_categories(df: pd.DataFrame, columns_and_maps: dict, placeholder_tokens: set) -> pd.DataFrame:
    out = df.copy()
    for col, mapping in columns_and_maps.items():
        if col not in out.columns:
            continue
        was_missing = out[col].isna()
        cleaned = out[col].astype(str).str.strip()
        lowered = cleaned.str.lower()
        out[col] = lowered.map({str(k).lower(): v for k, v in (mapping or {}).items()}).fillna(cleaned)
        # week 9: a real NaN stays NaN (astype(str) had turned it into the text "nan", which only
        # became missing again if a config happened to list "nan" as a placeholder token)
        out.loc[was_missing | out[col].astype(str).str.strip().isin(placeholder_tokens), col] = np.nan
    return out


def _placeholders_to_nan(df: pd.DataFrame, placeholder_tokens: set) -> pd.DataFrame:
    """
    Turns placeholder tokens ("?", "n/a", ...) into NaN in EVERY text column, not only the
    columns listed in `canonical_categories` (week 9: Adult's "?" sits in `workclass`,
    `occupation` and `native_country`, which have no spelling variants to canonicalize, so
    `_canonicalize_categories` never touched them and "?" reached the model as a category).
    Numeric columns are skipped; real NaNs stay NaN.
    """
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            continue
        is_placeholder = out[col].notna() & out[col].astype(str).str.strip().isin(placeholder_tokens)
        if is_placeholder.any():
            out.loc[is_placeholder, col] = np.nan
    return out


def clean_dataset(df: pd.DataFrame, diagnostics_config: dict) -> pd.DataFrame:
    """
    Applies the week 3 diagnosis: category cleanup, domain-rule/placeholder -> NaN
    conversion, and redundant-column removal. Target-agnostic -- safe to call on
    label-free inference data, since none of this depends on a target column.

    Row-preserving (week 4): every input row comes out, in the same order. Removing
    duplicate rows is a *training-only* decision and lives in `drop_duplicate_rows()` --
    at prediction time every row needs a prediction (a Kaggle submission needs one per id).
    """
    out = df.copy()
    placeholder_tokens = set(diagnostics_config.get("placeholder_tokens") or [])

    # pandas "category" columns (e.g. from parquet) -> plain values, so imputers and encoders
    # treat them like any text column
    for col in out.columns:
        if isinstance(out[col].dtype, pd.CategoricalDtype):
            out[col] = out[col].astype(object)

    # numeric columns that load as text purely because of a placeholder token
    for col in diagnostics_config.get("numeric_text_columns") or []:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col].replace(list(placeholder_tokens), np.nan), errors="coerce")

    flag_invalid_values(out, diagnostics_config.get("validity_rules") or {})

    out = _canonicalize_categories(out, diagnostics_config.get("canonical_categories") or {}, placeholder_tokens)
    out = _placeholders_to_nan(out, placeholder_tokens)

    columns_to_drop = [c for c in diagnostics_config.get("redundant_columns") or [] if c in out.columns]
    out = out.drop(columns=columns_to_drop)

    return out


def drop_duplicate_rows(df: pd.DataFrame, id_column: str = None) -> pd.DataFrame:
    """
    TRAINING DATA ONLY (week 4). Drops exact duplicate rows and repeated ids (keeping
    the first), so the same person can't be counted twice -- or land in both the
    development and the locked test set. Must run *before* `split_dev_test()`.

    Never call this on data you're predicting for: every row there needs a prediction.
    """
    out = df.drop_duplicates()
    if id_column and id_column in out.columns:
        out = out.drop_duplicates(subset=id_column, keep="first")
    return out


def drop_missing_target(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    TRAINING DATA ONLY (week 9). Drops rows whose target is missing: they can't be learned
    from or scored, and a target is never imputed. Data being predicted has no target at all
    and never goes through this.
    """
    if target not in df.columns:
        return df
    return df[df[target].notna()]


def encode_target(y: pd.Series, positive_class=None):
    """
    Week 9. Maps the target's labels to 0 .. k-1, the form every model and metric here expects
    (XGBoost requires it), and returns (y_encoded, classes): `classes[i]` is the original label
    of code i, used to name classes in the reports and in the predictions.

      - two classes: the negative class becomes 0 and the positive class 1. The positive class
        is `positive_class` when given, otherwise the larger label in sorted order (1 for 0/1,
        True for True/False, "yes" for "no"/"yes"). Metrics such as F1, precision and recall
        are computed for class 1.
      - three or more classes: codes follow the sorted labels.

    A 0/1 target keeps exactly its values (COMPAS and Adult are unchanged).
    """
    labels = sorted(pd.unique(y.dropna()), key=lambda v: (str(type(v)), v))
    if len(labels) == 2 and positive_class is not None:
        labels = [label for label in labels if label != positive_class] + [positive_class]
    codes = {label: code for code, label in enumerate(labels)}
    return y.map(codes).astype("int64"), list(labels)


def add_missingness_indicators(df: pd.DataFrame, mnar_indicator_sources: list) -> pd.DataFrame:
    """Adds a `<col>_was_missing` flag for each MNAR-diagnosed column, before that
    column gets imputed -- so a model can still see the pattern even though the fill
    value itself (median/mode) can't carry it. Target-agnostic."""
    out = df.copy()
    for col in mnar_indicator_sources:
        if col in out.columns:
            out[f"{col}_was_missing"] = out[col].isna().astype(int)
    return out


def split_features_target(df: pd.DataFrame, data_config: dict, mnar_indicator_sources: list):
    """
    Returns (X, y, extras). `y` is `None` and `extras` has no target column when called
    on label-free inference data -- nothing downstream requires the target to be present.

    `extras` holds the columns the fairness report needs, row-aligned with X: the sensitive
    attribute and, when `data.reference_score` is set, the existing score it is compared
    with (COMPAS: `score_text`). Week 9: this used to name `score_text` directly.
    """
    target = data_config["target"]
    sensitive_attr = data_config.get("sensitive_attr")           # optional (week 9)
    drop_columns = data_config.get("drop_columns") or []
    reference_column = (data_config.get("reference_score") or {}).get("column")

    df = add_missingness_indicators(df, mnar_indicator_sources)
    y = df[target] if target in df.columns else None

    # always a DataFrame (possibly with no columns), row-aligned with X, so the split never
    # has to handle a missing `extras`
    extras_cols = [c for c in [sensitive_attr, reference_column] if c and c in df.columns]
    extras = df[extras_cols].copy()

    always_drop = set(drop_columns) | {target} | ({sensitive_attr} if sensitive_attr else set())
    feature_cols = [c for c in df.columns if c not in always_drop]
    X = df[feature_cols]
    return X, y, extras


def check_feature_columns(X: pd.DataFrame, preprocessing_config: dict) -> list:
    """
    Week 9. The ColumnTransformer only uses the columns listed in `numeric_features`,
    `categorical_features` and `mnar_indicator_sources`, and silently drops every other one.
    Raises an error if a listed column is not in X (a typo or a column from another dataset),
    and returns the columns of X that are listed nowhere, so a run can report them.
    """
    listed = (list(preprocessing_config["numeric_features"]) + list(preprocessing_config["categorical_features"])
              + [f"{c}_was_missing" for c in preprocessing_config.get("mnar_indicator_sources") or []])
    absent = [c for c in listed if c not in X.columns]
    if absent:
        raise ValueError(f"Features listed in the config but not in the data: {absent}")
    not_numeric = [c for c in preprocessing_config["numeric_features"] if not pd.api.types.is_numeric_dtype(X[c])]
    if not_numeric:
        raise ValueError(f"Numeric features that hold text: {not_numeric}. Add them to "
                         "`diagnostics.numeric_text_columns` or list them as categorical.")
    return [c for c in X.columns if c not in listed]


_SCALERS = {"none": "passthrough", "standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}
# Each entry takes a random_state (only the target encoder actually uses it).
# Target encoding uses sklearn's TargetEncoder (new in week 4, replacing category_encoders'):
# during fit it *cross-fits* -- each training row is encoded with category means computed
# on the OTHER internal folds, never on its own label. Without that, a row's own target
# leaks into its own feature value, and the model learns to trust the encoding more than
# it deserves. Unseen categories at predict time get the overall target mean.
_ENCODERS = {
    "onehot": lambda seed: OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    "ordinal": lambda seed: OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
    "count": lambda seed: CountEncoder(handle_unknown=0, handle_missing=0),
    # target_type "auto" (week 9, was "binary"): binary for a yes/no target -- the same encoding
    # as before -- and one column per class for a multiclass target
    "target": lambda seed: TargetEncoder(target_type="auto", cv=StratifiedKFold(5, shuffle=True, random_state=seed)),
}


def build_preprocessor(preprocessing_config: dict) -> ColumnTransformer:
    """
    Factory: builds a leak-safe ColumnTransformer for the chosen encoder/scaler pair --
    read from config.yaml's `preprocessing` section.

    Numeric imputation supports:
      - "median" / "mean": impute before scaling
      - "knn": scale before KNN imputation so distance calculations are not
        dominated by high-magnitude features.

    Nothing is fit here: fitting happens later, on the training part of each
    CV fold only, because this object is placed inside the model's sklearn
    Pipeline.
    """
    encoder_name = preprocessing_config["encoder"]
    scaler_name = preprocessing_config["scaler"]
    numeric_features = preprocessing_config["numeric_features"]
    categorical_features = preprocessing_config["categorical_features"]
    mnar_indicator_sources = preprocessing_config.get("mnar_indicator_sources") or []
    imputation = preprocessing_config.get("imputation") or {}

    scaler_factory = _SCALERS[scaler_name]
    scaler = scaler_factory() if callable(scaler_factory) else scaler_factory
    encoder = _ENCODERS[encoder_name](
        preprocessing_config.get("random_state")
    )

    numeric_strategy = imputation.get("numeric_strategy", "median")

    if numeric_strategy == "knn":
        numeric_pipeline = Pipeline([
            ("scale", scaler),
            (
                "impute",
                KNNImputer(
                    n_neighbors=imputation.get("n_neighbors", 5)
                ),
            ),
        ])
    else:
        numeric_pipeline = Pipeline([
            (
                "impute",
                SimpleImputer(strategy=numeric_strategy),
            ),
            ("scale", scaler),
        ])

    categorical_strategy = imputation.get("categorical_strategy", "most_frequent")
    categorical_pipeline = Pipeline([
        (
            "impute",
            SimpleImputer(
                strategy=categorical_strategy,
                # "constant": an explicit category for missing values (week 9: configurable)
                fill_value=imputation.get("categorical_fill_value", "missing") if categorical_strategy == "constant" else None,
            ),
        ),
        ("encode", encoder),
    ])

    indicator_cols = [
        f"{c}_was_missing"
        for c in mnar_indicator_sources
    ]

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
        ("indicators", "passthrough", indicator_cols),
    ])

def split_dev_test(X, y, extras, test_size: float, random_state: int):
    """
    Sets the final test set aside (week 4 -- replaces week 2/3's `split_train_test`).

    Stratified split of X, y and the extras frame (race/score_text, kept for the fairness
    report) together, so all three stay row-aligned. Returns a *development* set and a
    *locked test set*:
      - development set: everything we're allowed to learn from and compare models on.
        Cross-validation (src/evaluate.py) splits it again into train/validation folds.
      - locked test set: never used to fit, tune, compare or choose anything. Its size and seed live in config.yaml's `test_set` section and are never changed after today.
    """
    X_dev, X_test, y_dev, y_test, extras_dev, extras_test = train_test_split(
        X, y, extras, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_dev, X_test, y_dev, y_test, extras_dev, extras_test
