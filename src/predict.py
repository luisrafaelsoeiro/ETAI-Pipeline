"""Predicting new, unlabelled rows with the final (refit) pipeline -- week 9."""
import os

import numpy as np
import pandas as pd

from src.evaluate import predicted_classes
from src.preprocessing import check_feature_columns, clean_dataset, split_features_target


def predict_new_data(pipeline, df_raw: pd.DataFrame, config: dict, threshold: float = 0.5,
                     classes=None) -> pd.DataFrame:
    """
    Scores raw, label-free rows with the fitted `pipeline`, through the same steps as the
    training data:
      - the same cleaning (`clean_dataset`), but never `drop_duplicate_rows`: every row needs
        a prediction;
      - the same feature split (`split_features_target`; there is no target, so `y` is None);
        a feature the model needs but the new file lacks stops here with a clear error;
      - the pipeline's probabilities and a prediction: for two classes, the positive class when
        its probability is >= `threshold`; for three or more, the most probable class.

    `classes` are the original labels of the class codes (`encode_target`), so predictions come
    out in the data's own labels ("yes" / "no", "High" / ...), not as codes.

    Returns one row per input row, in the input order: the id (`diagnostics.id_column`, or the
    row number when the file has none), the probability -- of the positive class for a yes/no
    target, one `probability_<class>` column per class otherwise -- and the prediction.
    """
    df = clean_dataset(df_raw, config["diagnostics"])
    mnar_sources = config["preprocessing"].get("mnar_indicator_sources") or []
    X, _, _ = split_features_target(df, config["data"], mnar_sources)
    try:
        check_feature_columns(X, config["preprocessing"])
    except ValueError as error:
        raise ValueError(f"The new data can't be predicted: {error}") from None

    proba = pipeline.predict_proba(X)
    codes = predicted_classes(proba, threshold)
    names = list(classes) if classes is not None else list(range(proba.shape[1]))

    id_column = config["diagnostics"].get("id_column")
    ids = df[id_column].to_numpy() if id_column and id_column in df.columns else np.arange(len(df))
    predictions = pd.DataFrame({id_column if id_column and id_column in df.columns else "row": ids})
    if proba.shape[1] == 2:
        predictions["probability"] = proba[:, 1]
    else:
        for code, name in enumerate(names):
            predictions[f"probability_{name}"] = proba[:, code]
    predictions["prediction"] = np.asarray(names, dtype=object)[codes] if classes is not None else codes
    if classes is not None and all(isinstance(n, (int, np.integer)) for n in names):
        predictions["prediction"] = predictions["prediction"].astype("int64")

    if len(predictions) != len(df_raw):
        raise AssertionError(f"{len(df_raw)} rows in, {len(predictions)} predictions out: every row needs one.")
    return predictions


def save_predictions(predictions: pd.DataFrame, path: str) -> str:
    """Writes the predictions to `path` (folder created if needed) and returns the path."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    predictions.to_csv(path, index=False)
    return path
