"""
Evaluation on the development set: stratified k-fold cross-validation of the whole pipeline on
several metrics, out-of-fold reports from the out-of-fold probabilities, a fairness check and a
paired comparison of two models. The locked test set is only used by `holdout_report` /
`holdout_evaluation`.

Week 9 -- any classification dataset: the target arrives encoded as 0 .. k-1
(`encode_target` in src/preprocessing.py), probabilities are kept as one column per class, and
the reports name the classes. With two classes everything reads and scores exactly as before
(class 1 = the positive class, a 0.5 threshold on its probability); with three or more, the
prediction is the most probable class.
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    classification_report, confusion_matrix, fbeta_score, get_scorer, make_scorer, roc_auc_score,
)
from sklearn.model_selection import cross_validate


def build_scorers(metrics: list) -> dict:
    """
    {name: scorer} for scikit-learn metric names (`accuracy`, `roc_auc`, `f1_macro`, ...), plus
    `fbeta_<beta>` (e.g. `fbeta_2`: recall weighted twice as much as precision) and, for three or
    more classes, `fbeta_<beta>_<average>` (e.g. `fbeta_2_macro`).
    """
    scorers = {}
    for name in metrics:
        if name.startswith("fbeta_"):
            parts = name.split("_")
            try:
                beta = float(parts[1])
            except (IndexError, ValueError):
                raise ValueError(f"'{name}': write fbeta_<beta> or fbeta_<beta>_<average>, e.g. fbeta_2") from None
            options = {"average": parts[2]} if len(parts) > 2 else {}
            scorers[name] = make_scorer(fbeta_score, beta=beta, zero_division=0, **options)
        else:
            scorers[name] = get_scorer(name)
    return scorers


def predicted_classes(proba, threshold: float = 0.5) -> np.ndarray:
    """
    Class codes from probabilities. Two classes: class 1 when its probability is >= `threshold`
    (accepts the full (n, 2) matrix or just P(class 1)). Three or more: the most probable class.
    """
    proba = np.asarray(proba)
    if proba.ndim == 1:
        return (proba >= threshold).astype(int)
    if proba.shape[1] == 2:
        return (proba[:, 1] >= threshold).astype(int)
    return proba.argmax(axis=1)


def _positive_or_all(proba):
    """What `roc_auc_score` expects: P(class 1) for two classes, the whole matrix otherwise."""
    proba = np.asarray(proba)
    return proba if proba.ndim == 1 else (proba[:, 1] if proba.shape[1] == 2 else proba)


def cross_validate_pipeline(pipeline, X, y, cv, scorers: dict, selection_metric: str, n_jobs: int = 1):
    """
    Fits a fresh copy of `pipeline` (preprocessing included) on each fold's training part and
    scores it on that fold's validation part, on every metric in `scorers`.

    Returns (fold_scores, oof_proba):
      - fold_scores: one row per fold with the validation score of every metric, plus the
        train score and train-validation gap of `selection_metric`;
      - oof_proba: out-of-fold probabilities, one row per row of X and one column per class
        (column 1 = the positive class for a yes/no target), from the fold model that did not
        train on that row. `cv` must be a partition (each row validated exactly once).
    """
    scores = cross_validate(pipeline, X, y, cv=cv, scoring=scorers, return_train_score=True,
                            return_estimator=True, return_indices=True, n_jobs=n_jobs)
    fold_scores = pd.DataFrame({"fold": range(1, cv.get_n_splits() + 1)})
    fold_scores["train"] = scores[f"train_{selection_metric}"]
    fold_scores["validation"] = scores[f"test_{selection_metric}"]
    fold_scores["gap"] = fold_scores["train"] - fold_scores["validation"]
    for name in scorers:
        fold_scores[name] = scores[f"test_{name}"]

    oof_proba = np.zeros((len(X), len(np.unique(y))))
    for model, val_idx in zip(scores["estimator"], scores["indices"]["test"]):
        # columns placed by the fold model's own classes (codes 0 .. k-1)
        oof_proba[np.ix_(val_idx, model.classes_)] = model.predict_proba(X.iloc[val_idx])
    return fold_scores, oof_proba


def cv_report(fold_scores: pd.DataFrame, metrics: list, selection_metric: str, title: str) -> str:
    """Per-fold table and the mean and std of every metric, as text."""
    lines = [
        f"{title} ({len(fold_scores)} stratified folds, selection metric: {selection_metric})",
        "",
        fold_scores.to_string(index=False, float_format=lambda v: f"{v:.3f}"),
        "",
        fold_scores[metrics].agg(["mean", "std"]).T.to_string(float_format=lambda v: f"{v:.3f}"),
        f"\n{selection_metric}: train-validation gap mean = {fold_scores['gap'].mean():+.3f}",
    ]
    return "\n".join(lines)


def corrected_paired_test(scores_a, scores_b, test_train_ratio: float) -> dict:
    """
    Corrected paired t-test (Nadeau & Bengio, 2003) for two models scored on the same CV folds.
    Folds share most of their training rows, so their scores are correlated and the plain
    t-test is overconfident; the variance of the differences is inflated by `test_train_ratio`
    (validation rows / training rows per fold: 1 / (k - 1) for k-fold).

    Returns the mean difference (a - b), the folds where a scores higher, t and the two-sided
    p-value.
    """
    diff = np.asarray(scores_a, dtype=float) - np.asarray(scores_b, dtype=float)
    k = len(diff)
    se = np.sqrt((1 / k + test_train_ratio) * diff.var(ddof=1))
    t = float(diff.mean() / se) if se > 0 else 0.0
    p_value = float(2 * stats.t.sf(abs(t), df=k - 1))
    return {"mean_diff": float(diff.mean()), "wins": f"{int((diff > 0).sum())}/{k}", "t": t, "p_value": p_value}


def _class_names(proba, classes) -> list:
    n_classes = 2 if np.asarray(proba).ndim == 1 else np.asarray(proba).shape[1]
    return [str(c) for c in (classes if classes is not None else range(n_classes))]


def _confusion_and_report(y_true, y_pred, names) -> str:
    codes = list(range(len(names)))
    cm = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=codes),
                      index=[f"true {n}" for n in names], columns=[f"predicted {n}" for n in names])
    report = classification_report(y_true, y_pred, labels=codes, target_names=names, zero_division=0)
    return f"{cm.to_string()}\n\n{report}"


def oof_classification_report(y_true, oof_proba, threshold: float = 0.5, classes=None) -> str:
    """Confusion matrix and classification report of the out-of-fold predictions, as text.
    `classes`: the original label of each class code (default: the codes themselves)."""
    names = _class_names(oof_proba, classes)
    rule = f"threshold {threshold}" if len(names) == 2 else "most probable class"
    y_pred = predicted_classes(oof_proba, threshold)
    return (f"Out-of-fold predictions, development set ({rule}):\n\n"
            f"{_confusion_and_report(y_true, y_pred, names)}")


def fairness_report(y_true, oof_proba, extras: pd.DataFrame, sensitive_attr: str = "race",
                    threshold: float = 0.5, reference: dict = None, classes=None) -> str:
    """
    False positive rate per group of `sensitive_attr`. Rows with an unknown group are reported
    as "unknown". A simple check, not a full audit.

      - two classes: the share of the people whose true class is 0 who were predicted 1, for
        the model's out-of-fold predictions and, when `reference` is given, for an existing
        score on the same rows (`data.reference_score`): every value of `reference["column"]`
        other than `reference["negative"]` counts as predicted 1 (COMPAS: a `score_text` other
        than "Low" = predicted to reoffend);
      - three or more classes: for each class, the share of the people NOT truly in it who were
        predicted to be in it (one vs rest).

    Week 9: COMPAS's `score_text` comparison used to be built in, the sensitive attribute was
    required and the report assumed two classes. Without `sensitive_attr` the report is skipped.
    """
    if not sensitive_attr or extras is None or sensitive_attr not in extras.columns:
        return "Fairness report: skipped (no sensitive attribute; set data.sensitive_attr to audit one)\n"
    names = _class_names(oof_proba, classes)
    df = extras.copy()
    df[sensitive_attr] = df[sensitive_attr].astype(object).fillna("unknown")
    df["y_true"] = pd.Series(y_true).values
    df["y_pred_model"] = predicted_classes(oof_proba, threshold)

    if len(names) > 2:
        lines = [
            f"False positive rate per class by {sensitive_attr} (development set, out-of-fold)",
            "(for each class: share of the people NOT truly in it, but predicted to be in it)",
            "",
        ]
        table = {}
        for code, name in enumerate(names):
            table[name] = {group: (g.loc[g["y_true"] != code, "y_pred_model"] == code).mean()
                           for group, g in df.groupby(sensitive_attr)}
        counts = df.groupby(sensitive_attr).size().rename("n")
        lines.append(pd.DataFrame(table).join(counts).to_string(float_format=lambda v: f"{v:.2f}"))
        return "\n".join(lines) + "\n"

    predictions = [("Our model", "y_pred_model")]
    if reference and reference.get("column") in df.columns:
        df["y_pred_reference"] = (df[reference["column"]] != reference["negative"]).astype(int)
        predictions.append((reference.get("label", reference["column"]), "y_pred_reference"))

    lines = [
        f"False positive rate by {sensitive_attr} (development set, out-of-fold)",
        f"(share of the people whose true class is {names[0]}, but were predicted {names[1]})",
        "",
    ]
    for label, col in predictions:
        lines.append(f"  {label}:")
        for group, g in df.groupby(sensitive_attr):
            negatives = g[g["y_true"] == 0]
            if len(negatives) == 0:
                continue
            fpr = (negatives[col] == 1).mean()
            lines.append(f"    {str(group):<20s} FPR = {fpr:.2f}  (n={len(negatives)})")
        lines.append("")
    return "\n".join(lines)


def holdout_report(model, X_test, y_test, extras_test, classes=None, sensitive_attr: str = None,
                   threshold: float = 0.5, reference: dict = None):
    """
    The fitted final `model` scored once on the locked test set, as text: accuracy, ROC-AUC
    (one vs rest for three or more classes), confusion matrix, classification report and the
    fairness check on the test rows. Returns (text, accuracy, roc_auc, y_test_proba).
    """
    proba = model.predict_proba(X_test)
    names = _class_names(proba, classes)
    y_pred = predicted_classes(proba, threshold)
    accuracy = (y_pred == np.asarray(y_test)).mean()
    if len(names) == 2:
        roc_auc, rule, auc_name = roc_auc_score(y_test, proba[:, 1]), f"threshold {threshold}", "roc_auc"
    else:
        roc_auc = roc_auc_score(y_test, proba, multi_class="ovr", labels=list(range(len(names))))
        rule, auc_name = "most probable class", "roc_auc_ovr"
    text = (
        f"Locked test set ({len(X_test)} rows, evaluated once, {rule})\n"
        f"  accuracy: {accuracy:.3f}   {auc_name}: {roc_auc:.3f}\n\n"
        f"{_confusion_and_report(y_test, y_pred, names)}\n"
        "Fairness on the locked test set (header says out-of-fold; these are test rows):\n"
        + fairness_report(y_test, proba, extras_test, sensitive_attr, threshold, reference, classes)
    )
    return text, accuracy, roc_auc, proba


def holdout_evaluation(pipeline, X_dev, y_dev, X_test, y_test, extras_test,
                       sensitive_attr: str = "race", threshold: float = 0.5, reference: dict = None,
                       classes=None):
    """
    Fit the final pipeline on the entire development set, then evaluate once
    on the locked test set.

    The test set must not be used for model selection, tuning, or preprocessing.
    Because preprocessing is inside the Pipeline, it is fitted only on X_dev.

    Returns (accuracy, roc_auc, y_test_proba): the probabilities have one column per class.
    """
    pipeline.fit(X_dev, y_dev)                     # fit ONLY on the development data
    text, accuracy, roc_auc, proba = holdout_report(pipeline, X_test, y_test, extras_test, classes,
                                                    sensitive_attr, threshold, reference)
    print(text)
    return accuracy, roc_auc, proba
