"""
Hyperparameter tuning with Optuna, and nested cross-validation.

Tuning uses the development set only, and scores every candidate by cross-validation of the
whole pipeline on the same folds, on the selection metric. The best trial's score is the maximum
of many noisy scores, so it is optimistic; the honest estimate of "tune, then refit" comes from
`nested_cross_validate`, which also scores every reported metric on each outer fold.
"""
import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import cross_val_score


def tune_pipeline(pipeline, X, y, cv, scorer, search_space: dict, n_trials: int,
                  random_state: int, n_jobs: int = 1):
    """
    Runs an Optuna study (seeded TPE sampler) on (X, y). Each trial draws one value per entry
    of `search_space` (`type`: int / float / categorical; keys are Pipeline parameter names,
    e.g. `model__max_depth`), sets them on a fresh copy of `pipeline`, and returns its mean
    CV score over `cv` with `scorer` (the selection metric).

    Returns (best_pipeline, study): an unfitted copy of `pipeline` with the best trial's
    hyperparameters, and the study with every trial (mean score and fold std).
    """
    def objective(trial):
        params = {}
        for name, spec in search_space.items():
            if spec["type"] == "int":
                params[name] = trial.suggest_int(name, spec["low"], spec["high"], log=spec.get("log", False))
            elif spec["type"] == "float":
                params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
            elif spec["type"] == "categorical":
                params[name] = trial.suggest_categorical(name, spec["choices"])
            else:
                raise ValueError(f"Unknown search-space type for {name}: {spec['type']}. "
                                 "Options: int, float, categorical")
        candidate = clone(pipeline).set_params(**params)
        scores = cross_val_score(candidate, X, y, cv=cv, scoring=scorer, n_jobs=n_jobs)
        trial.set_user_attr("std", float(scores.std(ddof=1)))
        return scores.mean()

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials)
    best_pipeline = clone(pipeline).set_params(**study.best_params)
    return best_pipeline, study


def nested_cross_validate(pipeline, X, y, outer_cv, inner_cv, scorers: dict, selection_metric: str,
                          search_space: dict, n_trials: int, random_state: int, n_jobs: int = 1):
    """
    Honest estimate of the tuning procedure. For every outer fold: tune on the outer-training
    rows (`tune_pipeline` with `inner_cv`, on `selection_metric`), refit the best pipeline on
    them, and score it on the outer-validation rows, which took no part in the tuning.

    Returns (fold_scores, oof_proba), as `cross_validate_pipeline` does, with two extra kinds
    of columns: `inner_best` (the tuning score inside that fold) and the hyperparameters chosen.
    """
    rows = []
    oof_proba = np.zeros((len(X), len(np.unique(y))))      # one column per class (week 9)
    for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X, y), start=1):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        best_pipeline, study = tune_pipeline(pipeline, X_train, y_train, inner_cv, scorers[selection_metric],
                                             search_space, n_trials, random_state, n_jobs)
        best_pipeline.fit(X_train, y_train)

        train_score = scorers[selection_metric](best_pipeline, X_train, y_train)
        val_scores = {name: scorer(best_pipeline, X_val, y_val) for name, scorer in scorers.items()}
        oof_proba[np.ix_(val_idx, best_pipeline.classes_)] = best_pipeline.predict_proba(X_val)
        rows.append({"fold": fold, "train": train_score, "validation": val_scores[selection_metric],
                     "gap": train_score - val_scores[selection_metric], **val_scores,
                     "inner_best": study.best_value, **study.best_params})
    return pd.DataFrame(rows), oof_proba


def nested_cv_report(nested_scores: pd.DataFrame, metrics: list, selection_metric: str) -> str:
    """
    Report the honest nested-CV performance on the selection metric, the mean and std of every
    reported metric on the outer folds, and the hyperparameters chosen in each outer fold, as text.

    The outer validation folds estimate the performance of the complete
    tuning procedure on unseen development data.
    """
    lines = [
        f"Nested cross-validation ({len(nested_scores)} outer folds, "
        f"selection metric: {selection_metric})",
        "",
    ]

    labels = {
        "train": "Train score",
        "validation": "Outer validation score (HONEST estimate)",
        "gap": "Train-validation gap",
        "inner_best": "Inner best-trial score",
    }
    for column, label in labels.items():
        if column not in nested_scores.columns:
            continue
        values = nested_scores[column]
        sign = "+" if column == "gap" else ""
        lines.append(f"{label:<43} mean = {values.mean():{sign}.3f}   std = {values.std(ddof=1):.3f}")

    present_metrics = [m for m in metrics if m in nested_scores.columns]
    lines.extend([
        "",
        "Every metric on the outer folds:",
        nested_scores[present_metrics].agg(["mean", "std"]).T.to_string(float_format=lambda v: f"{v:.3f}"),
        "",
        "Hyperparameters selected in each outer fold:",
    ])

    parameter_columns = [
        column for column in nested_scores.columns
        if column not in {"fold", "train", "validation", "gap", "inner_best", *metrics}
    ]
    if parameter_columns:
        lines.append(nested_scores[["fold"] + parameter_columns].to_string(index=False))
    else:
        lines.append("No hyperparameter columns were recorded.")

    return "\n".join(lines)


def tuning_report(study, nested_scores: pd.DataFrame, selection_metric: str, top: int = 5) -> str:
    """
    Best hyperparameters and top trials of the final study (on the whole development set),
    plus the optimism check (mean tuning score vs mean honest score over the outer folds),
    as text.
    """
    trials = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs"))
    trials = trials.rename(columns=lambda c: c.replace("params_", "").replace("user_attrs_", ""))
    trials = trials.rename(columns={
        "number": "trial",
        "value": f"mean {selection_metric}",
        "std": "fold_std",
    })

    # Only completed trials with a valid objective score.
    completed = trials.dropna(subset=[f"mean {selection_metric}"])
    top_trials = completed.sort_values(f"mean {selection_metric}", ascending=False).head(top)

    inner_mean = nested_scores["inner_best"].mean()
    outer_mean = nested_scores["validation"].mean()
    outer_std = nested_scores["validation"].std(ddof=1)
    gap_mean = nested_scores["gap"].mean()
    gap_std = nested_scores["gap"].std(ddof=1)

    lines = [
        "=" * 72,
        "OPTUNA HYPERPARAMETER TUNING",
        "=" * 72,
        f"Tuning on the whole development set ({len(completed)} completed trials, "
        f"metric: {selection_metric})",
        "",
        f"Best trial number: {study.best_trial.number}",
        f"Best mean CV score: {study.best_value:.3f}  (the maximum over all trials -- optimistic)",
        f"Best trial fold std: {study.best_trial.user_attrs.get('std', float('nan')):.3f}",
        f"Best hyperparameters: {study.best_params}",
        "",
        f"TOP {len(top_trials)} TRIALS",
        top_trials.to_string(index=False, float_format=lambda v: f"{v:.3f}"),
        "",
        "=" * 72,
        "OPTIMISM CHECK (NESTED CV)",
        "=" * 72,
        f"Tuning score inside the outer folds (inner_best) mean = {inner_mean:.3f}",
        f"Honest score on the outer folds (validation)     mean = {outer_mean:.3f} ± {outer_std:.3f}",
        f"Train-validation gap:                                   {gap_mean:+.3f} ± {gap_std:.3f}",
        f"Optimism (inner minus outer):                           {inner_mean - outer_mean:+.3f}",
        "",
        "Use the outer validation score to report tuned model performance.",
        "The best Optuna trial score is a selection score, not an unbiased estimate.",
    ]
    return "\n".join(lines)