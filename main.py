"""
Entry point for the predictive pipeline.

Run with:
    python main.py

Workflow:
    load config -> load and clean data -> drop duplicate rows
    -> split features/target -> lock the final test set
    -> no-model floor and default stratified cross-validation (every metric)
    -> optional Optuna tuning (on the selection metric) and nested cross-validation
    -> out-of-fold reports (confusion matrix, classification report, fairness)
    -> refit the selected pipeline on all development data
    -> add the run to the scoreboard -> save results
    -> (only when test_set.evaluate is true) evaluate once on the locked test set
"""
from datetime import datetime

import numpy as np
import optuna
import pandas as pd
import yaml
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.data import load_data
from src.preprocessing import (
    clean_dataset,
    drop_duplicate_rows,
    split_features_target,
    split_dev_test,
)
from src.model import build_pipeline
from src.evaluate import (
    build_scorers,
    cross_validate_pipeline,
    cv_report,
    oof_classification_report,
    fairness_report,
)
from src.results import save_run, save_scores
from src.tuning import (
    tune_pipeline,
    nested_cross_validate,
    tuning_report,
    nested_cv_report,
)


def load_config(path: str = "config.yaml") -> dict:
    """Load the project configuration."""
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main():
    config = load_config()
    run = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_type = config["model"]["type"]
    sensitive_attr = config["data"]["sensitive_attr"]
    report = ""

    # ---------------------------------------------------------
    # 1. LOAD, CLEAN AND PREPARE DATA
    # ---------------------------------------------------------

    # Cleaning is row-preserving and does not learn preprocessing
    # parameters from the data.
    df_raw = load_data(config["data"]["path"])
    df_clean = clean_dataset(df_raw, config["diagnostics"])

    # Prevent duplicate records from being counted more than once.
    df_clean = drop_duplicate_rows(df_clean, config["diagnostics"].get("id_column"))

    mnar_sources = config["preprocessing"].get("mnar_indicator_sources", [])
    X, y, extras = split_features_target(df_clean, config["data"], mnar_sources)

    # ---------------------------------------------------------
    # 2. LOCK THE FINAL TEST SET
    # ---------------------------------------------------------

    # From here on, every decision uses the development set only.
    X_dev, X_test, y_dev, y_test, extras_dev, extras_test = split_dev_test(
        X, y, extras,
        test_size=config["test_set"]["size"],
        random_state=config["test_set"]["random_state"],
    )

    # ---------------------------------------------------------
    # 3. CV FOLDS, METRICS AND SCORERS
    # ---------------------------------------------------------

    cv_config = config["cv"]
    shuffle = cv_config.get("shuffle", True)
    cv = StratifiedKFold(
        n_splits=cv_config["n_splits"],
        shuffle=shuffle,
        random_state=cv_config.get("random_state") if shuffle else None,
    )
    metrics = cv_config["metrics"]
    selection_metric = cv_config["selection_metric"]
    if selection_metric not in metrics:
        raise ValueError(f"cv.selection_metric '{selection_metric}' must be one of cv.metrics: {metrics}")
    scorers = build_scorers(metrics)
    n_jobs = cv_config.get("n_jobs", 1)

    print(f"Model: {model_type}   |   selection metric: {selection_metric}   |   "
          f"{cv_config['n_splits']}-fold CV")
    print(f"Data:  {len(X_dev)} development rows   |   {len(X_test)} test rows locked away")

    # ---------------------------------------------------------
    # 4. NO-MODEL FLOOR AND DEFAULT CROSS-VALIDATION
    # ---------------------------------------------------------

    # Preprocessing stays inside the pipeline so that it is fitted
    # separately on each training fold during cross-validation.
    pipeline = build_pipeline(config["preprocessing"], config["model"])

    # The floor: a dummy classifier on the same folds (what "no model" scores).
    floor_pipeline = build_pipeline(config["preprocessing"], {"type": "dummy"})
    floor_scores, _ = cross_validate_pipeline(
        floor_pipeline, X_dev, y_dev, cv,
        {selection_metric: scorers[selection_metric]}, selection_metric, n_jobs,
    )
    fold_scores, oof_proba = cross_validate_pipeline(
        pipeline, X_dev, y_dev, cv, scorers, selection_metric, n_jobs,
    )
    floor = floor_scores["validation"].mean()
    untuned = fold_scores["validation"].mean()

    print("\nDefault hyperparameters (config.yaml)")
    print(f"  {selection_metric}: {untuned:.3f} ± {fold_scores['validation'].std(ddof=1):.3f}   "
          f"train-validation gap: {fold_scores['gap'].mean():+.3f}   no-model floor: {floor:.3f}")

    report += (
        f"Hyperparameters from config.yaml: {config['model'].get('params')}\n\n"
        + cv_report(fold_scores, metrics, selection_metric, "Cross-validation (default hyperparameters)")
        + f"\n\nNo-model floor (dummy, same folds): {selection_metric} = {floor:.3f}"
    )

    # By default, the final model is the untuned pipeline and its CV scores.
    final_pipeline = pipeline
    final_scores = fold_scores

    # ---------------------------------------------------------
    # 5. OPTIONAL OPTUNA TUNING AND NESTED CROSS-VALIDATION
    # ---------------------------------------------------------

    tuning_config = config.get("tuning", {})
    tuning_enabled = tuning_config.get("enabled", False)
    study = None

    if tuning_enabled:
        # Supported configuration layouts:
        #   tuning.search_spaces[model_type]   (preferred)
        #   tuning.search_space
        search_spaces = tuning_config.get("search_spaces") or {}
        search_space = search_spaces.get(model_type) or tuning_config.get("search_space") or {}
        if not search_space:
            raise ValueError(
                f"Optuna tuning is enabled, but no search space was found for model '{model_type}'. "
                f"Add 'tuning.search_spaces.{model_type}' to config.yaml, or set tuning.enabled: false."
            )

        n_trials = tuning_config.get("n_trials", 10)
        tuning_seed = tuning_config.get("random_state", cv_config.get("random_state"))
        tuning_n_jobs = tuning_config.get("n_jobs", n_jobs)
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Inner folds choose the hyperparameters. The OUTER folds are the same `cv` folds as the
        # default run, so the tuned and untuned scores (and every run on the scoreboard) are paired.
        inner_cv = StratifiedKFold(
            n_splits=tuning_config.get("n_splits", cv_config["n_splits"]),
            shuffle=True,
            random_state=tuning_seed,
        )

        # 5a. Nested CV: honest estimate of the whole tuning procedure. Its out-of-fold
        # probabilities feed the reports below.
        print(f"\nTuning ({n_trials} trials, nested CV) ... this takes a while", flush=True)
        final_scores, oof_proba = nested_cross_validate(
            pipeline, X_dev, y_dev, cv, inner_cv, scorers, selection_metric,
            search_space, n_trials, tuning_seed, n_jobs=tuning_n_jobs,
        )
        print(f"  {selection_metric}: {final_scores['validation'].mean():.3f} ± "
              f"{final_scores['validation'].std(ddof=1):.3f}   (honest estimate of the tuned model)")

        report += "\n\n" + cv_report(
            final_scores, metrics, selection_metric,
            "Nested cross-validation (the tuning procedure, estimated honestly)",
        )
        report += "\n\n" + nested_cv_report(final_scores, metrics, selection_metric)

        # 5b. Tune again on all development rows to choose the final hyperparameters.
        final_pipeline, study = tune_pipeline(
            pipeline, X_dev, y_dev, inner_cv, scorers[selection_metric],
            search_space, n_trials, tuning_seed, n_jobs=tuning_n_jobs,
        )
        print("  Chosen hyperparameters:")
        for name, value in study.best_params.items():
            shown = f"{value:.4g}" if isinstance(value, float) else value
            print(f"    {name.replace('model__', ''):<30s} {shown}")

        # 5c. Best Optuna trial vs the honest nested-CV estimate.
        report += "\n\n" + tuning_report(study, final_scores, selection_metric)

    # ---------------------------------------------------------
    # 6. EVERY METRIC AND OUT-OF-FOLD REPORTS
    # ---------------------------------------------------------

    print(f"\nAll metrics ({'tuned model, nested CV' if tuning_enabled else 'default hyperparameters'}, "
          f"mean ± std)")
    for name in metrics:
        print(f"  {name:<18s} {final_scores[name].mean():.3f} ± {final_scores[name].std(ddof=1):.3f}")

    report += "\n\n" + oof_classification_report(y_dev, oof_proba)
    report += "\n" + fairness_report(y_dev, oof_proba, extras_dev, sensitive_attr=sensitive_attr)

    # ---------------------------------------------------------
    # 7. REFIT THE SELECTED PIPELINE ON ALL DEVELOPMENT DATA
    # ---------------------------------------------------------

    final_model = final_pipeline.fit(X_dev, y_dev)
    refit = (f"Final model: {model_type} {'(tuned)' if tuning_enabled else '(default)'} "
             f"refit on all {len(X_dev)} development rows.")
    report += "\n" + refit + "\n"

    # ---------------------------------------------------------
    # 8. SCOREBOARD
    # ---------------------------------------------------------

    row = {
        "run": run,
        "data": config["data"]["path"],
        "model": model_type,
        "tuned": tuning_enabled,
        "encoder": config["preprocessing"]["encoder"],
        "scaler": config["preprocessing"]["scaler"],
        "params": str(config["model"].get("params")),
        "tuned_params": str(study.best_params) if tuning_enabled else "",
        "metric": selection_metric,
        "mean": final_scores["validation"].mean(),
        "std": final_scores["validation"].std(ddof=1),
        "untuned": untuned,
        "floor": floor,
        **{f"fold_{i}": score for i, score in enumerate(final_scores["validation"], start=1)},
    }
    runs = save_scores(config.get("output", {}).get("scoreboard", "results/runs.csv"), row)
    same = runs[(runs["data"] == row["data"]) & (runs["metric"] == selection_metric)]
    board = (
        same.sort_values("mean", ascending=False)
        [["run", "model", "tuned", "encoder", "scaler", "mean", "std", "untuned", "floor"]]
        .rename(columns={"mean": selection_metric})
        .to_string(index=False, float_format=lambda v: f"{v:.3f}")
    )
    print(f"\nScoreboard (runs on the same data and metric; best first)\n{board}")
    report += f"\nScoreboard:\n{board}\n"

    # ---------------------------------------------------------
    # 9. FINAL HOLDOUT EVALUATION (only when explicitly switched on)
    # ---------------------------------------------------------

    # The locked test set is used once, for the final chosen model, never while comparing
    # models. Keep test_set.evaluate: false for every scoreboard run.
    if config["test_set"].get("evaluate", False):
        y_test_proba = final_model.predict_proba(X_test)[:, 1]
        y_test_pred = (y_test_proba >= 0.5).astype(int)
        cm = pd.DataFrame(confusion_matrix(y_test, y_test_pred), index=["true 0", "true 1"],
                          columns=["predicted 0", "predicted 1"])
        holdout = (
            f"Locked test set ({len(X_test)} rows, evaluated once, threshold 0.5)\n"
            f"  accuracy: {(y_test_pred == np.asarray(y_test)).mean():.3f}   "
            f"roc_auc: {roc_auc_score(y_test, y_test_proba):.3f}\n\n"
            f"{cm.to_string()}\n\n"
            f"{classification_report(y_test, y_test_pred, zero_division=0)}\n"
            "Fairness on the locked test set (header says out-of-fold; these are test rows):\n"
            + fairness_report(y_test, y_test_proba, extras_test, sensitive_attr=sensitive_attr)
        )
        print("\n" + holdout)
        report += "\n" + holdout + "\n"

    # ---------------------------------------------------------
    # 10. SAVE RESULTS
    # ---------------------------------------------------------

    path = save_run(config.get("output", {}).get("results_dir", "results"), run, config, report)
    print(f"\n{refit}\nFull report (per-fold tables, tuning, confusion matrix, fairness): {path}")


if __name__ == "__main__":
    main()