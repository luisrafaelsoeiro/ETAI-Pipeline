"""
Entry point for the predictive pipeline.

Run with:
    python main.py                      # COMPAS (config.yaml)
    python main.py config_adult.yaml    # any other problem, with its own config file

Workflow:
    load and check the config -> load and clean data -> check the config against the data
    -> drop duplicate rows and rows without a target -> split features/target
    -> encode the target labels as 0 .. k-1 -> lock the final test set
    -> no-model floor and default stratified cross-validation (every metric)
    -> optional Optuna tuning (on the selection metric) and nested cross-validation
    -> out-of-fold reports (confusion matrix, classification report, fairness)
    -> refit the selected pipeline on all development data
    -> predict the new, unlabelled rows (only when the config has a `predict` section)
    -> add the run to the scoreboard -> save results
    -> (only when test_set.evaluate is true) evaluate once on the locked test set
"""
import argparse
import shutil
import tempfile
from datetime import datetime
import optuna
from sklearn.model_selection import StratifiedKFold

from src.config import check_dataset, load_config, search_space_for
from src.data import load_data
from src.preprocessing import (
    check_feature_columns,
    clean_dataset,
    drop_duplicate_rows,
    drop_missing_target,
    encode_target,
    split_features_target,
    split_dev_test,
)
from src.predict import predict_new_data, save_predictions
from src.model import build_pipeline
from src.evaluate import (
    build_scorers,
    cross_validate_pipeline,
    cv_report,
    holdout_report,
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


def parse_args():
    parser = argparse.ArgumentParser(description="Run the predictive pipeline for one problem.")
    parser.add_argument("config", nargs="?", default="config.yaml",
                        help="config file of the problem to run (default: config.yaml, COMPAS)")
    return parser.parse_args()


def run(config_path: str = "config.yaml") -> dict:
    """
    Runs the whole pipeline for the problem described by `config_path` and returns the main
    results (report path, scores, classes, predictions path). The config is checked on its own
    and then against the data before anything is fitted (`src/config.py`): a wrong setting stops
    the run at the start with every problem listed.
    """
    config = load_config(config_path)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_type = config["model"]["type"]
    sensitive_attr = config["data"].get("sensitive_attr")      # optional: no fairness report without it
    reference_score = config["data"].get("reference_score")   # e.g. COMPAS's own score, for the fairness report
    report = ""
    print(f"Config: {config_path}")

    # ---------------------------------------------------------
    # 1. LOAD, CLEAN AND PREPARE DATA
    # ---------------------------------------------------------

    # Cleaning is row-preserving and does not learn preprocessing
    # parameters from the data.
    df_raw = load_data(config["data"]["path"], **config["data"]["read_options"])
    df_clean = clean_dataset(df_raw, config["diagnostics"])

    # The config against the data: target, columns, classes, metrics (errors stop the run here).
    for warning in check_dataset(df_clean, config, config_path):
        print(f"Note: {warning}")
        report += f"Note: {warning}\n"

    # Training only: duplicate records counted once; rows without a target can't be learned from.
    df_clean = drop_duplicate_rows(df_clean, config["diagnostics"].get("id_column"))
    df_clean = drop_missing_target(df_clean, config["data"]["target"])

    mnar_sources = config["preprocessing"].get("mnar_indicator_sources") or []
    X, y, extras = split_features_target(df_clean, config["data"], mnar_sources)

    # Every listed feature must exist; a column listed nowhere would be dropped silently.
    unused = check_feature_columns(X, config["preprocessing"])
    if unused:
        print(f"Columns not listed as features (left out of the model): {unused}")
        report += f"Columns not listed as features (left out of the model): {unused}\n"

    # Labels -> 0 .. k-1 (any labels, two or more classes); `classes` keeps the original names.
    y, classes = encode_target(y, config["data"].get("positive_class"))
    target_line = (f"Target: {config['data']['target']}   classes: {classes}"
                   + (f"   (positive class: {classes[1]!r})" if len(classes) == 2 else ""))
    print(target_line)
    report += target_line + "\n"

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
    # Feature selection (optional) sits between preprocessing and the model, inside the pipeline.
    selection_config = config.get("feature_selection") or {}
    selection_enabled = selection_config.get("enabled", False)
    selection_label = (f"VoteSelector(min_votes={selection_config.get('min_votes', 2)})"
                       if selection_enabled else "off")
    print(f"Feature selection: {selection_label}")

    # Fitted prep/select steps are cached in a fresh folder for this run (deleted at the end),
    # so tuning the model doesn't refit them in every trial.
    cache_dir = tempfile.mkdtemp(prefix="pipeline_cache_")
    try:
        return _cross_validate_tune_refit(
            config, config_path, run_id, report, cache_dir, X_dev, X_test, y_dev, y_test,
            extras_dev, extras_test, classes, cv, scorers, metrics, selection_metric, n_jobs,
            selection_config, selection_enabled, selection_label, sensitive_attr, reference_score,
        )
    finally:
        # removed even when the run fails (week 9: it used to be left behind on an error)
        shutil.rmtree(cache_dir, ignore_errors=True)


def _cross_validate_tune_refit(config, config_path, run_id, report, cache_dir, X_dev, X_test, y_dev, y_test,
                               extras_dev, extras_test, classes, cv, scorers, metrics, selection_metric,
                               n_jobs, selection_config, selection_enabled, selection_label,
                               sensitive_attr, reference_score) -> dict:
    """Steps 4-10 of `run`: CV, optional tuning, reports, refit, predictions, scoreboard, saving."""
    model_type = config["model"]["type"]
    cv_config = config["cv"]
    pipeline = build_pipeline(config["preprocessing"], config["model"], selection_config, memory=cache_dir)


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

    print(f"\nDefault hyperparameters ({config_path})")
    print(f"  {selection_metric}: {untuned:.3f} ± {fold_scores['validation'].std(ddof=1):.3f}   "
          f"train-validation gap: {fold_scores['gap'].mean():+.3f}   no-model floor: {floor:.3f}")

    report += (
        f"Feature selection: {selection_label}\n"
        f"Hyperparameters from {config_path}: {config['model'].get('params')}\n\n"
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
        # The model's space, plus how strict the selection is when selection is on (checked
        # by load_config: every parameter name exists in the pipeline).
        search_space = search_space_for(config)

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

    report += "\n\n" + oof_classification_report(y_dev, oof_proba, classes=classes)
    report += "\n" + fairness_report(y_dev, oof_proba, extras_dev, sensitive_attr=sensitive_attr,
                                     reference=reference_score, classes=classes)

    # ---------------------------------------------------------
    # 7. REFIT THE SELECTED PIPELINE ON ALL DEVELOPMENT DATA
    # ---------------------------------------------------------

    final_model = final_pipeline.fit(X_dev, y_dev)

    # The features the final model keeps (the select step's output names).
    kept_line = ""
    if "select" in final_model.named_steps:
        try:
            kept = final_model[:-1].get_feature_names_out()
            kept_line = f"Features kept ({len(kept)}): {', '.join(kept)}"
        except (AttributeError, ValueError):       # a step without feature names
            kept_line = f"Features kept: {int(final_model.named_steps['select'].support_.sum())} (names unavailable)"

    refit = (f"Final model: {model_type} {'(tuned)' if tuning_enabled else '(default)'} "
             f"refit on all {len(X_dev)} development rows.")
    report += "\n" + refit + "\n"
    
    if kept_line:
        report += kept_line + "\n"

    # ---------------------------------------------------------
    # 7b. PREDICT NEW, UNLABELLED ROWS (only with a `predict` section)
    # ---------------------------------------------------------

    # Same cleaning and feature split as the training data, no row dropped: rows out = rows in.
    predict_config = config.get("predict")
    saved = None
    if predict_config:
        new_raw = load_data(predict_config["path"],
                            **(predict_config.get("read_options") or config["data"]["read_options"]))
        predictions = predict_new_data(final_model, new_raw, config, predict_config.get("threshold", 0.5), classes)
        saved = save_predictions(predictions, predict_config["output"])
        shares = predictions["prediction"].value_counts(normalize=True)
        shown = (f"{shares.get(classes[1], 0.0):.1%} predicted {classes[1]}" if len(classes) == 2
                 else ", ".join(f"{name}: {shares.get(name, 0.0):.1%}" for name in classes))
        predicted_line = (f"Predicted {len(predictions)} new rows ({len(new_raw)} in) from {predict_config['path']}: "
                          f"{shown} -> {saved}")
        print(predicted_line)
        report += predicted_line + "\n"

    # ---------------------------------------------------------
    # 8. SCOREBOARD
    # ---------------------------------------------------------

    row = {
        "run": run_id,
        "data": config["data"]["path"],
        "target": config["data"]["target"],
        "config": config_path,
        "model": model_type,
        "tuned": tuning_enabled,
        "encoder": config["preprocessing"]["encoder"],
        "scaler": config["preprocessing"]["scaler"],
        "feature_selection": selection_label,
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
    # runs on the same data, target and metric (rows saved before week 9 have no target column)
    same_target = runs["target"].isna() | (runs["target"] == row["target"])
    same = runs[(runs["data"] == row["data"]) & same_target & (runs["metric"] == selection_metric)]
    board = (
        same.sort_values("mean", ascending=False)
        [["run", "model", "tuned", "encoder", "scaler", "feature_selection", "mean", "std", "untuned", "floor"]]        .rename(columns={"mean": selection_metric})
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
        holdout, _, _, _ = holdout_report(final_model, X_test, y_test, extras_test, classes,
                                          sensitive_attr, reference=reference_score)
        print("\n" + holdout)
        report += "\n" + holdout + "\n"

    # ---------------------------------------------------------
    # 10. SAVE RESULTS
    # ---------------------------------------------------------

    path = save_run(config.get("output", {}).get("results_dir", "results"), run_id, config, report, config_path)
    print(f"\n{refit}\nFull report (per-fold tables, tuning, confusion matrix, fairness): {path}")

    if kept_line:
        print(f"\n{kept_line}")
    return {"run": run_id, "report": path, "scores": final_scores, "untuned": untuned, "floor": floor,
            "classes": classes, "predictions": saved, "kept": kept_line}


def main():
    run(parse_args().config)


if __name__ == "__main__":
    main()