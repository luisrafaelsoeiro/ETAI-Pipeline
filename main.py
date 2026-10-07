"""
Entry point: `python main.py`.

load config -> load and clean data -> drop duplicate rows (training only) -> features/target
-> lock the test set -> cross-validate the pipeline on the development set
-> [if tuning is enabled: nested CV of the tuning procedure, then tuning on all development rows]
-> out-of-fold classification and fairness reports -> refit on all development rows -> save the report
"""
import optuna
import yaml
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

from src.data import load_data
from src.preprocessing import (
    clean_dataset, drop_duplicate_rows, split_features_target, build_preprocessor, split_dev_test,
)
from src.model import build_model
from src.evaluate import (
    cross_validate_pipeline, cv_report, oof_classification_report, fairness_report, holdout_evaluation
)
from src.results import save_run


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    df_raw = load_data(config["data"]["path"])
    df_clean = clean_dataset(df_raw, config["diagnostics"])
    df_clean = drop_duplicate_rows(df_clean, config["diagnostics"].get("id_column"))

    mnar_sources = config["preprocessing"].get("mnar_indicator_sources", [])
    X, y, extras = split_features_target(df_clean, config["data"], mnar_sources)

    # From here on, every decision uses the development set only; the test set stays locked.
    X_dev, X_test, y_dev, y_test, extras_dev, extras_test = split_dev_test(
        X, y, extras,
        test_size=config["test_set"]["size"],
        random_state=config["test_set"]["random_state"],
    )

    pipeline = build_pipeline(config["preprocessing"], config["model"])

    cv_config = config["cv"]
    shuffle = cv_config.get("shuffle", True)
    cv = StratifiedKFold(n_splits=cv_config["n_splits"], shuffle=shuffle,
                         random_state=cv_config.get("random_state") if shuffle else None)
    scoring = cv_config.get("scoring", "accuracy")
    n_jobs = cv_config.get("n_jobs", 1)

    fold_scores, y_oof = cross_validate_pipeline(pipeline, X_dev, y_dev, cv, scoring, n_jobs=n_jobs)
    header = f"Hyperparameters from config.yaml: {config['model'].get('params')}"
    print(header)
    report = header + "\n" + cv_report(fold_scores, scoring)

    tuning_config = config.get("tuning", {})
    tuning_enabled = tuning_config.get("enabled", False)
    if tuning_enabled:
        model_type = config["model"]["type"]
        search_spaces = tuning_config.get("search_spaces") or {}
        if model_type not in search_spaces:
            raise ValueError(f"tuning is enabled but config.yaml has no tuning.search_spaces for "
                             f"'{model_type}'. Add one, or set tuning.enabled: false.")
        search_space = search_spaces[model_type]
        n_trials = tuning_config["n_trials"]
        tuning_seed = tuning_config["random_state"]
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        inner_cv = StratifiedKFold(n_splits=tuning_config["n_splits"], shuffle=True, random_state=tuning_seed)

        # Honest estimate on the same outer folds; its out-of-fold predictions feed the reports.
        print(f"\nNested cross-validation: {cv.get_n_splits()} outer folds x {n_trials} trials x "
              f"{inner_cv.get_n_splits()} inner folds ...")
        nested_scores, y_oof = nested_cross_validate(
            pipeline, X_dev, y_dev, cv, inner_cv, scoring, search_space, n_trials, tuning_seed, n_jobs=n_jobs
        )
        report += "\n\nNested cross-validation (the tuning procedure, estimated honestly):\n"
        report += cv_report(nested_scores, scoring)

        # The hyperparameters that are kept: the same procedure, once, on all development rows.
        pipeline, study = tune_pipeline(
            pipeline, X_dev, y_dev, inner_cv, scoring, search_space, n_trials, tuning_seed, n_jobs=n_jobs
        )
        print()
        report += "\n\n" + tuning_report(study, nested_scores, scoring)

    report += "\n\n" + oof_classification_report(y_dev, y_oof)
    report += "\n" + fairness_report(y_dev, y_oof, extras_dev, sensitive_attr=config["data"]["sensitive_attr"])

    # the model we'd actually use: same pipeline, refit on EVERY development row. CV above
    # estimated how well this recipe does; it didn't produce a model.
    final_model = pipeline.fit(X_dev, y_dev)

    refit = (
        f"Final model: {config['model']['type']} "
        f"refit on all {len(X_dev)} development rows."
    )
    print(refit)
    report += "\n" + refit + "\n"

    # FINAL HOLDOUT EVALUATION:
    # Predict the locked test set exactly once, after the model has been finalized.
    y_test_pred = final_model.predict(X_test)
    holdout_accuracy = (y_test_pred == y_test).mean()

    holdout = (
        f"Holdout accuracy (locked test set): {holdout_accuracy:.3f}"
    )
    print(holdout)
    report += "\n" + holdout + "\n"

    # Classification report on the locked holdout set
    holdout_report = (
        "\nClassification report (locked holdout test set):\n"
        + classification_report(y_test, y_test_pred, zero_division=0)
    )
    print(holdout_report)
    report += holdout_report + "\n"

    # Fairness report on the locked holdout set
    holdout_fairness = fairness_report(
        y_test,
        y_test_pred,
        extras_test,
        sensitive_attr=config["data"]["sensitive_attr"],
    )
    report += "\n" + holdout_fairness + "\n"

    locked = (
        f"Locked test set: {len(X_test)} rows evaluated once. "
        f"Development set: {len(X_dev)} rows."
    )
    print(locked)
    report += "\n" + locked + "\n"


    results_dir = config.get("output", {}).get("results_dir", "results")
    path = save_run(results_dir, config, report)
    print(f"Full results saved to {path}")


if __name__ == "__main__":
    main()
