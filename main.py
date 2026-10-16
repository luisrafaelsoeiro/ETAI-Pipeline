
"""
Entry point for the predictive pipeline.

Run with:
    python main.py

Workflow:
    load config -> load and clean data -> drop duplicate rows
    -> split features/target -> lock the final test set
    -> default stratified cross-validation and OOF reports
    -> optional Optuna tuning and nested cross-validation
    -> refit the selected pipeline on all development data
    -> evaluate once on the locked test set -> save results
"""

import yaml
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.data import load_data
from src.preprocessing import (
    clean_dataset,
    drop_duplicate_rows,
    split_features_target,
    build_preprocessor,
    split_dev_test,
)
from src.model import build_model
from src.evaluate import (
    cross_validate_pipeline,
    cv_report,
    oof_classification_report,
    fairness_report,
)
from src.results import save_run

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
    report = ""

    # ---------------------------------------------------------
    # 1. LOAD, CLEAN AND PREPARE DATA
    # ---------------------------------------------------------

    # Cleaning is row-preserving and does not learn preprocessing
    # parameters from the data.
    df_raw = load_data(config["data"]["path"])
    df_clean = clean_dataset(df_raw, config["diagnostics"])

    # Prevent duplicate records from being counted more than once.
    df_clean = drop_duplicate_rows(
        df_clean,
        config["diagnostics"].get("id_column"),
    )

    mnar_sources = config["preprocessing"].get(
        "mnar_indicator_sources", []
    )

    X, y, extras = split_features_target(
        df_clean,
        config["data"],
        mnar_sources,
    )

    # ---------------------------------------------------------
    # 2. LOCK THE FINAL TEST SET
    # ---------------------------------------------------------

    # The test set must not be used for model selection or tuning.
    X_dev, X_test, y_dev, y_test, extras_dev, extras_test = (
        split_dev_test(
            X,
            y,
            extras,
            test_size=config["test_set"]["size"],
            random_state=config["test_set"]["random_state"],
        )
    )

    print(f"Development rows: {len(X_dev)}")
    print(f"Locked test rows: {len(X_test)}")

    # ---------------------------------------------------------
    # 3. BUILD THE PIPELINE
    # ---------------------------------------------------------

    # Preprocessing stays inside the pipeline so that it is fitted
    # separately on each training fold during cross-validation.
    pipeline = Pipeline([
        (
            "prep",
            build_preprocessor(config["preprocessing"]),
        ),
        (
            "model",
            build_model(config["model"]),
        ),
    ])

    # ---------------------------------------------------------
    # 4. DEFAULT CROSS-VALIDATION
    # ---------------------------------------------------------

    cv_config = config["cv"]
    scoring = cv_config.get("scoring", "accuracy")
    n_splits = cv_config["n_splits"]
    shuffle = cv_config.get("shuffle", True)
    cv_random_state = cv_config.get("random_state", 42)

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=cv_random_state if shuffle else None,
    )

    fold_scores, y_oof = cross_validate_pipeline(
        pipeline=pipeline,
        X=X_dev,
        y=y_dev,
        cv=cv,
        scoring=scoring,
        n_jobs=cv_config.get("n_jobs", 1),
    )

    report += cv_report(fold_scores, scoring) + "\n\n"

    report += (
        oof_classification_report(y_dev, y_oof) + "\n\n"
    )

    report += fairness_report(
        y_dev,
        y_oof,
        extras_dev,
        sensitive_attr=config["data"]["sensitive_attr"],
    ) + "\n"

    # By default, the final model is the original, untuned pipeline.
    final_pipeline = pipeline

    # ---------------------------------------------------------
    # 5. OPTIONAL OPTUNA TUNING AND NESTED CROSS-VALIDATION
    # ---------------------------------------------------------

    tuning_config = config.get("tuning", {})
    tuning_enabled = tuning_config.get("enabled", False)

    if tuning_enabled:
        print("\n" + "=" * 72)
        print("HYPERPARAMETER TUNING AND NESTED CROSS-VALIDATION")
        print("=" * 72)

        # Retrieve the search space for the selected model.
        # Supported configuration layouts:
        #   tuning.search_space
        #   tuning.search_spaces[model_type]
        model_type = config["model"]["type"]
        search_spaces = tuning_config.get("search_spaces", {})

        if model_type in search_spaces:
            search_space = search_spaces[model_type]
        else:
            search_space = tuning_config.get("search_space", {})

        if not search_space:
            raise ValueError(
                "Optuna tuning is enabled, but no search space was "
                f"found for model '{model_type}'. Add either "
                "'tuning.search_space' or "
                f"'tuning.search_spaces.{model_type}' to config.yaml."
            )

        n_trials = tuning_config.get("n_trials", 10)
        tuning_random_state = tuning_config.get(
            "random_state", cv_random_state
        )
        tuning_n_jobs = tuning_config.get(
            "n_jobs", cv_config.get("n_jobs", 1)
        )

        # Use separate outer and inner CV objects.
        # Outer validation folds are not used to select their own
        # hyperparameters.
        nested_n_splits = tuning_config.get(
            "n_splits", n_splits
        )

        outer_cv = StratifiedKFold(
            n_splits=nested_n_splits,
            shuffle=True,
            random_state=tuning_random_state,
        )

        inner_cv = StratifiedKFold(
            n_splits=nested_n_splits,
            shuffle=True,
            random_state=tuning_random_state,
        )

        # 5a. Nested CV estimates the performance of the entire
        # tuning procedure on unseen development rows.
        nested_scores, _ = nested_cross_validate(
            pipeline=pipeline,
            X=X_dev,
            y=y_dev,
            outer_cv=outer_cv,
            inner_cv=inner_cv,
            scoring=scoring,
            search_space=search_space,
            n_trials=n_trials,
            random_state=tuning_random_state,
            n_jobs=tuning_n_jobs,
        )

        report += (
            "\n\n"
            + nested_cv_report(nested_scores, scoring)
            + "\n"
        )

        # 5b. Tune again on all development rows to select the
        # hyperparameters for the final model.
        final_pipeline, study = tune_pipeline(
            pipeline=pipeline,
            X=X_dev,
            y=y_dev,
            cv=inner_cv,
            scoring=scoring,
            search_space=search_space,
            n_trials=n_trials,
            random_state=tuning_random_state,
            n_jobs=tuning_n_jobs,
        )

        # 5c. Report the best Optuna trial and compare its score
        # with the honest nested-CV estimate.
        report += (
            "\n\n"
            + tuning_report(
                study=study,
                nested_scores=nested_scores,
                scoring=scoring,
                top=5,
            )
            + "\n"
        )

    # ---------------------------------------------------------
    # 6. REFIT THE SELECTED PIPELINE ON ALL DEVELOPMENT DATA
    # ---------------------------------------------------------

    final_model = final_pipeline.fit(X_dev, y_dev)

    refit = (
        f"Final model: {config['model']['type']} "
        f"{'(tuned)' if tuning_enabled else '(default)'} "
        f"refit on all {len(X_dev)} development rows."
    )

    print("\n" + refit)
    report += "\n" + refit + "\n"

    # ---------------------------------------------------------
    # 7. FINAL HOLDOUT EVALUATION
    # ---------------------------------------------------------

    # The locked test set is used only here, after model selection.
    y_test_pred = final_model.predict(X_test)
    holdout_accuracy = (y_test_pred == y_test).mean()

    holdout = (
        f"Holdout accuracy (locked test set): "
        f"{holdout_accuracy:.3f}"
    )

    print("\n" + holdout)
    report += "\n" + holdout + "\n"

    # Classification report on the locked test set.
    holdout_report = (
        "\nClassification report (locked holdout test set):\n"
        + classification_report(
            y_test,
            y_test_pred,
            zero_division=0,
        )
    )

    print(holdout_report)
    report += holdout_report + "\n"

    # Fairness report on the locked test set.
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

    print("\n" + locked)
    report += "\n" + locked + "\n"

    # ---------------------------------------------------------
    # 8. SAVE RESULTS
    # ---------------------------------------------------------

    results_dir = config.get("output", {}).get(
        "results_dir", "results"
    )

    output_path = save_run(results_dir, config, report)
    print(f"\nFull results saved to {output_path}")


if __name__ == "__main__":
    main()
