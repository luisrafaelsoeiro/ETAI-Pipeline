"""Saving results: each run's report to disk (`save_run`) and one row per run in the scoreboard (`save_scores`)."""
import os

import pandas as pd


def save_scores(path: str, row: dict) -> pd.DataFrame:
    """Appends one run to the CSV at `path` (created if needed) and returns all runs."""
    runs = pd.read_csv(path, dtype={"run": str}) if os.path.exists(path) else pd.DataFrame()
    runs = pd.concat([runs, pd.DataFrame([row])], ignore_index=True)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    runs.to_csv(path, index=False)
    return runs


def save_run(results_dir: str, run: str, config: dict, report_text: str, config_path: str = "config.yaml") -> str:
    """
    Writes the run's settings and full report to:
    `results_dir/<model_abbreviation>_<run>.txt`

    `run` is the run id built in `main.py` (a timestamp), the same id as the run's row on the
    scoreboard, so the report file and the scoreboard row can be matched.
    The folder is created if needed, and the file path is returned.
    """

    # Create the results directory if it does not already exist
    os.makedirs(results_dir, exist_ok=True)

    # Get the model type from the configuration
    model_type = config["model"]["type"]

    # Convert the model name into a short abbreviation
    model_abbreviations = {
        "logistic_regression": "lg",
        "decision_tree": "dt",
        "random_forest": "rf",
        "dummy": "dm",
        "oblique_random_forest": "ob_rf",
        "adaboost": "ab",
        "gradient_boosting": "gb",
        "hist_gradient_boosting": "hgb",
        "naive_bayes": "nb",
        "xgboost": "xgb",
    }

    # Use the abbreviation if it exists; otherwise keep the original model name
    model_abbreviation = model_abbreviations.get(model_type, model_type)

    # Build the output filename (same id as the scoreboard row)
    path = os.path.join(results_dir, f"{model_abbreviation}_{run}.txt")

    cv = config["cv"]
    tuning = config.get("tuning", {})
    test_set = config["test_set"]

    tuning_line = (
        # the same defaults main.py uses when a key is left out (week 9: no KeyError)
        f"Optuna, {tuning.get('n_trials', 10)} trials x {tuning.get('n_splits', cv['n_splits'])} folds, "
        f"nested CV  random_state={tuning.get('random_state', cv.get('random_state'))}  "
        f"(maximises {cv['selection_metric']})"
        if tuning.get("enabled")
        else "off"
    )

    header = (
        f"Run: {run}\n"
        f"Config: {config_path}   data: {config['data']['path']}   target: {config['data']['target']}\n"
        f"Model: {model_type}  "
        f"params={config['model'].get('params')}\n"
        f"Preprocessing: encoder={config['preprocessing']['encoder']}  "
        f"scaler={config['preprocessing']['scaler']}\n"
        f"CV: {cv['n_splits']} stratified folds  "
        f"shuffle={cv.get('shuffle', True)}  "
        f"random_state={cv.get('random_state')}\n"
        f"Metrics: {', '.join(cv['metrics'])}\n"
        f"Selection metric: {cv['selection_metric']}\n"
        f"Tuning: {tuning_line}\n"
        f"Locked test set: size={test_set['size']}  "
        f"random_state={test_set['random_state']}  "
        f"({'evaluated once' if test_set.get('evaluate', False) else 'not evaluated'})\n"
        + "=" * 60 + "\n\n"
    )

    # Write the header and report to the output file (utf-8: the report contains "±")
    with open(path, "w", encoding="utf-8") as f:
        f.write(header + report_text)

    return path