"""Saving each run's report to disk."""
import os
from datetime import datetime


def save_run(results_dir: str, config: dict, report_text: str) -> str:
    """
    Writes the run's settings and full report to:
    `results_dir/<model_abbreviation>_<timestamp>.txt`

    The folder is created if needed, and the file path is returned.
    """

    # Create the results directory if it does not already exist
    os.makedirs(results_dir, exist_ok=True)

    # Generate a timestamp for the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Get the model type from the configuration
    model_type = config["model"]["type"]

    # Convert the model name into a short abbreviation
    model_abbreviations = {
        "logistic_regression": "lg",
        "decision_tree": "dt",
        "random_forest": "rf",
        "dummy": "dummy",
    }

    # Use the abbreviation if it exists; otherwise keep the original model name
    model_abbreviation = model_abbreviations.get(model_type, model_type)

    # Build the output filename
    path = os.path.join(
        results_dir,
        f"{model_abbreviation}_{timestamp}.txt"
    )

    cv = config["cv"]
    tuning = config.get("tuning", {})

    tuning_line = (
        f"Optuna, {tuning['n_trials']} trials x {tuning['n_splits']} folds, "
        f"nested CV  random_state={tuning['random_state']}"
        if tuning.get("enabled")
        else "off"
    )

    header = (
        f"Run: {timestamp}\n"
        f"Model: {config['model']['type']}  "
        f"params={config['model'].get('params')}\n"
        f"Preprocessing: encoder={config['preprocessing']['encoder']}  "
        f"scaler={config['preprocessing']['scaler']}\n"
        f"CV: {cv['n_splits']} stratified folds  "
        f"shuffle={cv.get('shuffle', True)}  "
        f"random_state={cv.get('random_state')}  "
        f"metric={cv.get('scoring', 'accuracy')}\n"
        f"Tuning: {tuning_line}\n"
        f"Locked test set: size={config['test_set']['size']}  "
        f"random_state={config['test_set']['random_state']}  "
        f"(not evaluated)\n"
        + "=" * 60 + "\n\n"
    )

    # Write the header and report to the output file
    with open(path, "w") as f:
        f.write(header + report_text)

    return path