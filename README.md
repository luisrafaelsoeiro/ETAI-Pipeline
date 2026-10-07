# Baseline Predictive Pipeline -- ETAI
Luis Soeiro 20211536

**CONCLUSIONS SUMMARY**

The logistic regression model performed slightly better overall, achieving a higher test accuracy than the decision tree (0.677 vs. 0.668). Both models have relatively small differences between training and test accuracy, with gaps of 0.001 and 0.012 respectively, indicating no clear signs of overfitting.

Looking at the classification metrics in more detail, the two models exhibit different performance characteristics. Logistic regression performs better at identifying non-recidivists (class 0), achieving a higher recall (0.74 vs. 0.68) and F1-score (0.72 vs. 0.69). In contrast, the decision tree performs better at identifying recidivists (class 1), with higher recall (0.65 vs. 0.60) and a slightly higher F1-score (0.64 vs. 0.63). Therefore, the decision tree improves the detection of recidivism at the expense of performance on non-recidivists, rather than providing an overall improvement. Nevertheless, it is important to understand that only a simple model was used on noth logistic regression and decision tree, meaning no optimal / different parameters were considered to undertand which model could further improve the results

Regarding the fairness analysis, the results cannot yet be considered reliable because the race variable contains inconsistent representations of the same categories. The race categories should therefore be standardized and the false-positive rates recalculated before drawing conclusions about differences between the models and COMPAS.

| Model               | Holdout accuracy (W3) | CV accuracy (mean ± std) |   CV train–val gap |
| ------------------- | --------------------: | -----------------------: | -----------------: |
| Dummy               |             **0.550** |        **0.549 ± 0.000** | **−0.000 ± 0.000** |
| Logistic regression |             **0.661** |        **0.673 ± 0.013** | **+0.001 ± 0.015** |
| Decision tree       |             **0.619** |        **0.608 ± 0.014** | **+0.094 ± 0.020** |
| Random forest       |                 **—** |        **0.645 ± 0.020** | **+0.093 ± 0.028** |

I would trust Logistic regression's mean cross-validation accuracy of 0.673. Its accuracy on the locked holdout set was 0.661 with a very small train–validation gap of 0.001 provides a similar estimate of generalization performance.

Yes — if your assignment’s Week 3 holdout accuracy was supposed to be recorded before the Random Forest was introduced, then you should not backfill the Random Forest holdout accuracy into that table.

The distinction is:

Dummy, Logistic Regression, Decision Tree: have the original Week 3 holdout accuracy → report those values.
Random Forest: if it was not evaluated on the holdout set as part of Week 3 → use — or N/A, even though you now technically have a holdout accuracy from the Week 4 run.
The Random Forest's CV results can still be reported because those are part of the current Week 4 evaluation.

So the table should be:

Model	Holdout accuracy (W3)	CV accuracy (mean ± std)	CV train–val gap
Dummy	0.550	0.549 ± 0.000	−0.000 ± 0.000
Logistic regression	0.661	0.673 ± 0.013	+0.001 ± 0.015
Decision tree	0.619	0.608 ± 0.014	+0.094 ± 0.020
Random forest	—	0.645 ± 0.020	+0.093 ± 0.028

The 0.636 Random Forest holdout accuracy should be kept in your Week 4 results, but not inserted into a column explicitly labelled “Holdout accuracy (W3)”.

If this table is specifically meant to reproduce the Week 3 → Week 4 comparison, then — is the correct value for Random Forest.

which number would you trust out of all of those model results, and why?

For this experiment, I would trust the 5-fold cross-validation validation accuracy more than any single holdout number when judging how the models generalize, because it averages performance across five different validation splits.

Your results:

Model	CV validation accuracy	Holdout accuracy	Train–validation gap
Dummy	0.549 ± 0.000	0.550	~0.000
Logistic regression	0.673 ± 0.013	0.661	+0.001
Decision tree	0.608 ± 0.014	0.619	+0.094
Random forest	0.645 ± 0.020	0.636	+0.093
The number I'd put the most weight on

Logistic regression: 0.673 ± 0.013 CV accuracy.

Why:

It is based on five validation folds, rather than one particular test split.
Its validation performance is relatively consistent (±0.013).
Most importantly, its train–validation gap is only +0.001, meaning there isn't much evidence of overfitting.
Its holdout accuracy, 0.661, is reasonably close to the CV estimate of 0.673. That gives some consistency between the development-set evaluation and the locked test set.

The Random Forest's 0.636 holdout accuracy is still a legitimate result, but its CV result comes with a much larger train–validation gap (+0.093), which indicates substantially more overfitting than logistic regression.

One important distinction

I wouldn't say 0.673 is "the true accuracy." It's an estimate of expected performance based on your five CV validation folds. The 0.661 holdout accuracy is an independent final estimate from the locked test set.

So for a report, I'd phrase it roughly as:

Logistic regression achieved a mean cross-validation accuracy of 0.673 ± 0.013, with a very small train–validation gap of 0.001. Its accuracy on the locked holdout set was 0.661, providing a similar estimate of generalisation performance.

That is a much stronger interpretation than simply choosing whichever single accuracy number is highest.

3–5 sentences: Does your week 2/3 "best model" conclusion still hold under cross-validation?

Models:
Run: 20260923_094943 
Model: decision_tree  params={'max_depth': 15} 
Test size: 0.2  random_state: 42 
============================================================ 
 
Train accuracy: 0.770 
Test accuracy:  0.621 
Gap (train - test): +0.149 
 
Classification report (test set): 
              precision    recall  f1-score   support 
 
           0       0.63      0.74      0.68       793 
           1       0.60      0.48      0.53       650 
 
    accuracy                           0.62      1443 
   macro avg       0.62      0.61      0.61      1443 
weighted avg       0.62      0.62      0.61      1443 
 
False positive rate by race 
(share of people who did NOT reoffend, but were predicted to) 
 
  Our model: 
    African-American     FPR = 0.32  (n=349) 
    Asian                FPR = 0.00  (n=2) 
    Caucasian            FPR = 0.23  (n=290) 
    Hispanic             FPR = 0.19  (n=85) 
    Native American      FPR = 0.00  (n=1) 
    Other                FPR = 0.26  (n=54) 
 
  COMPAS's own score: 
    African-American     FPR = 0.44  (n=349) 
    Asian                FPR = 0.00  (n=2) 
    Caucasian            FPR = 0.24  (n=290) 
    Hispanic             FPR = 0.16  (n=85) 
    Native American      FPR = 1.00  (n=1) 
    Other                FPR = 0.20  (n=54) 
Run: 20260929_141153 
Model: logistic_regression  params={'max_iter': 2000} 
Test size: 0.2  random_state: 42 
============================================================ 
 
Train accuracy: 0.676 
Test accuracy:  0.657 
Gap (train - test): +0.019 
 
Classification report (test set): 
              precision    recall  f1-score   support 
 
           0       0.65      0.80      0.72       793 
           1       0.66      0.48      0.56       650 
 
    accuracy                           0.66      1443 
   macro avg       0.66      0.64      0.64      1443 
weighted avg       0.66      0.66      0.65      1443 
 
False positive rate by race 
(share of people who did NOT reoffend, but were predicted to) 
 
  Our model: 
    African-American     FPR = 0.28  (n=349) 
    Asian                FPR = 0.00  (n=2) 
    Caucasian            FPR = 0.14  (n=290) 
    Hispanic             FPR = 0.11  (n=85) 
    Native American      FPR = 0.00  (n=1) 
    Other                FPR = 0.19  (n=54) 
 
  COMPAS's own score: 
    African-American     FPR = 0.44  (n=349) 
    Asian                FPR = 0.00  (n=2) 
    Caucasian            FPR = 0.24  (n=290) 
    Hispanic             FPR = 0.16  (n=85) 
    Native American      FPR = 1.00  (n=1) 
    Other                FPR = 0.20  (n=54) 
conclusions:
The logistic regression model performed slightly better overall, achieving a higher test accuracy than the decision tree (0.677 vs. 0.668). Both models have relatively small differences between training and test accuracy, with gaps of 0.001 and 0.012 respectively, indicating no clear signs of overfitting.

Looking at the classification metrics in more detail, the two models exhibit different performance characteristics. Logistic regression performs better at identifying non-recidivists (class 0), achieving a higher recall (0.74 vs. 0.68) and F1-score (0.72 vs. 0.69). In contrast, the decision tree performs better at identifying recidivists (class 1), with higher recall (0.65 vs. 0.60) and a slightly higher F1-score (0.64 vs. 0.63). Therefore, the decision tree improves the detection of recidivism at the expense of performance on non-recidivists, rather than providing an overall improvement. Nevertheless, it is important to understand that only a simple model was used on noth logistic regression and decision tree, meaning no optimal / different parameters were considered to undertand which model could further improve the results

Regarding the fairness analysis, the results cannot yet be considered reliable because the race variable contains inconsistent representations of the same categories. The race categories should therefore be standardized and the false-positive rates recalculated before drawing conclusions about differences between the models and COMPAS.

No, the Week 2/3 conclusion does not fully hold under cross-validation. While logistic regression was already slightly better on the original test set, cross-validation strengthens this finding: logistic regression achieved 0.673 validation accuracy compared with 0.608 for the decision tree. The decision tree also shows a much larger train–validation gap (0.094 vs. 0.001), suggesting substantially more overfitting than the logistic regression model. Therefore, the cross-validation results support logistic regression as the more consistent model, although the original fairness conclusions should still be treated cautiously.

## Project structure

```
.
├── main.py                  # entry point: run the whole pipeline
├── config.yaml               # all tunable settings live here
├── requirements.txt
├── src/
│   ├── data.py               # loading
│   ├── preprocessing.py      # row-preserving cleaning (incl. domain-rule checks), training-only de-duplication, deployable preprocessing pipeline, and the split that locks the final test set away (week 4)
│   ├── model.py               # model construction + build_pipeline(): preprocessing + model as one estimator (week 5)
│   ├── evaluate.py           # stratified k-fold cross-validation, out-of-fold report + fairness check (week 4)
│   ├── tuning.py             # hyperparameter tuning with Optuna + nested cross-validation (week 5)
│   └── results.py            # saves each run's report to disk
├── results/                  # created automatically -- one file per run (not tracked in git)
└── data/
    ├── compas_two_year_recidivism.csv
    └── README.md              # problem description + full data dictionary
```

## Pipeline progress

This table is updated after each practical class, so you can always see what changed in the pipeline and why -- it's a running log, not a fixed syllabus.

| Week | Practical class focus | Added to the pipeline |
|------|------------------------|------------------------|
| 2 | Introduction & baseline pipeline | Initial version: project structure, a single naive train/test split (no cross-validation), minimal preprocessing (drop rows with missing values, one-hot encode categoricals), logistic regression baseline, a first (deliberately simple) fairness check comparing our model's and COMPAS's own false-positive rate by race, train-vs-test accuracy reporting (to start spotting overfitting), and each run's full report saved automatically to `results/` |
| 3 | EDA + preprocessing -- diagnose the data, then fix it | `src/data_diagnostics.py` (missingness-mechanism test via chi-square + Cramér's V, domain-rule invalid-value detection, two-way duplicate check) and `src/preprocessing.py` (leak-safe category cleanup, mechanism-matched imputation with `_was_missing` indicators for MNAR columns, a deployable `ColumnTransformer`, **and** the train/test split itself, all in the one file rather than split across two) replace the old naive `dropna()`/`pd.get_dummies()` preprocessing; encoder/scaler pair (target encoding + standard scaling) chosen by an empirical grid over 15 repeated splits, checked against the runner-up with a paired comparison so the win isn't just noise; three redundant columns (found via correlation + VIF) dropped; `config.yaml` gains `diagnostics` and `preprocessing` sections -- see "Preprocessing decisions" below. Threshold-independent metrics (ROC-AUC/PR-AUC) and a calibration check are deliberately **not** added yet -- not yet |
| 4 | Preprocessing inside the pipeline + cross-validation -- evaluating a model honestly | A **locked final test set** (20%, stratified, seed 42) is set aside by `split_dev_test()` (replaces `split_train_test()`) and never scored; models are now judged by **stratified 5-fold cross-validation** of the whole pipeline (preprocessing + model) on the development set, reported per fold with mean ± std and the train-validation gap; the classification report and fairness check now use out-of-fold predictions; target encoding switched to scikit-learn's cross-fitting `TargetEncoder` (a row's own label never leaks into its own encoding), encoder/scaler set by hand in `config.yaml` (target encoding + robust scaling, reasons in the comments); **two fixes** in `clean_dataset()`: genuine `NaN`s in categorical columns were being turned into the string `"nan"` (a fake category), so 229 `c_charge_degree` gaps were never imputed or flagged -- fixed in `config.yaml` alone: `"nan"` added to `diagnostics.placeholder_tokens` (the category cleanup's last step turns listed tokens into `NaN`, after its text conversion); and it no longer drops rows -- de-duplication moved to a separate, training-only `drop_duplicate_rows()` (run before the dev/test split), so the same cleaning can run on new data where every row needs a prediction; `src/data_diagnostics.py` removed -- its one cleaning function (`flag_invalid_values`) moved into `preprocessing.py`, and the EDA-only checks (missingness test, duplicate counts) live in the EDA notebooks, not in every pipeline run; `dummy` (majority-class) model added as the floor to beat, and `random_forest` registered (sensible defaults, untuned); the final model is refit on the whole development set after CV; `config.yaml` gains `test_set` and `cv` sections -- see "Model evaluation" below |
| 5 | Hyperparameter tuning without leakage | New `src/tuning.py`: hyperparameters are tuned with **Optuna** (seeded TPE sampler) on the development set only -- every candidate is scored by stratified 5-fold CV of the **whole** pipeline (preprocessing re-fit in every fold of every trial), on the same folds for every candidate; the tuning procedure is estimated honestly by **nested cross-validation** (the tuning repeated inside each outer CV fold, scored on the outer fold it never saw), whose out-of-fold predictions now feed the classification report and fairness check; the reported "best trial" score is shown next to the nested one as an optimism check; the final model is the tuned pipeline refit on all development rows; `build_pipeline()` in `src/model.py` now builds the preprocessing + model `Pipeline` (one place to add a learned step later); `config.yaml` gains a `tuning` section (on/off, trials, inner folds, seed, one search space per model type) and the model becomes the decision tree; `optuna` added to `requirements.txt`. The locked test set is still untouched -- see "Hyperparameter tuning" below |

## Preprocessing decisions

*(Written straight from the diagnosis in `Practical/W3/notebooks/01_eda_introduction.ipynb`; the preprocessing walkthrough is in `Practical/W4/notebooks/02_preprocessing.ipynb`. This is the summary.)*

| Column(s) | Issue found | Mechanism | What was done |
|---|---|---|---|
| `age` | 2.0% missing | MCAR | median impute, no indicator needed |
| `juv_fel_count` | 3.0% missing | MCAR | median impute, no indicator needed |
| `priors_count` | ~7% missing (incl. placeholder tokens) | MNAR -- tied to `age_cat` | median impute + `priors_count_was_missing` flag |
| `c_charge_degree` | 3.2% missing | MNAR -- tied to `age_cat` | mode impute + `c_charge_degree_was_missing` flag |
| `race` | ~1% missing (placeholder tokens) | MCAR | mode impute, no indicator (excluded from model features anyway) |
| `sex` | ~1.5% missing (incl. placeholder tokens) | MCAR | mode impute, no indicator needed |
| `age`, `decile_score`, `juv_fel_count`, `priors_count` | invalid values (out-of-range or negative) | domain rule | converted to `NaN` before imputation |
| `sex` / `race` / `c_charge_degree` / `score_text` | inconsistent category spelling (casing, whitespace, abbreviations) | data entry | canonicalized to one spelling per category |
| whole rows | 72 exact-duplicate rows, all sharing a repeated `id` | data entry | dropped, kept first occurrence |
| `prior_offenses`, `age_in_months`, `juvenile_total` | redundant with other columns (correlation r=1.00, or -- for `juvenile_total` -- an exact sum caught only by VIF) | multicollinearity | dropped |

**Encoder/scaler pair:** chosen by hand in `config.yaml` -- **target encoding** (compact, informative and **robust scaling** (median/IQR, so the few extreme counts don't set the scale). The alternatives (`onehot`/`ordinal`/`count`, `none`/`standard`/`minmax`) are one config change away.


## Hyperparameter tuning

*(Walkthrough: `Practical/W5/notebooks/04_hyperparameter_tuning.ipynb`.)* With `tuning.enabled: true` in `config.yaml`, `python main.py` also:

1. **Nested cross-validation** -- for each of the 5 outer CV folds, runs a full Optuna study on the outer-training rows only (each trial = 5-fold CV of the whole pipeline), refits the best candidate on those rows and scores it on the outer-validation fold. This is the honest estimate of the tuned model.
2. **Tuning on the whole development set** -- the same study, once, on every development row: these are the hyperparameters the final model uses.

The best trial's score is printed too, but it is **optimistic** (the maximum of many noisy CV scores) -- report the nested one. Search spaces live in `config.yaml` under `tuning.search_spaces`, one per model type, keyed by Pipeline parameter name (`model__max_depth`); a model without a search space can't be tuned until you add one. Cost: `(cv.n_splits + 1) x n_trials x tuning.n_splits` fits.

## Environment setup

You only need to do this once per machine.

### macOS / Linux
```bash
python3 -m venv venv                 # creates an isolated Python environment in a folder called "venv"
source venv/bin/activate             # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```

### Windows -- PowerShell
```powershell
python -m venv venv                  # creates an isolated Python environment in a folder called "venv"
venv\Scripts\activate                # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```
If PowerShell blocks the activation script, run this once first:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Windows -- cmd.exe
Same three steps as above, just with cmd's own activation command:
```cmd
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

Once the environment is active you'll see `(venv)` at the start of your prompt. To leave it later, run `deactivate` (same command on every OS).

### Every time after the first

Creating the environment and installing packages only needs to happen once, ever. Every other time you sit down to work -- a new terminal window, the next practical class, tomorrow -- you don't repeat any of the steps above. From the project's root folder, you just need to:

**macOS / Linux**
```bash
source venv/bin/activate
python main.py
```

**Windows**
```powershell
venv\Scripts\activate
python main.py
```

That's it -- activate, then run.

**When `requirements.txt` changes** (a week adds a package -- e.g. week 5 adds `optuna`), reinstall once, with the venv active: `pip install -r requirements.txt`. Otherwise `python main.py` fails with `ModuleNotFoundError`. If you don't see `(venv)` at the start of your prompt, the environment isn't active and `python main.py` may use the wrong Python (or fail to find a package) entirely.

## Environment Troubleshooting

Two Windows issues come up often enough to note here -- if you hit either, this saves you re-diagnosing it from scratch.

**PowerShell blocks the venv activation script, every new terminal.** The `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` line above only fixes it for that one terminal window -- close it and it's back. For a fix that actually sticks across sessions, run this **once**, instead:
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```
If it still doesn't stick (common on locked-down school/lab machines with a Group Policy that resets it on every logon), skip PowerShell entirely: use **Git Bash** (`source venv/Scripts/activate`) or **cmd.exe** (`venv\Scripts\activate.bat`) instead -- neither is affected by PowerShell's execution policy.

**Windows blocks the terminal/Python from reading or writing files in Documents (or Desktop/Pictures).** Shows up as an "Access is denied" error, or a silent failure to create/update a file, only when the project sits inside one of those folders. Two independent settings can cause this -- check both:
- **Windows Security -> Virus & threat protection -> Manage ransomware protection** -- turn off **Controlled folder access**, or add your terminal/Python/editor to its allowed-apps list.
- **Settings -> Privacy & security -> File system** -- make sure the terminal/Python has access.

## Running the pipeline

With the environment active (see above), from the project's root
folder, on any OS:
```bash
python main.py
```

This loads `config.yaml`, diagnoses and cleans the data (week 3), locks the final test set away, cross-validates preprocessing + model on the development set (week 4), tunes the model's hyperparameters if `tuning.enabled` is on (week 5 -- see "Hyperparameter tuning" above), and prints:
- **a per-fold cross-validation table** -- train and validation accuracy for each of the 5 folds, the gap between them, and their mean ± std. Comparing train and validation is how you catch overfitting: if the model looks much better on the data it was trained on than on data it's never seen, it has memorised rather than learned something that generalises.
- with tuning on: the nested cross-validation table (the honest score of the tuned model, and the hyperparameters chosen in each fold), and the tuning report (best hyperparameters, top trials, and the optimism check)
- a classification report on the out-of-fold predictions (from nested CV when tuning is on)
- a false-positive-rate-by-race comparison between our model and COMPAS's own score (same rows)
- the final model -- the same pipeline (tuned, if tuning is on) refit on all development rows (CV estimated how good it is; this is the model itself)
- a reminder of how many rows are in the locked test set -- which is **not** evaluated

All of this is also saved to a timestamped file in `results/` (e.g.`results/run_20260916_143012.txt`), so it doesn't just scroll past in your terminal -- open it later, or change something in `config.yaml` (like the model type) and compare the new file to the last one.
`results/` is created automatically the first time you run the
pipeline, and isn't tracked in git (see `.gitignore`) since it's
generated output, not source.

You're free to improve on this structure or restructure it entirely -- what matters is that your project stays runnable end-to-end with a single command, and that each piece (data, preprocessing, model, evaluation) stays easy to find and change independently.

## Push to GitHub via Terminal

Standard workflow, from the project's root folder, with the venv active:
```bash
git add .
git commit -m "short description of what changed"
git push
```

**If `git push` asks for a password and rejects your normal GitHub password:** GitHub no longer accepts account passwords for git over HTTPS -- you need a **Personal Access Token (PAT)** instead.
1. On GitHub: **Settings -> Developer settings -> Personal access tokens -> Tokens (classic)** -> **Generate new token**, with at least `repo` scope.
2. When `git push` prompts for a password, paste the token instead (username stays your GitHub username).
3. So you're not asked every time: `git config --global credential.helper manager` (Windows, usually already set up by Git for Windows) or `git config --global credential.helper store` (caches it in plaintext -- fine on a personal machine, not a shared one).

Alternative: set up an SSH key once (`ssh-keygen -t ed25519`, then add the public key under **GitHub -> Settings -> SSH and GPG keys**) and use the repo's SSH remote URL (`git@github.com:...`) instead of HTTPS -- no token to manage or renew.

## Dataset

See `data/README.md`.
