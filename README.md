# Baseline Predictive Pipeline -- ETAI
Luis Soeiro 20211536

# Pratical work conclusions
## Week 2 | Baseline

 **1. Column Disagreement: Are there rows where two columns disagree with each other (inconsistent together)**
 
 The audit (consistency_audit_summary) identified two potential cross-column inconsistencies, involving age categories and COMPAS score labels. The other tested relationships showed no disagreements among comparable records. The duplicate checks also identified repeated records that should be investigated before preprocessing

 **2. Additional Domain Rules: Is there a column you'd want a domain rule for that this notebook didn't build one for?**

 - juv_misd_count and juv_other_count: Both should be non-negative because offense counts cannot be negative.
 - juvenile_total: Should be non-negative and equal the sum of juv_fel_count, juv_misd_count, and juv_other_count.
 - prior_offenses: Should be non-negative. It should also be compared with priors_count, since the two columns may represent the same information.
 - age_in_months: Should represent a plausible age in months and be consistent with age, according to the column's documented definition.
 - two_year_recid: Must contain only 0 or 1, because it is a binary classification target.
 - Categorical columns (sex, race, age_cat, c_charge_degree, and score_text): Should contain only documented category values. Inconsistent capitalization, whitespace, unexpected labels, and placeholder tokens should be investigated.
 - id: Should be present and unique if each identifier is expected to correspond to one record.

 **3. Additional validation check: Is there any other invalid value that was not found in this notebook?**
 - Unexpected categories in c_charge_degree, race, and sex: The audit flagged 274 records in c_charge_degree, 205 in race, and 49 in sex.
 - Repeated identifiers (id): A total of 144 records belong to groups with repeated IDs. These records should be investigated to determine whether they are duplicate observations or whether the identifier's uniqueness assumption is incorrect.
 - Non-numeric values in prior_offenses: The audit found 81 values that could not be converted to numeric values. These may be placeholders or invalid entries and should be inspected before conversion or imputation.
 - Potentially invalid age_in_months values: Nine records fall below the assumed minimum of 216 months (18 years).
 - Negative juvenile_total values: Four records violate the non-negative count rule


## Week 2/3 conclusions

The logistic regression model performed slightly better overall, achieving a higher test accuracy than the decision tree (0.677 vs. 0.668). Both models have relatively small differences between training and test accuracy, with gaps of 0.001 and 0.012 respectively, indicating no clear signs of overfitting.

Looking at the classification metrics in more detail, the two models exhibit different performance characteristics. Logistic regression performs better at identifying non-recidivists (class 0), achieving a higher recall (0.74 vs. 0.68) and F1-score (0.72 vs. 0.69). In contrast, the decision tree performs better at identifying recidivists (class 1), with higher recall (0.65 vs. 0.60) and a slightly higher F1-score (0.64 vs. 0.63). Therefore, the decision tree improves the detection of recidivism at the expense of performance on non-recidivists, rather than providing an overall improvement. Nevertheless, it is important to understand that only a simple model was used on noth logistic regression and decision tree, meaning no optimal / different parameters were considered to undertand which model could further improve the results

Regarding the fairness analysis, the results cannot yet be considered reliable because the race variable contains inconsistent representations of the same categories. The race categories should therefore be standardized and the false-positive rates recalculated before drawing conclusions about differences between the models and COMPAS.


## Week 4 conclusions

| Model               | Holdout accuracy (W3) | CV accuracy (mean ± std) |   CV train–val gap |
| ------------------- | --------------------: | -----------------------: | -----------------: |
| Dummy               |             **0.550** |        **0.549 ± 0.000** | **−0.000 ± 0.000** |
| Logistic regression |             **0.661** |        **0.673 ± 0.013** | **+0.001 ± 0.015** |
| Decision tree       |             **0.619** |        **0.608 ± 0.014** | **+0.094 ± 0.020** |
| Random forest       |                 **—** |        **0.645 ± 0.020** | **+0.093 ± 0.028** |

I would trust Logistic regression's mean cross-validation accuracy of 0.673. Its accuracy on the locked holdout set was 0.661 with a very small train–validation gap of 0.001 provides a similar estimate of generalization performance.


The distinction is:

Dummy, Logistic Regression, Decision Tree: have the original Week 3 holdout accuracy → report those values.
Random Forest: if it was not evaluated on the holdout set as part of Week 3 → use — or N/A, even though you now technically have a holdout accuracy from the Week 4 run.
The Random Forest's CV results can still be reported because those are part of the current Week 4 evaluation.



For this experiment, I would trust the 5-fold cross-validation validation accuracy more than any single holdout number when judging how the models generalize, because it averages performance across five different validation splits.


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


Logistic regression achieved a mean cross-validation accuracy of 0.673 ± 0.013, with a very small train–validation gap of 0.001. Its accuracy on the locked holdout set was 0.661, providing a similar estimate of generalisation performance.

That is a much stronger interpretation than simply choosing whichever single accuracy number is highest.


conclusions:
The logistic regression model performed slightly better overall, achieving a higher test accuracy than the decision tree (0.677 vs. 0.668). Both models have relatively small differences between training and test accuracy, with gaps of 0.001 and 0.012 respectively, indicating no clear signs of overfitting.

Looking at the classification metrics in more detail, the two models exhibit different performance characteristics. Logistic regression performs better at identifying non-recidivists (class 0), achieving a higher recall (0.74 vs. 0.68) and F1-score (0.72 vs. 0.69). In contrast, the decision tree performs better at identifying recidivists (class 1), with higher recall (0.65 vs. 0.60) and a slightly higher F1-score (0.64 vs. 0.63). Therefore, the decision tree improves the detection of recidivism at the expense of performance on non-recidivists, rather than providing an overall improvement. Nevertheless, it is important to understand that only a simple model was used on noth logistic regression and decision tree, meaning no optimal / different parameters were considered to undertand which model could further improve the results

Regarding the fairness analysis, the results cannot yet be considered reliable because the race variable contains inconsistent representations of the same categories. The race categories should therefore be standardized and the false-positive rates recalculated before drawing conclusions about differences between the models and COMPAS.

No, the Week 2/3 conclusion does not fully hold under cross-validation. While logistic regression was already slightly better on the original test set, cross-validation strengthens this finding: logistic regression achieved 0.673 validation accuracy compared with 0.608 for the decision tree. The decision tree also shows a much larger train–validation gap (0.094 vs. 0.001), suggesting substantially more overfitting than the logistic regression model. Therefore, the cross-validation results support logistic regression as the more consistent model, although the original fairness conclusions should still be treated cautiously.


## Week 5 conclusions - Hyperparameter tuning
Interpretation:

- Best tuned CV accuracy: Random Forest at 0.683 ± 0.016, with a small mean train-validation gap of +0.008.
- Largest improvement over default CV: Decision Tree, from 0.610 to 0.666, while its mean gap falls from +0.085 to +0.007.
- Logistic Regression remains relatively stable, with a near-zero mean gap.
- Oblique Random Forest improves from 0.647 to 0.672, but retains a larger gap (+0.041) than the other tuned models.
- The tuning score is the best Optuna trial's mean inner-CV score, with its fold standard deviation shown. It is a selection score, not an unbiased estimate; use nested CV for comparing tuned performance.

The best trial's hyperparameters shown above are from the full-development-set tuning study. They may differ from the parameters selected in individual outer folds.


## Week 6 | Ensembles

**MODEL EVALUATION**

| Model                 | Default: CV mean ± std (gap) | Tuned: nested CV mean ± std (gap) | Tuning score (best trial) | Chosen hyperparameters                                                                                                      |
| --------------------- | ---------------------------- | --------------------------------- | ------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| Logistic regression   | 0.672 ± 0.013 (+0.003)       | 0.674 ± 0.015 (+0.001)            | 0.675 ± 0.016             | `C=0.3908`, `class_weight=None`                                                                                             |
| Decision tree         | 0.610 ± 0.018 (+0.085)       | 0.666 ± 0.007 (+0.007)            | 0.666 ± 0.007             | `max_depth=3`, `min_samples_leaf=49`, `min_samples_split=31`, `criterion='gini'`, `class_weight=None`                       |
| Random forest         | 0.650 ± 0.019 (+0.082)       | 0.683 ± 0.016 (+0.008)            | 0.683 ± 0.016             | `n_estimators=212`, `max_depth=None`, `min_samples_leaf=27`, `max_features='log2'`, `criterion='gini'`, `class_weight=None` |
| Oblique random forest | 0.647 ± 0.011 (+0.114)       | 0.672 ± 0.016 (+0.041)            | 0.672 ± 0.016             | `n_estimators=313`, `n_components=2`, `tree__max_depth=None`, `tree__min_samples_leaf=6`                                    |
| AdaBoost              | 0.676 ± 0.012 (+0.006)       | 0.678 ± 0.013 (+0.015)            | 0.678 ± 0.013             | `estimator__max_depth=6`, `n_estimators=89`, `learning_rate=0.3204`                                                         |

**1 & 3. Best model**
I would choose the Random Forest, with nested CV accuracy of 0.683 ± 0.016 and a mean train–validation gap of +0.008.
Its lead over the next-best model, AdaBoost (0.678), is only 0.005, which is smaller than the Random Forest's fold-to-fold standard deviation of 0.016. Therefore, the lead is not large enough to confidently conclude that Random Forest is better than AdaBoost. The results suggest similar performance, so I would choose Random Forest provisionally, while considering model simplicity and other evaluation metrics before making a final decision.

**2. Hyperparameter reflection**
The chosen hyperparameters mostly make sense given how each model works. Decision Tree selects a shallow tree (max_depth=3), which limits complexity and reduces overfitting, while Random Forest allows unlimited depth but uses large leaves (min_samples_leaf=27) to control overfitting across trees. Oblique Random Forest uses unlimited tree depth and smaller leaves (min_samples_leaf=6), resulting in more complex individual trees and a larger train–validation gap. AdaBoost uses moderately deep trees (max_depth=6), allowing individual learners to capture more complex patterns, although this may increase overfitting. Logistic Regression remains comparatively simple, with regularization controlled by C≈0.39, consistent with its near-zero train–validation gap.

## Week 7 | 

**1. Scoreboard**


| Model | Tuned (nested) | Untuned | Floor | vs best: diff, wins, p |
|---|---|---|---|---|
| Gradient boosting | **0.727 ± 0.014** | 0.727 | 0.500 | -- (best) |
| HistGradientBoosting | 0.727 ± 0.015 | 0.717 | 0.500 | +0.000, 3/5, p = 0.84 |
| Random forest | 0.726 ± 0.014 | 0.683 | 0.500 | +0.001, 4/5, p = 0.77 |
| AdaBoost | 0.724 ± 0.014 | 0.723 | 0.500 | +0.003, 5/5, p = 0.06 |
| Logistic regression | 0.718 ± 0.014 | 0.719 | 0.500 | +0.009, 5/5, p = 0.008 |
| Decision tree | 0.717 ± 0.016 | 0.604 | 0.500 | +0.010, 5/5, p = 0.035 |
| Naive Bayes | 0.688 ± 0.011 | 0.685 | 0.500 | +0.039, 5/5, p = 0.001 |
| Dummy (no model) | 0.500 ± 0.000 | -- | 0.500 | +0.227, 5/5, p < 0.001 |

*The last column compares gradient boosting with each model using the corrected paired t-test on the same folds (`corrected_paired_test()`, test/train ratio 1/4).


**Selection metric**

We chose **ROC-AUC** (`cv.selection_metric: "roc_auc"`), because COMPAS is used to rank defendants by risk. ROC-AUC measures exactly that ranking, without depending on the 0.5 threshold, so the threshold stays a separate decision made on development data. It is also more stable from fold to fold than threshold metrics like accuracy: in the corrected paired test, gradient boosting vs logistic regression was clearly different on ROC-AUC (p = 0.006) but indistinguishable on accuracy (p = 0.38).


## Week 8 | Feature Selection

**Feature selection benchmark**
With feature selection, tuned XGBoost keeps only 2 of the 10 features (age and priors_count) and reaches a nested-CV ROC-AUC of 0.725 ± 0.012, against 0.728 ± 0.016 with all features. The corrected paired t-test finds no real difference (mean difference ≈ −0.002, selection wins 2/5 folds, p ≈ 0.55), so the smaller model is as good: almost all the predictive signal in COMPAS sits in age and prior record. Tuning points the same way, since the final model kept the same two features even with the loosest setting (min_votes = 1). 

I would keep selection on: for the same score, a two-feature model is far easier to explain and audit, slightly more stable across folds (std 0.012 vs 0.016), and its false positive rates by race are essentially unchanged (0.31 vs 0.30 for African-American defendants). The costs are a slower pipeline and no fairness gain

## Week 9 | Beyond COMPAS (option A: Adult income)

**The question:** does a person earn more than 50,000 USD a year (`income_over_50k`)? Same pipeline, new data: 6,000 labelled people (`data/adult_train.csv`, 24% earn > 50K) and 2,000 new people to predict (`data/adult_new.csv`).

**How to run**

```bash
python main.py config_adult.yaml    # Adult income
python main.py                      # COMPAS, unchanged (config.yaml)
```

Everything Adult-specific is in `config_adult.yaml`, with its own output folder (`results/adult/`, scoreboard `results/adult/runs.csv`). The final run tunes XGBoost with feature selection in nested CV (about 6 minutes). With `tuning.enabled: false` it takes about 10 seconds. After the final refit it writes the predictions for `adult_new.csv` to `predictions/adult_new_predictions.csv` (`id`, `probability`, `prediction`).

**What broke, why, and how it was fixed**

| What broke | Why: the hidden COMPAS assumption | Fix |
|---|---|---|
| `python main.py` always ran COMPAS | `load_config()` always read `config.yaml` | `python main.py [config]` (argparse, default `config.yaml`). The run prints its config file and writes it in the report header |
| `"?"` stayed a real category in `workclass` (379), `occupation` (381) and `native_country` (86), and the `workclass_was_missing` flag was 0 on every row. **Nothing crashed** | placeholder tokens became `NaN` only inside `_canonicalize_categories()`, which only touches the columns listed in `canonical_categories` (COMPAS's columns with spelling variants). Adult's columns have no spelling variants, so they were never cleaned | `_placeholders_to_nan()` in `clean_dataset()`: every text column. After it: 379 / 381 / 86 missing, no `"?"` left |
| fairness report: `KeyError: 'score_text'` | `fairness_report()` always compared the model with COMPAS's own score, and `split_features_target()` always kept `score_text` in `extras` | the comparison score is now a config setting, `data.reference_score` (COMPAS: `score_text`, `"Low"` = predicted 0). Without one (Adult), the report shows the model's rates only. The header wording is generic ("true class 0, predicted 1") |
| validity rules | COMPAS's `age: {min: 18}` would turn all 74 seventeen-year-olds into missing values | config only: Adult's own rules (`age` 17–90, `hours_per_week` 1–99, `education_num` 1–16, capital gain / loss ≥ 0). 99999 in `capital_gain` is the census top code, so it is kept |
| a feature missing from the lists is dropped silently | the `ColumnTransformer` uses only the listed columns and drops the rest without an error | `check_feature_columns()`: raises if a listed feature isn't in the data, and the run prints any column listed nowhere (none, for both configs) |
| no way to predict new rows | `main.py` only evaluated on labelled data | `src/predict.py` → `predict_new_data()`: the same cleaning (but no `drop_duplicate_rows`), the same feature split (no target), the refit pipeline's probability and a 0/1 prediction at 0.5. It checks rows out = rows in. Turned on by a `predict` section in the config |
| messages naming `config.yaml` | written for one config | they name the config actually used |

Untouched and working on Adult:
- the target encoder (binary);
- `VoteSelector`;
- `drop_duplicate_rows` (no duplicates);
- stratified folds, tuning, nested CV, the scoreboard.

**COMPAS still runs exactly as before:** the same per-fold scores, chosen hyperparameters, nested ROC-AUC (0.728 ± 0.016), confusion matrix and false positive rates by race. The only difference in its report is the generic header line of the fairness section.

**Data decisions** (`config_adult.yaml`, from the data card in `notebooks/09_beyond_compas.ipynb`)
- **`"?"` → missing:**
  - categories: imputed with the most frequent value;
  - a `workclass_was_missing` flag: the same 379 people are missing `occupation`, and they earn > 50K half as often (12% vs 25%);
  - `v01`–`v06` (about 3% empty, no pattern): median.
- **Not features:**
  - `id`;
  - `fnlwgt`: a census sampling weight, not a trait of the person;
  - `education`: the same as `education_num`, one to one; `education_num` keeps the order.
- **Target encoding:** `native_country` has 41 values; one-hot would add about 40 columns.
- **Sensitive attributes:** the pipeline audits **`sex`** (never a feature). **`race` is not a feature either:** a protected attribute shouldn't decide an income-based outcome, and leaving it out costs nothing. Offered as a feature, it got no vote from the selector, and the scores were identical (0.913). `relationship` (husband / wife) and `marital_status` still carry information about sex, which is why the fairness report still matters.
- **All 25 undocumented `v` columns are offered as features,** and feature selection decides.

**Results** (development set, 4,800 rows, the same 5 stratified folds; nested CV when tuned)

| Run | Tuning | Feature selection | ROC-AUC | Accuracy | F1 (class 1, 0.5) |
|---|---|---|---|---|---|
| No model (always "≤ 50K") | — | — | 0.500 | 0.761 | 0.000 |
| XGBoost, default parameters | off | off | 0.905 ± 0.018 | 0.857 ± 0.014 | 0.663 ± 0.044 |
| XGBoost, default parameters | off | `min_votes = 2` | 0.913 ± 0.016 | 0.864 ± 0.011 | 0.686 ± 0.030 |
| XGBoost, tuned | 30 trials | off | 0.906 ± 0.015 | 0.858 ± 0.010 | 0.663 ± 0.027 |
| **XGBoost, tuned (`config_adult.yaml`)** | **30 trials** | **tuned (`min_votes` 1–2)** | **0.910 ± 0.014** | **0.859 ± 0.009** | **0.675 ± 0.028** |
| Target | | | ≥ 0.90 | ≥ 0.855 | ≥ 0.66 |

Corrected paired t-test on ROC-AUC, same folds:

| Comparison | Difference | Folds won | p |
|---|---|---|---|
| tuned: with selection vs without | +0.004 | 5/5 | 0.029 |
| default: with selection vs without | +0.009 | 4/5 | 0.098 |
| with selection: tuned vs default | −0.003 | 1/5 | 0.24 |
| without selection: tuned vs default | +0.002 | 3/5 | 0.62 |

**Features kept** by the final model (8 of 36):
- numeric: `age`, `education_num`, `capital_gain`, `capital_loss`, `hours_per_week`;
- categorical: `marital_status`, `occupation`, `relationship`.

**No `v` column:** all 25 get 0 of 3 votes, as do `workclass`, `native_country` and the `workclass` flag. So `min_votes` 1 and 2 keep the same 8 features.

**Fairness** (false positive rate by sex, out-of-fold, final model): women 0.02 (n = 1,390), men 0.09 (n = 2,261). Women earn > 50K less often in this sample, so the model rarely predicts it for them. A full audit would also compare the true positive rates.

**Predictions:** `predictions/adult_new_predictions.csv` has 2,000 rows in and 2,000 out, with 20.0% predicted > 50K (24% in the training data).

**Reflection**
- **The hardest assumption to find was the placeholder cleaning.** Nothing crashed: `"?"` simply became one more category and the missingness flag stayed 0. Only the `value_counts(dropna=False)` check showed it. The fairness report at least failed loudly.
- **What got us to the targets** was clean inputs, and above all feature selection. The 25 undocumented columns are noise (no model voted for any of them), and removing them is the only change the corrected test calls significant (+0.004 ROC-AUC, 5/5 folds, p = 0.03).
- **Tuning added nothing measurable on Adult.** The default XGBoost settings already suit it, and default + selection is as good as the tuned run (p = 0.24) at a fraction of the time.
- **The final config keeps tuning on,** so the reported scores are the honest nested estimate of the full procedure, and all three targets hold in that run.

### Running any dataset safely

Adult showed that some COMPAS assumptions fail silently. So after it, every module was audited for anything that only works for one dataset or one kind of target. The pipeline now runs any **classification** dataset from a config file, and a config that doesn't fit its data stops the run **before anything is fitted**, with every problem listed. A regression target (a continuous number) is rejected with a clear message: stratified folds, classifiers and class probabilities don't apply to it.

**What was assumed, and what changed**

| Part | Assumption found | Now |
|---|---|---|
| config (`main.py`) | read without any check: a typo in a name surfaced minutes into a run, or never (an unlisted column is just dropped) | new `src/config.py`. `load_config()` fills the defaults of optional settings and `validate_config()` checks the file on its own: required keys, known encoder / scaler / imputer / model / metric names, features listed twice or as both kinds, the target or sensitive attribute listed as a feature, search spaces (does every parameter exist in the pipeline? `low` < `high`?). `check_dataset()` then checks it against the data (below). Every problem is reported at once (`ConfigError`) |
| target labels | the target had to be 0/1: XGBoost needs 0 .. k-1, and the reports printed "true 0 / true 1" | `encode_target()` maps any labels ("yes"/"no", ">50K", "Low"/"Medium"/"High", True/False) to 0 .. k-1, and keeps the original names for every report and for the predictions file. Two classes: the positive class is `data.positive_class`, or the larger label by default. A 0/1 target keeps exactly its values |
| number of classes | binary only: probabilities stored as `[:, 1]`, a 2×2 confusion matrix, the binary target encoder, `VoteSelector`'s `coef_[0]`, a binary fairness report and locked-test block | probabilities are kept as one column per class. With three or more classes, the prediction is the most probable class, and the target encoder is `auto`. `VoteSelector` counts a feature as used when it is non-zero for any class (one-vs-rest L1, because `liblinear` fits two classes only). The confusion matrix and report name the classes. The fairness report gives a per-class false positive rate by group, and the locked test reports `roc_auc_ovr` |
| metrics | `precision`, `recall`, `f1`, `roc_auc`, `average_precision` and `fbeta_2` only exist for two classes | with three or more classes these are rejected with a suggestion (`f1_macro`, `roc_auc_ovr`, ...); `fbeta_<beta>_<average>` (e.g. `fbeta_2_macro`) added |
| sensitive attribute | required | optional (`data.sensitive_attr: null`): the fairness report says it was skipped |
| rows without a target | crashed the stratified split | dropped from the training data only, and reported |
| data files | CSV with default settings only | `.csv`, `.txt`, `.tsv`, `.parquet`, `.xlsx`, `.xls` and `.json`, by extension; reader options from `data.read_options` (e.g. `{sep: ";", decimal: ","}`); clear errors for a missing or empty file, an unsupported type or repeated column names |
| cleaning | `_canonicalize_categories` turned real NaNs into the text `"nan"`, which only became missing again because COMPAS lists `"nan"` as a placeholder | a real NaN stays NaN; pandas `category` columns become plain values |
| text in a numeric feature | reached the imputer and crashed there | stopped at the start: add the column to `diagnostics.numeric_text_columns` or list it as categorical |
| feature selection | could keep zero features (then the model crashes) | keeps the most-voted features when none reaches `min_votes` |
| new data | needed an `id` column and printed "% predicted 1" | uses the row number when there is no id; outputs the original labels (and one `probability_<class>` column per class for multiclass); stops with a clear error if a feature is missing |
| run folder, report | the cache folder was left behind when a run failed; `tuning['n_trials']` etc. crashed when a key was left out | the cache is always removed; the defaults match `main.py`. The report header and scoreboard also record the config file and target |

**Checked against the data before fitting** (`check_dataset()`):
- the target exists, has at least 2 classes and isn't continuous;
- every class has enough rows for the locked test split and the stratified folds, including the inner tuning folds;
- every listed feature, indicator source, sensitive attribute and reference column exists;
- numeric features really are numeric, and no feature is entirely empty;
- the metrics suit the number of classes;
- `data.positive_class` is one of the labels.

Columns listed nowhere, a missing id column and rows without a target are reported as notes.

**Adding a new dataset**
1. Put the file in `data/` and copy `config_adult.yaml` to `config_<name>.yaml`. Its comments say what each setting is for.
2. `data`:
   - `path` and `target` (any labels);
   - `drop_columns` (ids, leakage, non-features);
   - `sensitive_attr` (or `null`);
   - optionally `positive_class` and `read_options`.
3. `diagnostics`:
   - placeholder tokens;
   - this dataset's own validity rules;
   - spelling variants;
   - text columns that are really numeric;
   - redundant columns.
4. `preprocessing`:
   - `numeric_features`, `categorical_features` and `mnar_indicator_sources`. A column listed nowhere is left out (the run says which).
5. `cv.metrics` and `selection_metric`: with three or more classes, use the multiclass versions (`f1_macro`, `roc_auc_ovr`, ...).
6. `output`: its own results folder and scoreboard. Add a `predict` section to score new rows.
7. Run `python main.py config_<name>.yaml` with `tuning.enabled: false` first. Any config problem is listed at the start.

**Automated checks** (`python tests/test_datasets.py`, 6 tests, all passing). Synthetic datasets run through the whole pipeline:
1. a yes/no target with text labels, from a semicolon-separated file, with placeholders, spelling variants, a numeric column stored as text, rows without a target, repeated ids, an unlisted column, and new data without an id;
2. a three-class target with XGBoost, tuning, nested CV, feature selection, multiclass metrics, the locked-test evaluation and per-class probabilities;
3. numeric features only, no sensitive attribute, a tab-separated file;
4. eleven configs that must be rejected before fitting, each with the expected message;
5. target label encoding and the positive class;
6. `config.yaml` and `config_adult.yaml` passing every check.

**COMPAS and Adult give the same numbers as before this audit.** Both were re-run and their reports compared line by line.

## Project structure

```
.
├── main.py                  # entry point: python main.py [config file] (default config.yaml)
├── config.yaml               # COMPAS: all tunable settings live here
├── config_adult.yaml         # Adult income (week 9): the same pipeline on a new dataset
├── requirements.txt
├── src/
│   ├── config.py             # loads a config and checks it, on its own and against the data, before anything is fitted (week 9)
│   ├── data.py               # loading (csv, tsv, txt, parquet, excel, json; reader options from the config)
│   ├── preprocessing.py      # row-preserving cleaning (incl. domain-rule checks), training-only de-duplication, deployable preprocessing pipeline, and the split that locks the final test set away (week 4)
│   ├── model.py               # model construction (incl. the oblique random forest and AdaBoost builders, week 6) + build_pipeline(): preprocessing + model as one estimator (week 5)
│   ├── evaluate.py           # stratified k-fold cross-validation, out-of-fold report + fairness check (week 4), corrected paired test to compare two models (week 6)
│   ├── tuning.py             # hyperparameter tuning with Optuna + nested cross-validation (week 5)
│   ├── predict.py            # predicts new, unlabelled rows with the refit pipeline (week 9)
│   └── results.py            # saves each run's report to disk
├── tests/
│   └── test_datasets.py      # synthetic datasets (text labels, three classes, numeric only) and configs that must be rejected (week 9)
├── results/                  # created automatically -- one file per run (not tracked in git); results/adult/ for Adult
├── predictions/
│   └── adult_new_predictions.csv   # id, probability, prediction for the 2,000 rows of adult_new.csv (week 9)
└── data/
    ├── compas_two_year_recidivism.csv
    ├── README.md              # problem description + full data dictionary
    ├── adult_train.csv        # Adult income: 6,000 labelled people (week 9)
    ├── adult_new.csv          # Adult income: 2,000 new people, no target
    └── ADULT_README.md        # Adult column descriptions
```

## Pipeline progress

This table is updated after each practical class, so you can always see what changed in the pipeline and why -- it's a running log, not a fixed syllabus.

| Week | Practical class focus | Added to the pipeline |
|------|------------------------|------------------------|
| 2 | Introduction & baseline pipeline | Initial version: project structure, a single naive train/test split (no cross-validation), minimal preprocessing (drop rows with missing values, one-hot encode categoricals), logistic regression baseline, a first (deliberately simple) fairness check comparing our model's and COMPAS's own false-positive rate by race, train-vs-test accuracy reporting (to start spotting overfitting), and each run's full report saved automatically to `results/` |
| 3 | EDA + preprocessing -- diagnose the data, then fix it | `src/data_diagnostics.py` (missingness-mechanism test via chi-square + Cramér's V, domain-rule invalid-value detection, two-way duplicate check) and `src/preprocessing.py` (leak-safe category cleanup, mechanism-matched imputation with `_was_missing` indicators for MNAR columns, a deployable `ColumnTransformer`, **and** the train/test split itself, all in the one file rather than split across two) replace the old naive `dropna()`/`pd.get_dummies()` preprocessing; encoder/scaler pair (target encoding + standard scaling) chosen by an empirical grid over 15 repeated splits, checked against the runner-up with a paired comparison so the win isn't just noise; three redundant columns (found via correlation + VIF) dropped; `config.yaml` gains `diagnostics` and `preprocessing` sections -- see "Preprocessing decisions" below. Threshold-independent metrics (ROC-AUC/PR-AUC) and a calibration check are deliberately **not** added yet -- not yet |
| 4 | Preprocessing inside the pipeline + cross-validation -- evaluating a model honestly | A **locked final test set** (20%, stratified, seed 42) is set aside by `split_dev_test()` (replaces `split_train_test()`) and never scored; models are now judged by **stratified 5-fold cross-validation** of the whole pipeline (preprocessing + model) on the development set, reported per fold with mean ± std and the train-validation gap; the classification report and fairness check now use out-of-fold predictions; target encoding switched to scikit-learn's cross-fitting `TargetEncoder` (a row's own label never leaks into its own encoding), encoder/scaler set by hand in `config.yaml` (target encoding + robust scaling, reasons in the comments); **two fixes** in `clean_dataset()`: genuine `NaN`s in categorical columns were being turned into the string `"nan"` (a fake category), so 229 `c_charge_degree` gaps were never imputed or flagged -- fixed in `config.yaml` alone: `"nan"` added to `diagnostics.placeholder_tokens` (the category cleanup's last step turns listed tokens into `NaN`, after its text conversion); and it no longer drops rows -- de-duplication moved to a separate, training-only `drop_duplicate_rows()` (run before the dev/test split), so the same cleaning can run on new data where every row needs a prediction; `src/data_diagnostics.py` removed -- its one cleaning function (`flag_invalid_values`) moved into `preprocessing.py`, and the EDA-only checks (missingness test, duplicate counts) live in the EDA notebooks, not in every pipeline run; `dummy` (majority-class) model added as the floor to beat, and `random_forest` registered (sensible defaults, untuned); the final model is refit on the whole development set after CV; `config.yaml` gains `test_set` and `cv` sections -- see "Model evaluation" below |
| 5 | Hyperparameter tuning without leakage | New `src/tuning.py`: hyperparameters are tuned with **Optuna** (seeded TPE sampler) on the development set only -- every candidate is scored by stratified 5-fold CV of the **whole** pipeline (preprocessing re-fit in every fold of every trial), on the same folds for every candidate; the tuning procedure is estimated honestly by **nested cross-validation** (the tuning repeated inside each outer CV fold, scored on the outer fold it never saw), whose out-of-fold predictions now feed the classification report and fairness check; the reported "best trial" score is shown next to the nested one as an optimism check; the final model is the tuned pipeline refit on all development rows; `build_pipeline()` in `src/model.py` now builds the preprocessing + model `Pipeline` (one place to add a learned step later); `config.yaml` gains a `tuning` section (on/off, trials, inner folds, seed, one search space per model type) and the model becomes the decision tree; `optuna` added to `requirements.txt`. The locked test set is still untouched -- see "Hyperparameter tuning" below |
| 6 | Ensembles in depth: random forest, oblique random forest, AdaBoost | Two new model types in `src/model.py`: `oblique_random_forest` (a forest of oblique trees built from scikit-learn parts -- every tree splits on its own random sparse linear combinations of the features, via `SparseRandomProjection` inside `BaggingClassifier`) and `adaboost` (AdaBoost over decision trees of a chosen depth; stumps by default), both registered in `_MODELS`; the random forest is now the model `main.py` runs (200 trees, all CPU cores); `config.yaml` gains search spaces for the three ensembles -- the forests tune tree size (`max_depth`, `min_samples_leaf`) and decorrelation (`max_features` / number of combinations) but not the number of trees, AdaBoost tunes its base trees' depth, its number of rounds and its learning rate; `corrected_paired_test()` in `src/evaluate.py` compares two models scored on the same folds (Nadeau & Bengio's corrected paired t-test), so a "winner" is only called when the difference is bigger than fold-to-fold noise; `scipy` listed in `requirements.txt` (it already came with scikit-learn, so nothing new to install) -- see "Ensembles" below |
| 7 | Boosting + evaluating with probabilities: several metrics, one selection metric, a scoreboard | New models in `_MODELS`: `gradient_boosting` (now the default), `hist_gradient_boosting` and `naive_bayes`, with search spaces in `config.yaml`; every run reports **eight metrics** (`cv.metrics`, built by `build_scorers()`), while tuning maximises one **selection metric** (`cv.selection_metric: "roc_auc"`, replacing `cv.scoring`) -- see "Selection metric" below; CV and nested CV keep **out-of-fold probabilities**, so the threshold is an explicit choice, and the report gains a confusion matrix; nested CV uses the same outer folds as the default CV; a **scoreboard** (`results/runs.csv`, written by `save_scores()`) logs each run's per-fold scores, untuned score and no-model floor, so runs can be compared with `corrected_paired_test()` -- see "Scoreboard" below; the locked test set is scored only when `test_set.evaluate: true`; no new packages |
| 8 | XGBoost + feature selection | New model `xgboost` in `_MODELS` (now the default), with a search space that includes its in-split brakes (`gamma`, `reg_alpha`, `reg_lambda`, `min_child_weight`); new `VoteSelector` (random forest, gradient boosting and L1 logistic regression vote; keeps features with ≥ `min_votes`) added by `build_pipeline()` as a `"select"` step, so selection is re-fit in every fold; `memory=` caches fitted steps in a temporary folder per run; `config.yaml` gains `feature_selection` (with `select__min_votes` tuned in nested CV); `main.py` reports the kept features and logs the setting on the scoreboard; result: selection keeps only `age` and `priors_count` with no significant loss (ROC-AUC 0.725 vs 0.728, p ≈ 0.55), so it stays on; `xgboost>=2.0` in `requirements.txt` -- see "Feature selection" below |
| 9 | Beyond COMPAS -- option A: a new dataset (Adult income) | `main.py` takes the config file as an argument (`python main.py config_adult.yaml`; default `config.yaml`); new `config_adult.yaml` with its own output folder; placeholder tokens now become `NaN` in **every** text column (`_placeholders_to_nan()`), not only in `canonical_categories`; the fairness report's comparison with COMPAS's `score_text` moved to the config (`data.reference_score`); `check_feature_columns()` catches listed features that aren't in the data and reports unlisted columns; then every module audited for dataset-specific assumptions: new `src/config.py` checks a config on its own and against the data before anything is fitted; any labels (`encode_target()`) and two or more classes (probabilities per class, class-named reports, multiclass fairness, one-vs-rest L1 in `VoteSelector`); optional sensitive attribute; more file formats and reader options; `tests/test_datasets.py`; new `src/predict.py` and a `predict` config section predict new rows after the final refit (`predictions/adult_new_predictions.csv`); the COMPAS run gives exactly the same numbers -- see "Beyond COMPAS" above |


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

## Ensembles

Set `model.type` in `config.yaml` to choose the model; each type has a commented `params` example and its own search space under `tuning.search_spaces`.

| `model.type` | What it is | Search space (what the tuner chooses) |
|---|---|---|
| `random_forest` | many deep-ish trees on bootstrap samples, random features at every split, averaged | `max_depth`, `min_samples_leaf`, `max_features` |
| `oblique_random_forest` | the same, but each tree splits on random linear combinations of the features | the trees' `max_depth`, `min_samples_leaf`; `n_components` (combinations per tree) |
| `adaboost` | shallow trees built in sequence, each focused on the rows the previous ones got wrong; weighted vote | base-tree depth, `n_estimators`, `learning_rate` |

The forests' `n_estimators` is fixed in `params`, not tuned: more trees never overfit, they only cost time. Tuning an ensemble takes minutes per run (900 fits with the default settings) -- while testing, use a small `tuning.n_trials` or `tuning.enabled: false`.

**Comparing two models.** Every run uses the same CV folds (`cv.random_state`), so two runs' per-fold validation scores (in their `results/` files) can be compared fold by fold. Use `corrected_paired_test()`, from the project root:

```python
from src.evaluate import corrected_paired_test
adaboost = [0.674, 0.658, 0.701, 0.688, 0.691]      # validation column of one run's nested CV table
forest   = [0.675, 0.658, 0.693, 0.684, 0.696]      # ... and of another run's
corrected_paired_test(adaboost, forest, test_train_ratio=1 / 4)   # 5-fold: 1 / (5 - 1)
```

It returns the mean difference, the folds the first model wins, t and the p-value. Plain t-tests over CV folds are overconfident (the folds share most of their training rows); this one corrects for it. With only 5 folds, a difference of about one point is rarely significant: don't call a winner the test can't see.

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

To run another problem, pass its config file (week 9; see "Beyond COMPAS" above):
```bash
python main.py config_adult.yaml
```

To check that the pipeline still handles other kinds of datasets and rejects bad configs (a few seconds):
```bash
python tests/test_datasets.py
```

This loads `config.yaml`, diagnoses and cleans the data (week 3), locks the final test set away, cross-validates preprocessing + model on the development set (week 4), tunes the model's hyperparameters if `tuning.enabled` is on (week 5 -- see "Hyperparameter tuning" above; minutes, not seconds, for the ensembles of week 6), and prints:
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
