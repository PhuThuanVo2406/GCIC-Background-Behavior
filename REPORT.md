# Background or Behavior? Full EDA and model comparison on OULAD

**Goal:** find which features drive course success and which model predicts it best.
**Audience:** professors and GCIC conference judges.
**Data:** `../clean/oulad_clean.csv`, 32,593 student-course registrations from the Open University Learning Analytics Dataset (OULAD), built by `../code/prep_oulad.py`.
**Target:** `success` = 1 for Pass or Distinction, 0 for Fail or Withdrawn (binary classification).

Every number below comes from a file in `outputs/tables/`, and every chart is in `outputs/figures/` (PNG, 200 dpi). All randomness uses `random_state=42`.

### Assumptions and choices

1. **No rows dropped.** The clean table has no missing values or duplicates, so all 32,593 rows are used (0% dropped).
2. **Features.** 7 categorical columns (course module, gender, age band, highest education, IMD band, region, disability) and 14 numeric ones (previous attempts plus 13 behavior features). Left out: `id_student` (identifier), `code_presentation` (time period, would not generalise to new presentations), `final_result` (the outcome itself), and the `*_ord` / `region_*` columns (duplicates of the raw labels, which the pipeline one-hot encodes itself).
3. **Course module is included** as a feature. It is neither background nor behavior, so it is reported as its own "course" group.
4. **Behavior features are full-course totals.** Students who withdraw stop clicking and submitting, so these features partly record the outcome. That inflates every model's score. Step 7 measures how much with a day-28 version.
5. **"Top 10 + Other" rule.** Categories are lumped into "Other" only when that merges at least two of them. That means region (13 values) gets an "Other" bar, but IMD band (11 values) keeps its "Unknown" label visible.
6. **Correlations use Spearman** because most click features are heavily skewed. Pearson is saved alongside it in `outputs/tables/step4_pearson_corr.csv`.
7. **Model selection uses the mean 5-fold CV ROC-AUC on the training set.** The test set is used only once, for the final scores, so it does not influence which model is chosen.
8. **SVM scores** come from `decision_function`, not probabilities. That is fine for ROC and PR curves and avoids slow Platt scaling.

## Step 1 – Data audit

Figures: `step1_missing_values.png`, `step1_target_distribution.png`. Tables: `step1_column_audit.csv`, `step1_summary.json`.

**Key insights**
- The table has 32,593 rows and 45 columns (24 integer, 12 decimal, 9 text). It has no duplicate rows and no repeated student-module-presentation keys, so each row is one registration.
- The clean table has 0 missing values. The raw source was missing IMD band for 1,111 students (3.4%), now kept as an "Unknown" category, and 45 registration dates (0.1%), filled with the median. The 69% blank unregistration dates are structural: a blank means the student never withdrew.
- The target is close to balanced: 15,385 successful (47.2%) and 17,208 not successful (52.8%). Accuracy is therefore a fair metric, though F1 and ROC-AUC are reported too.
- "Not successful" mixes two different groups: 10,156 withdrawals (31% of all registrations) and 7,052 fails. Withdrawals are the source of the leakage problem: their median submission rate is 0 and their median is 7 active days, against 78 for passes.

## Step 2 – Univariate EDA

Figures: `step2_numeric_histograms.png`, `step2_numeric_boxplots.png`, `step2_categorical_counts.png`. Table: `step2_numeric_summary_outliers.csv`.

**Key insights**
- Every click feature is strongly right-skewed (skew 2.0 to 3.3), with 6–17% of values beyond the 1.5×IQR fences. These are real heavy users, not errors, because clicks were already capped at the 99th percentile during prep. They are kept, and scaling handles them for the linear and distance-based models.
- Submission rate and average score are bimodal. A spike at 0 comes from the 20.7% of registrations with no submissions at all, and a second peak sits at full submission (41.5% submitted everything).
- Some "outliers" are artefacts of narrow distributions. 87% of students have 0 previous attempts, so any resit is flagged (12.8%). Days submitted early is 0 for 26% of students, which gives it 22% flagged values.
- The course modules differ a lot in size (BBB 7,909 vs AAA 748). The students are 55% male, 70% under 35 and 90% without a declared disability, and 43% have A levels as their highest qualification.

## Step 3 – Feature vs target

Figures: `step3_numeric_vs_target_violins.png`, `step3_categorical_vs_target.png`, `step3_pairplot_top5.png`. Tables: `step3_numeric_tests.csv`, `step3_categorical_tests.csv`.

**Key insights**
- Behavior separates the outcomes far more than background. Submission rate has a point-biserial r of 0.81 with success (rank-biserial 0.89), active days 0.63, average score 0.61 and total clicks 0.51.
- Background effects are real but modest. Highest education has Cramér's V = 0.15 (30% success with no formal qualifications vs 65% with a postgraduate one), IMD band 0.15 (35% in the most deprived tenth vs 58% in the least deprived) and course module 0.14 (38% in CCC vs 71% in AAA). Gender barely matters (V = 0.02; 48% for women vs 46% for men).
- With 32,593 rows every test is significant (all p < 0.001), so effect sizes, not p-values, are what tell the features apart.
- Students with an unknown IMD band succeed most often (66%). They are concentrated in the North region and Ireland, so the missing value carries information and is kept as its own category rather than imputed away.
- The pairplot of the top 5 continuous features shows unsuccessful students piled up at zero on all five axes. Most of them are withdrawals.

## Step 4 – Heatmaps

Figures: `step4_correlation_heatmap.png`, `step4_cramers_v_heatmap.png`, `step4_pivot_education_imd.png`. Tables: `step4_spearman_corr.csv`, `step4_cramers_v.csv`, `step4_high_corr_pairs.csv`, `step4_pivot_education_imd_rate.csv`.

**Key insights**
- The click features are highly collinear: 15 pairs have |ρ| ≥ 0.8 (for example active days and "other" clicks at 0.97). This does not hurt prediction, but it spreads importance across correlated features and makes single logistic-regression coefficients unstable, so importance should be read for the group, not one column.
- Success correlates most with submission rate (ρ = 0.80), active days (0.69) and total clicks (0.65). Previous attempts (−0.11), studied credits (−0.13) and days registered before the start (−0.05) are weak.
- The one numeric background feature, previous attempts, is only weakly related to behavior (|ρ| ≤ 0.19, the largest being with studied credits). This heatmap cannot test the categorical background features against behavior; the research-plan analysis found the background mix nearly identical across its four behavior clusters.
- Among the categorical features, the strongest link is course module with gender (V = 0.60; some modules are mostly male or mostly female), followed by IMD band with region (0.21). Gender's raw gap may partly reflect which courses students take.
- Education and deprivation add up. Within every education level, success rises with less deprivation, for example A-level students go from 43% (most deprived) to 60% (least deprived) and HE-qualified students from 37% to 67%. Cells marked * hold fewer than 30 students.

## Step 5 – Preprocessing

Tables: `step5_preprocessing.json`, `step5_encoded_features.csv`, `step5_test_index.csv`.

**Key insights**
- The 80/20 stratified split gives 26,074 training and 6,519 test rows, both with a 47.2% success rate. It is the same split sizes as the research-plan analysis.
- Preprocessing is a single scikit-learn `ColumnTransformer` inside each model's `Pipeline`: median imputation and standard scaling for the 14 numeric features, most-frequent imputation and one-hot encoding for the 7 categorical ones (57 columns after encoding). Because it is inside the pipeline, it is re-fit on the training folds only during cross-validation, so nothing leaks from validation or test rows. A check in the script confirms the scaler's means equal the training means.
- No resampling or class weights were used. The minority class is 47.2%, which is not imbalanced enough for SMOTE or reweighting to help; they would only shift the decision threshold.
- The pipeline does not fix the main leakage risk, which is in the features themselves (full-course behavior). That is addressed in step 7.

## Step 6 – Modeling

Tables: `step6_model_results.csv`, `step6_best_params.json`, `step6_test_predictions.csv`. Model: `models/best_model.joblib`.

| Rank | Model | CV ROC-AUC (± SD) | Test ROC-AUC | Accuracy | F1 | Tuning time* |
|---|---|---|---|---|---|---|
| 1 | XGBoost | 0.9775 ± 0.0014 | 0.980 | 93.1% | 0.929 | 24 s |
| 2 | LightGBM | 0.9773 ± 0.0014 | 0.980 | 92.9% | 0.926 | 30 s |
| 3 | Gradient Boosting | 0.9772 ± 0.0015 | 0.980 | 92.7% | 0.925 | 200 s |
| 4 | Random Forest | 0.9762 ± 0.0019 | 0.979 | 92.6% | 0.924 | 183 s |
| 5 | Logistic Regression | 0.9742 ± 0.0019 | 0.976 | 92.4% | 0.921 | 4 s |
| 6 | SVM (RBF) | 0.9740 ± 0.0022 | 0.977 | 92.4% | 0.923 | 123 s |
| 7 | Decision Tree | 0.9724 ± 0.0022 | 0.975 | 91.9% | 0.916 | 16 s |
| 8 | KNN | 0.9612 ± 0.0025 | 0.965 | 90.4% | 0.905 | 41 s |

*Wall-clock time for the whole search on 4 CPU cores; it varies by a few seconds between runs. Scores are identical on every rerun.

**Key insights**
- XGBoost has the best cross-validated ROC-AUC (0.9775) and is the selected model, with a test ROC-AUC of 0.980 and 93.1% accuracy. The three boosting models are only 0.0003 apart, a fifth of one fold-to-fold standard deviation, so they are statistically tied. LightGBM would be an equally good choice.
- Model choice matters much less than the features. Seven of the eight models land between 0.972 and 0.978. Plain logistic regression is only 0.003 behind XGBoost and tunes in 4 seconds, so the relationships are mostly monotonic, and an interpretable model loses almost nothing. Its odds ratios (research-plan stage 3) are a fair companion to the SHAP results on a poster.
- KNN is clearly weakest (0.961). Distances over 57 mixed one-hot and scaled columns, many of them collinear click counts, blur the neighbourhoods, and it trades precision (0.851) for recall (0.966).
- Every model scores 0.002–0.004 higher on the test set than in CV. The gap is the same for all eight models, so it reflects a slightly easier test split, not overfitting to it.
- Two searches stopped at the edge of their grids (SVM at C = 3 and gamma = 0.01; KNN at k = 51). Widening them might add a little, but it would not close the gap to the boosters.

## Step 7 – Results visualization

Figures: `step7_model_comparison_bars.png`, `step7_roc_pr_curves.png`, `step7_confusion_matrices_top3.png`, `step7_learning_curve_best.png`, `step7_feature_importance_builtin.png`, `step7_shap_summary_beeswarm.png`, `step7_shap_bar.png`, `step7_summary_table.png`. Tables: `step7_shap_importance.csv`, `step7_shap_by_original_feature.csv`, `step7_shap_share_by_group.csv`, `step7_feature_importance_builtin.csv`, `step7_learning_curve.csv`, `step7_leakage_check_day28.json`.

**Key insights**
- **Behavior drives the predictions.** SHAP gives behavior features 92.4% of the best model's importance, the course module 3.9% and background 3.7%. Submission rate alone accounts for 45%, average score for 20% and active days for 10%. The top background feature is highest education (2.0%, 8th overall). IMD band and region add 0.6% each, and gender, age, disability and previous attempts are each under 0.2%. XGBoost's built-in total-gain importance agrees: 97% behavior, with submission rate alone at 61%.
- **Read the SHAP directions with care.** High submission rates, scores and active days push towards success. Early clicks (days 0–28) show the opposite sign once everything else is held fixed: among students with the same total activity, those whose clicks were front-loaded tended to fade out later. On its own, early clicking is positively related to success (ρ = 0.41), so this must not be presented as "clicking early hurts".
- **Errors.** XGBoost catches 95.6% of successful students and 90.8% of unsuccessful ones. Its 316 false "successes" (students predicted to pass who did not) outnumber the 136 missed successes. Those are the students an early-warning system would most need to catch.
- **More data would not help much.** On the learning curve, CV ROC-AUC only rises from 0.974 at 2,085 training rows to 0.977 at 20,859, and the train/CV gap narrows from 0.021 to 0.008. The model is not data-starved or badly overfit; better features, not more rows, are the lever.
- **Leakage check.** I refit the same XGBoost settings using only data up to day 28, on the students still enrolled that day (n = 27,538). Test ROC-AUC falls from 0.980 to 0.788, and accuracy from 93.1% to 72.5%. The drop comes from the features, not from the change of population: on its own test students who were still enrolled on day 28, the full-course model still scores 0.972. Most of the full-course accuracy therefore uses information that only exists later in the course, including the withdrawal itself. The full-course results answer "what distinguishes successful students?", but they do not measure early-warning accuracy.

## Consistency with the research-plan results

The numbers match `../results/RESULTS.md` closely. The research-plan models scored a test ROC-AUC of 0.979 (XGBoost), 0.978 (random forest) and 0.975 (logistic regression) with background + behavior features, against 0.980, 0.979 and 0.976 here. SHAP gave behavior 94% there and 92.4% here. The day-28 XGBoost scored 0.781 there and 0.788 here. The small differences come from two choices in this pipeline: the course module is included as a feature here (3.9% of SHAP importance) but was not in the earlier models, and each model's hyperparameters were tuned with 5-fold CV.
