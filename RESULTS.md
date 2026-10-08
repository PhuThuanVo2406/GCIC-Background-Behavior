# GCIC results: stages 1–6 (first full run, 2026-10-08)

Code: `/mnt/project-files/gcic/code/` (prep_oulad.py, prep_cutoffs.py, analysis.py). Every number below comes from a CSV in this folder.

## Setup (steps 9–10)
- 80/20 stratified split: 26,074 train / 6,519 test. Extra "future" test: train on 2013B, 2013J and 2014B, test on 2014J (11,260 rows).
- Early-cutoff tables for days 14, 28, 56 and 90 (`../clean/oulad_cutoff_day*.csv`). They use only data up to the cutoff and keep only students still enrolled on that day (28,119 to 25,562 rows). Students who had already withdrawn are left out, because their outcome is already known.

## Stage 1: Who succeeds (RQ1)
- Every background variable is linked to success (chi-square, all p < 0.001). The strongest are education (p ≈ 1e-158) and IMD (p ≈ 1e-148).
- Pass rate rises from 35.2% in the most deprived band to 57.5% in the least deprived.
- Behavior gaps are much larger. Successful vs unsuccessful students: submission rate 96% vs 28%, active days 91 vs 23, average score 77 vs 38. Mann-Whitney effect sizes r = 0.89, 0.80 and 0.69.

## Stage 2: Background vs behavior (H1)
Test AUC, with the 2014J future test in brackets:

| Model | Logistic reg. | Random forest | XGBoost |
|---|---|---|---|
| A: Background | 0.639 (0.627) | 0.621 (0.605) | 0.641 (0.625) |
| B: Behavior | 0.975 (0.973) | 0.978 (0.975) | 0.980 (0.976) |
| C: Both | 0.975 (0.974) | 0.978 (0.974) | 0.979 (0.976) |

**H1 supported:** behavior beats background by about 34 AUC points, and adding background to behavior adds nothing. **Caveat:** these are full-course totals, so part of the gap is leakage (students who withdraw stop submitting). Stage 5 is the fair version, and H1 still holds there.

## Stage 3: What matters (RQ1)
- SHAP on XGBoost model C: behavior carries **94%** of total importance, background 6%. Top features: submission rate, average score, active days, early clicks (days 0–28). The top background feature is education (rank 9).
- Decision-tree rules (depth 3, 91.9% test accuracy):
  1. Submitting fewer than ~37% of assessments means almost certain non-success (12 of 9,629 succeeded).
  2. Submitting more than ~86% with an average above ~59 means 94% success (10,084 of 10,770).
  3. Submitting 49–74% can still work if quiz activity is high (more than ~480 quiz clicks: 65% success vs 17%).
- Odds ratios (plain units, from a logistic model that adjusts for the other variables):
  - Each extra active week: ×1.13.
  - +10 points of submission rate: ×2.29.
  - +10 points of average score: ×2.22.
  - Each step up in education: ×1.19.
  - Each step toward a less deprived IMD band: ×1.02.
  - Each previous attempt: ×0.82.
  - Male vs female: ×0.82.
  - Early clicks show ×0.83 per 100, but that is an adjustment effect: alone, early clicks predict success. Don't present it as "early clicking hurts".

## Stage 4: Paths to success (RQ3)
- k = 4 has the best silhouette score among 3–5, but separation is weak (0.18). Present these as tendencies, not sharp types.
- The four clusters (13,385 successful students):
  - **Power users** (2,979): 138 active days, 4,400 clicks, submit far ahead of deadlines. 24% got a Distinction.
  - **Steady workers** (6,692): 108 active days, 2,060 clicks, the most forum-heavy group. 22% Distinction.
  - **Efficient minimalists** (4,880): only 43 active days and 590 clicks, but still submit 96% of assessments and score 72. 14% Distinction.
  - **Selective quizzers** (834): submit only 61% of assessments but are quiz-heavy and score 77. 18% Distinction.
- Background mix is almost the same in every cluster (mean IMD 4.4–4.8, similar education and age). There is more than one way to succeed, and it isn't decided by background.

## Stage 5: How early (RQ4, H3)
Test AUC (XGBoost):

| Cutoff day | 14 | 28 | 56 | 90 |
|---|---|---|---|---|
| A: Background | 0.635 | 0.625 | 0.641 | 0.638 |
| B: Behavior | 0.690 | 0.766 | 0.815 | 0.848 |
| C: Both | 0.726 | 0.781 | 0.826 | 0.855 |

**H3 partly supported.** By week 4, behavior predicts clearly better than background (0.77 vs 0.63), but it doesn't yet "separate most" students; accuracy is 71%. Prediction becomes strong (AUC 0.85) by day 90. Early on, background adds a little on top of behavior (0.69 → 0.73 at day 14).

## Stage 6: Deprivation gap (H2)
Pass rate, most vs least deprived band (0–10% vs 90–100%):
- **Overall:** 35.2% vs 57.5%, a gap of 22.4 points.
- **Within full-course active-days quartiles:** gaps of 0.4, 7.0, 17.2 and 9.0 points. The gap shrinks, but the lowest quartile is mostly withdrawals, so this is partly leakage.
- **Within first-28-days active-days quartiles** (students still enrolled at day 28): gaps of 15.5, 20.8, 15.8 and 19.0 points against 21.2 overall. It barely shrinks.
- **Logistic check:** the odds ratio per IMD band falls from 1.098 to 1.062 once active weeks are controlled. Engagement explains about a third of the deprivation effect.

**H2 partly supported.** Deprived students do pass less. Engagement narrows the gap somewhat, but students from deprived areas with the same early engagement still pass less often. That is an honest finding for the poster's economics angle.

## Fairness check (ethics section)
- Model C misses more true successes among the most deprived (5.8% vs 3.3% for the least deprived) and among disabled students (7.8% vs 5.0%).
- Overall accuracy is similar across groups (92–94%).
- Worth stating on the poster as a limitation.
