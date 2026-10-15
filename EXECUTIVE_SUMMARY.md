# Background or Behavior? Executive summary

**Question.** Does what students *do* in an online course predict success better than who they *are*?
**Data.** 32,593 student-course registrations from the Open University Learning Analytics Dataset (OULAD, 7 courses, 2013–2014). Success means Pass or Distinction, which applies to 47.2% of registrations. No rows were dropped.

## Findings

1. **Behavior dominates.** Submission rate has a point-biserial correlation of 0.81 with success, active days 0.63 and average score 0.61. The strongest background factors, education and neighbourhood deprivation, reach a Cramér's V of only 0.15.
2. **Background still matters, but modestly.** Success rises from 30% with no formal qualifications to 65% with a postgraduate degree, and from 35% in the most deprived tenth of areas to 58% in the least deprived. The two add up: within every education level, less deprived students succeed more often.
3. **The best model is very accurate, and simple models are nearly as good.** Eight tuned models were compared with 5-fold CV. XGBoost scored highest (CV ROC-AUC 0.978; test ROC-AUC 0.980, accuracy 93.1%, F1 0.929). LightGBM and gradient boosting are statistically tied with it, and logistic regression is only 0.003 behind. The signal sits in the features, not in model complexity.
4. **Explaining the model confirms H1.** SHAP gives behavior features 92% of the best model's importance, the course module 4% and background 4%. Submission rate alone accounts for 45%. The top background feature, education, ranks 8th.
5. **Early prediction is harder.** Using only the first 28 days of data, the same model reaches a ROC-AUC of 0.79 (72.5% accuracy). That is still well above background alone (0.63 in the research-plan analysis), but far below the full-course 0.98.

## Limitations

- **Leakage in full-course features.** Students who withdraw stop clicking and submitting, so full-course totals partly record the outcome. On the same students (those still enrolled on day 28), full-course features score 0.97 against 0.79 for day-28 features. The 0.98 result describes what separates successful students; it is not a forecast.
- **Association, not causation.** The data are observational. Submitting more is linked to passing, but this study cannot show that making students submit more would raise pass rates.
- **One institution and period.** OULAD covers seven Open University courses from 2013–2014. Results may not carry over to other universities, subjects or years.
- **Proxies and confounding.** IMD is an area-level deprivation measure, not household income. Gender is strongly tied to course choice (Cramér's V 0.60), so raw gender gaps partly reflect which course a student took.
- **Fairness.** The research-plan analysis found the model misses more true successes among the most deprived (5.8% vs 3.3%) and among disabled students (7.8% vs 5.0%). Any real use would need a per-group audit.

## Next steps

1. Make the day-28 and day-56 models the headline "early warning" result, and keep the full-course model as the explanation of what success looks like.
2. Validate on a later presentation (train on 2013–2014B, test on 2014J), as the research-plan analysis did, to show the model holds up over time.
3. Calibrate the probabilities and choose an alert threshold based on how costly a missed at-risk student is compared with a false alarm.
4. For the poster, use the SHAP bar chart, the success-by-category bars (education and IMD), the education × IMD heatmap, the model comparison table and the day-28 vs full-course contrast.

*Full method and figures: `REPORT.md` and `outputs/figures/`. Interactive version: `streamlit run dashboard/app.py`.*
