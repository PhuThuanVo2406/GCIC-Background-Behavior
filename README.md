# Clean OULAD table (data prep steps 1–8)

`oulad_clean.csv`: 32,593 rows (one per student-course registration), 45 columns, no blanks. Built by `/mnt/project-files/gcic/code/prep_oulad.py` from the raw CSVs in `/mnt/project-files/data/oulad/`.

- **Keys and target:** id_student, code_module, code_presentation, starts_feb (1 = B / February presentation), final_result, success (1 = Pass or Distinction).
- **Background features** (`features_background.txt`): gender_male, age_ord (0-35=0, 35-55=1, 55+=2), education_ord (No formal=0 to Post grad=4), imd_ord (0 = 0–10% most deprived to 9 = 90–100%), imd_unknown, disability_yes, num_of_prev_attempts, region_* one-hot (all 13; drop one for logistic regression).
- **Behavior features** (`features_behavior.txt`): studied_credits, reg_days_before_start, total_clicks, active_days, early_clicks_d0_28, clicks_quiz / clicks_forum / clicks_content / clicks_other, avg_days_early (deadline minus submit day; banked excluded), submission_rate (submitted / non-exam assessments), avg_score (weight-weighted non-exam score; plain mean when all weights are 0), no_submissions.
- **Raw labels kept** for charts: gender, age_band, highest_education, imd_band ("Unknown" for missing), region, disability.

Choices made:
- Activity groups: quiz = quiz, externalquiz; forum = forumng, oucollaborate, ouelluminate, ouwiki; content = oucontent, resource, page, subpage, sharedsubpage, url, dualpane, folder, htmlactivity, glossary; other = homepage, questionnaire, dataplus, repeatactivity.
- No clicks or submissions get 0. no_submissions flags the 0 avg_score / avg_days_early that means "nothing submitted".
- Missing IMD: imd_band = "Unknown", imd_ord = median, imd_unknown = 1. 45 missing registration dates use the median.
- Click columns capped at the 99th percentile.
- These are full-course totals, so they include the leakage the plan warns about. Early-cutoff features (stage 5) still need building.
