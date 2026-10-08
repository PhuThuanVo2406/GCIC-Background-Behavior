"""Build the clean OULAD table for the GCIC "Background or Behavior?" study.

Data preparation steps 1-8 of the research plan: one row per student-course
registration (32,593 rows), one column per background / behavior variable.

Usage: python prep_oulad.py <oulad_csv_dir> <output_dir>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["id_student", "code_module", "code_presentation"]
EARLY_DAYS = (0, 28)

# Step 8 orderings
EDU_ORDER = {
    "No Formal quals": 0,
    "Lower Than A Level": 1,
    "A Level or Equivalent": 2,
    "HE Qualification": 3,
    "Post Graduate Qualification": 4,
}
AGE_ORDER = {"0-35": 0, "35-55": 1, "55<=": 2}
# "10-20" is spelled without a % sign in the raw data
IMD_ORDER = {
    "0-10%": 0, "10-20": 1, "10-20%": 1, "20-30%": 2, "30-40%": 3, "40-50%": 4,
    "50-60%": 5, "60-70%": 6, "70-80%": 7, "80-90%": 8, "90-100%": 9,
}
# Step 4: map the 20 VLE activity types into the plan's resource groups
ACTIVITY_GROUP = {
    "quiz": "quiz", "externalquiz": "quiz",
    "forumng": "forum", "oucollaborate": "forum", "ouelluminate": "forum", "ouwiki": "forum",
    "oucontent": "content", "resource": "content", "page": "content", "subpage": "content",
    "sharedsubpage": "content", "url": "content", "dualpane": "content", "folder": "content",
    "htmlactivity": "content", "glossary": "content",
}


def read(d, name, **kw):
    return pd.read_csv(Path(d) / f"{name}.csv", na_values=["?"], **kw)


def build(src):
    # Step 1: load
    info = read(src, "studentInfo")
    reg = read(src, "studentRegistration")
    assess = read(src, "assessments")
    sassess = read(src, "studentAssessment")
    vle = read(src, "vle", usecols=["id_site", "activity_type"])
    svle = read(
        src, "studentVle",
        usecols=["code_module", "code_presentation", "id_student", "id_site", "date", "sum_click"],
        dtype={"code_module": "category", "code_presentation": "category",
               "id_student": "int32", "id_site": "int32", "date": "int16", "sum_click": "int32"},
    )

    # Step 2: target
    df = info.copy()
    df["success"] = df["final_result"].isin(["Pass", "Distinction"]).astype(int)

    # Step 3: click aggregates
    g = svle.groupby(KEYS, observed=True)
    clicks = pd.DataFrame({
        "total_clicks": g["sum_click"].sum(),
        "active_days": g["date"].nunique(),
    })
    early = svle[svle["date"].between(*EARLY_DAYS)]
    clicks["early_clicks_d0_28"] = early.groupby(KEYS, observed=True)["sum_click"].sum()

    # Step 4: clicks by resource type
    svle = svle.merge(vle, on="id_site", how="left")
    svle["group"] = svle["activity_type"].map(ACTIVITY_GROUP).fillna("other")
    by_type = svle.pivot_table(index=KEYS, columns="group", values="sum_click",
                               aggfunc="sum", observed=True)
    by_type.columns = [f"clicks_{c}" for c in by_type.columns]
    clicks = clicks.join(by_type).reset_index()

    # Step 5: assessments (non-exam only)
    course_assess = assess[assess["assessment_type"] != "Exam"]
    n_due = course_assess.groupby(["code_module", "code_presentation"]).size().rename("n_assessments")
    sub = sassess.merge(course_assess, on="id_assessment", how="inner")
    sub["days_early"] = sub["date"] - sub["date_submitted"]
    sub["w_score"] = sub["score"] * sub["weight"]
    sub["w_scored"] = sub["weight"].where(sub["score"].notna())
    g = sub.groupby(KEYS)
    subs = pd.DataFrame({
        "n_submitted": g.size(),
        # Banked submissions were carried over from an earlier attempt, so their timing is not this course's behavior
        "avg_days_early": sub[sub["is_banked"] == 0].groupby(KEYS)["days_early"].mean(),
        "w_score_sum": g["w_score"].sum(),
        "w_sum": g["w_scored"].sum(),
        "plain_avg_score": g["score"].mean(),
    })
    # Weighted by assessment weight; modules whose submissions all carry weight 0 fall back to the plain mean
    subs["avg_score"] = np.where(subs["w_sum"] > 0, subs["w_score_sum"] / subs["w_sum"].replace(0, np.nan),
                                 subs["plain_avg_score"])
    subs = subs[["n_submitted", "avg_days_early", "avg_score"]].reset_index()

    # Step 6: join everything onto studentInfo
    df = df.merge(reg[KEYS + ["date_registration"]], on=KEYS, how="left")
    df["reg_days_before_start"] = -df["date_registration"]
    df = df.merge(clicks, on=KEYS, how="left").merge(subs, on=KEYS, how="left")
    df = df.merge(n_due.reset_index(), on=["code_module", "code_presentation"], how="left")
    click_cols = [c for c in df.columns if c.startswith("clicks_")] + ["total_clicks", "active_days", "early_clicks_d0_28"]
    df[click_cols + ["n_submitted"]] = df[click_cols + ["n_submitted"]].fillna(0)
    df["submission_rate"] = (df["n_submitted"] / df["n_assessments"]).clip(upper=1)
    # No submissions: no timing or score to average. 0 per the plan; flag kept so models can tell it apart
    df["no_submissions"] = (df["n_submitted"] == 0).astype(int)
    df[["avg_days_early", "avg_score"]] = df[["avg_days_early", "avg_score"]].fillna(0)
    # 45 registrations have no registration date; use the median
    df["reg_days_before_start"] = df["reg_days_before_start"].fillna(df["reg_days_before_start"].median())

    # Step 7: clean
    assert not df.duplicated(KEYS).any()
    df["imd_band"] = df["imd_band"].fillna("Unknown")
    for c in click_cols:
        df[c] = df[c].clip(upper=df[c].quantile(0.99))

    # Step 8: encode
    df["education_ord"] = df["highest_education"].map(EDU_ORDER)
    df["age_ord"] = df["age_band"].map(AGE_ORDER)
    df["imd_unknown"] = (df["imd_band"] == "Unknown").astype(int)
    df["imd_ord"] = df["imd_band"].map(IMD_ORDER)
    df["imd_ord"] = df["imd_ord"].fillna(df["imd_ord"].median())
    df["disability_yes"] = (df["disability"] == "Y").astype(int)
    df["gender_male"] = (df["gender"] == "M").astype(int)
    df["starts_feb"] = df["code_presentation"].str.endswith("B").astype(int)
    region = pd.get_dummies(df["region"], prefix="region", dtype=int)
    region.columns = [c.replace(" ", "_") for c in region.columns]
    df = pd.concat([df, region], axis=1)

    background = ["gender_male", "age_ord", "education_ord", "imd_ord", "imd_unknown",
                  "disability_yes", "num_of_prev_attempts"] + list(region.columns)
    behavior = ["studied_credits", "reg_days_before_start", "total_clicks", "active_days",
                "early_clicks_d0_28"] + sorted(c for c in df.columns if c.startswith("clicks_")) + \
               ["avg_days_early", "submission_rate", "avg_score", "no_submissions"]
    ids = KEYS + ["starts_feb", "final_result", "success"]
    raw = ["gender", "age_band", "highest_education", "imd_band", "region", "disability"]
    return df[ids + background + behavior + raw], background, behavior


def main(src, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    df, background, behavior = build(src)
    df.to_csv(out / "oulad_clean.csv", index=False)
    pd.Series(background).to_csv(out / "features_background.txt", index=False, header=False)
    pd.Series(behavior).to_csv(out / "features_behavior.txt", index=False, header=False)
    print(df.shape)


if __name__ == "__main__":
    main(*sys.argv[1:3])
