"""Step 9: behavior features using only data up to a cutoff day (early-warning models, stage 5).

Each table keeps only registrations still enrolled at the cutoff day: students who
already withdrew are a known outcome, so including them would inflate early accuracy.

Usage: python prep_cutoffs.py <oulad_csv_dir> <output_dir>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from prep_oulad import ACTIVITY_GROUP, KEYS, read

CUTOFFS = (14, 28, 56, 90)


def build_cutoff(info, reg, assess, sassess, svle, day):
    base = info[KEYS + ["final_result", "studied_credits"]].merge(reg, on=KEYS)
    base = base[base["date_unregistration"].isna() | (base["date_unregistration"] > day)].copy()
    base["success"] = base["final_result"].isin(["Pass", "Distinction"]).astype(int)
    base["reg_days_before_start"] = (-base["date_registration"]).fillna(-reg["date_registration"].median())

    v = svle[svle["date"] <= day]
    g = v.groupby(KEYS, observed=True)
    clicks = pd.DataFrame({"total_clicks": g["sum_click"].sum(), "active_days": g["date"].nunique()})
    by_type = v.pivot_table(index=KEYS, columns="group", values="sum_click", aggfunc="sum", observed=True)
    by_type.columns = [f"clicks_{c}" for c in by_type.columns]
    clicks = clicks.join(by_type).reset_index()

    due = assess[(assess["assessment_type"] != "Exam") & (assess["date"] <= day)]
    n_due = due.groupby(["code_module", "code_presentation"]).size().rename("n_due").reset_index()
    sub = sassess.merge(due, on="id_assessment")
    sub = sub[sub["date_submitted"] <= day]
    sub["days_early"] = sub["date"] - sub["date_submitted"]
    g = sub.groupby(KEYS)
    subs = pd.DataFrame({
        "n_submitted": g.size(),
        "avg_days_early": sub[sub["is_banked"] == 0].groupby(KEYS)["days_early"].mean(),
        "avg_score": g["score"].mean(),
    }).reset_index()

    df = base.merge(clicks, on=KEYS, how="left").merge(subs, on=KEYS, how="left")
    df = df.merge(n_due, on=["code_module", "code_presentation"], how="left")
    for c in ["clicks_content", "clicks_forum", "clicks_other", "clicks_quiz"]:
        if c not in df:
            df[c] = 0
    fill0 = ["total_clicks", "active_days", "clicks_content", "clicks_forum", "clicks_other",
             "clicks_quiz", "n_submitted", "n_due"]
    df[fill0] = df[fill0].fillna(0)
    df["submission_rate"] = np.where(df["n_due"] > 0, (df["n_submitted"] / df["n_due"].replace(0, np.nan)).clip(upper=1), 0)
    df["any_due"] = (df["n_due"] > 0).astype(int)
    df["no_submissions"] = (df["n_submitted"] == 0).astype(int)
    df[["avg_days_early", "avg_score"]] = df[["avg_days_early", "avg_score"]].fillna(0)
    for c in ["total_clicks", "clicks_content", "clicks_forum", "clicks_other", "clicks_quiz"]:
        df[c] = df[c].clip(upper=df[c].quantile(0.99))
    df["cutoff_day"] = day
    cols = KEYS + ["cutoff_day", "final_result", "success", "studied_credits", "reg_days_before_start",
                   "total_clicks", "active_days", "clicks_content", "clicks_forum", "clicks_other",
                   "clicks_quiz", "avg_days_early", "submission_rate", "avg_score", "any_due", "no_submissions"]
    return df[cols]


def main(src, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
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
    svle = svle.merge(vle, on="id_site", how="left")
    svle["group"] = svle["activity_type"].map(ACTIVITY_GROUP).fillna("other")
    for day in CUTOFFS:
        df = build_cutoff(info, reg, assess, sassess, svle, day)
        df.to_csv(out / f"oulad_cutoff_day{day}.csv", index=False)
        print(day, df.shape, round(df["success"].mean(), 3))


if __name__ == "__main__":
    main(*sys.argv[1:3])
