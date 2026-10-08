"""Analysis stages 1-6 of the GCIC "Background or Behavior?" plan.

Reads the tables built by prep_oulad.py and prep_cutoffs.py and writes result
tables (CSV/TXT/JSON) to <results_dir>.

Usage: python analysis.py <clean_dir> <results_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text
from xgboost import XGBClassifier

SEED = 42
CUTOFFS = (14, 28, 56, 90)
CV = StratifiedKFold(5, shuffle=True, random_state=SEED)


def models():
    return {
        "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "Random forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=SEED),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
                                 colsample_bytree=0.8, n_jobs=-1, random_state=SEED, eval_metric="logloss"),
    }


def scores(model, X, y):
    p = model.predict_proba(X)[:, 1]
    return {"accuracy": accuracy_score(y, p >= 0.5), "f1": f1_score(y, p >= 0.5), "auc": roc_auc_score(y, p)}


def stage1(df, background_raw, behavior, out):
    rows = []
    for c in background_raw:
        tab = pd.crosstab(df[c], df["success"])
        chi2, p, _, _ = stats.chi2_contingency(tab)
        rates = df.groupby(c)["success"].agg(["mean", "size"]).reset_index()
        for _, r in rates.iterrows():
            rows.append({"variable": c, "group": r[c], "n": r["size"], "success_rate": r["mean"],
                         "chi2": chi2, "p_value": p})
    pd.DataFrame(rows).to_csv(out / "stage1_background_success_rates.csv", index=False)
    rows = []
    for c in behavior:
        a, b = df.loc[df.success == 1, c], df.loc[df.success == 0, c]
        u, p = stats.mannwhitneyu(a, b)
        rows.append({"variable": c, "mean_success": a.mean(), "mean_not_success": b.mean(),
                     "median_success": a.median(), "median_not_success": b.median(),
                     "effect_size_r": 1 - 2 * u / (len(a) * len(b)), "p_value": p})
    pd.DataFrame(rows).to_csv(out / "stage1_behavior_comparison.csv", index=False)


def stage2(train, test, future_train, future_test, sets, out):
    rows, fitted = [], {}
    for set_name, cols in sets.items():
        for algo, m in models().items():
            cv = cross_validate(m, train[cols], train["success"], cv=CV, scoring=["accuracy", "f1", "roc_auc"])
            m.fit(train[cols], train["success"])
            fitted[(set_name, algo)] = m
            t = scores(m, test[cols], test["success"])
            fm = models()[algo].fit(future_train[cols], future_train["success"])
            f = scores(fm, future_test[cols], future_test["success"])
            rows.append({"model": set_name, "algorithm": algo,
                         "cv_auc": cv["test_roc_auc"].mean(), "cv_auc_sd": cv["test_roc_auc"].std(),
                         "cv_accuracy": cv["test_accuracy"].mean(), "cv_f1": cv["test_f1"].mean(),
                         "test_auc": t["auc"], "test_accuracy": t["accuracy"], "test_f1": t["f1"],
                         "future_2014J_auc": f["auc"], "future_2014J_accuracy": f["accuracy"]})
            print(set_name, algo, round(t["auc"], 3))
    pd.DataFrame(rows).to_csv(out / "stage2_model_comparison.csv", index=False)
    return fitted


def stage3(fitted, train, test, sets, background, out):
    cols = sets["C: Both"]
    xgb = fitted[("C: Both", "XGBoost")]
    sample = test[cols].sample(min(3000, len(test)), random_state=SEED)
    sv = shap.TreeExplainer(xgb).shap_values(sample)
    imp = pd.DataFrame({"feature": cols, "mean_abs_shap": np.abs(sv).mean(axis=0)})
    imp["group"] = np.where(imp["feature"].isin(background), "background", "behavior")
    imp.sort_values("mean_abs_shap", ascending=False).to_csv(out / "stage3_shap_importance.csv", index=False)
    grp = imp.groupby("group")["mean_abs_shap"].sum()
    (grp / grp.sum()).to_csv(out / "stage3_shap_share_by_group.csv")

    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=200, random_state=SEED)
    tree.fit(train[cols], train["success"])
    (out / "stage3_decision_tree_rules.txt").write_text(
        f"Test accuracy {accuracy_score(test['success'], tree.predict(test[cols])):.3f}\n\n"
        + export_text(tree, feature_names=cols, show_weights=True))

    # Plain-unit logistic regression for odds ratios
    d = train.assign(active_weeks=train["active_days"] / 7, submission_rate_10pct=train["submission_rate"] * 10,
                     avg_score_10pts=train["avg_score"] / 10, early_clicks_100=train["early_clicks_d0_28"] / 100)
    feats = ["active_weeks", "submission_rate_10pct", "avg_score_10pts", "early_clicks_100", "imd_ord",
             "education_ord", "age_ord", "disability_yes", "num_of_prev_attempts", "gender_male"]
    lr = LogisticRegression(max_iter=5000).fit(d[feats], d["success"])
    unit = {"active_weeks": "per extra active week", "submission_rate_10pct": "per +10 points of submission rate",
            "avg_score_10pts": "per +10 points of average score", "early_clicks_100": "per +100 clicks in days 0-28",
            "imd_ord": "per IMD band step toward less deprived", "education_ord": "per education level step",
            "age_ord": "per age band step", "disability_yes": "disability yes vs no",
            "num_of_prev_attempts": "per previous attempt", "gender_male": "male vs female"}
    pd.DataFrame({"feature": feats, "odds_ratio": np.exp(lr.coef_[0]), "meaning": [unit[f] for f in feats]}) \
        .to_csv(out / "stage3_odds_ratios.csv", index=False)


def stage4(df, behavior, out):
    s = df[df["success"] == 1].copy()
    cols = ["active_days", "total_clicks", "early_clicks_d0_28", "clicks_quiz", "clicks_forum", "clicks_content",
            "avg_days_early", "submission_rate", "avg_score", "reg_days_before_start", "studied_credits"]
    X = s[cols].copy()
    for c in ["total_clicks", "early_clicks_d0_28", "clicks_quiz", "clicks_forum", "clicks_content"]:
        X[c] = np.log1p(X[c])
    X = StandardScaler().fit_transform(X)
    rows = []
    rng = np.random.RandomState(SEED)
    idx = rng.choice(len(X), min(6000, len(X)), replace=False)
    for k in range(2, 8):
        km = KMeans(k, n_init=10, random_state=SEED).fit(X)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(X[idx], km.labels_[idx])})
    sel = pd.DataFrame(rows)
    sel.to_csv(out / "stage4_k_selection.csv", index=False)
    k = int(sel[sel.k.between(3, 5)].sort_values("silhouette", ascending=False).iloc[0]["k"])
    s["cluster"] = KMeans(k, n_init=10, random_state=SEED).fit_predict(X)
    prof = s.groupby("cluster").agg(n=("success", "size"), **{c: (c, "mean") for c in cols},
                                    distinction_share=("final_result", lambda x: (x == "Distinction").mean()),
                                    imd_ord=("imd_ord", "mean"), education_ord=("education_ord", "mean"),
                                    age_35plus=("age_ord", lambda x: (x > 0).mean()),
                                    disability=("disability_yes", "mean"))
    prof.to_csv(out / "stage4_cluster_profiles.csv")
    s[["id_student", "code_module", "code_presentation", "cluster"]].to_csv(out / "stage4_cluster_assignments.csv", index=False)
    return k


def stage5(cut_dir, df, background, out):
    beh = ["studied_credits", "reg_days_before_start", "total_clicks", "active_days", "clicks_content",
           "clicks_forum", "clicks_other", "clicks_quiz", "avg_days_early", "submission_rate", "avg_score",
           "any_due", "no_submissions"]
    keys = ["id_student", "code_module", "code_presentation"]
    rows = []
    for day in CUTOFFS:
        c = pd.read_csv(cut_dir / f"oulad_cutoff_day{day}.csv").merge(df[keys + background], on=keys)
        tr, te = train_test_split(c, test_size=0.2, stratify=c["success"], random_state=SEED)
        for set_name, cols in {"A: Background": background, "B: Behavior": beh, "C: Both": beh + background}.items():
            for algo in ["Logistic regression", "XGBoost"]:
                m = models()[algo]
                cv = cross_validate(m, tr[cols], tr["success"], cv=CV, scoring="roc_auc")
                m.fit(tr[cols], tr["success"])
                rows.append({"cutoff_day": day, "n": len(c), "success_rate": c["success"].mean(),
                             "model": set_name, "algorithm": algo, "cv_auc": cv["test_score"].mean(),
                             "test_auc": roc_auc_score(te["success"], m.predict_proba(te[cols])[:, 1]),
                             "test_accuracy": accuracy_score(te["success"], m.predict(te[cols]))})
        print("cutoff", day, "done")
    pd.DataFrame(rows).to_csv(out / "stage5_auc_by_cutoff.csv", index=False)


def stage6(df, cut_dir, out):
    keys = ["id_student", "code_module", "code_presentation"]
    ext = df[df["imd_band"].isin(["0-10%", "90-100%"])].copy()
    ext["deprivation"] = np.where(ext["imd_band"] == "0-10%", "most deprived (0-10%)", "least deprived (90-100%)")
    rows = []

    def add(frame, label, qcol):
        for name, g in [("all", frame)] + list(frame.groupby(qcol, observed=True)):
            r = g.groupby("deprivation")["success"].agg(["mean", "size"])
            if len(r) < 2:
                continue
            most, least = r.loc["most deprived (0-10%)"], r.loc["least deprived (90-100%)"]
            rows.append({"engagement_measure": label, "engagement_level": str(name),
                         "pass_rate_most_deprived": most["mean"], "n_most_deprived": most["size"],
                         "pass_rate_least_deprived": least["mean"], "n_least_deprived": least["size"],
                         "gap_points": 100 * (least["mean"] - most["mean"])})

    # Quartiles are set on all students, then applied to the two extreme bands
    q_full = pd.qcut(df["active_days"], 4, labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"])
    ext["q_full"] = q_full.loc[ext.index]
    add(ext, "active days, full course", "q_full")

    c28 = pd.read_csv(cut_dir / "oulad_cutoff_day28.csv")
    c28 = c28.merge(df[keys + ["imd_band"]], on=keys)
    c28["q28"] = pd.qcut(c28["active_days"].rank(method="first"), 4, labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"])
    e28 = c28[c28["imd_band"].isin(["0-10%", "90-100%"])].copy()
    e28["deprivation"] = np.where(e28["imd_band"] == "0-10%", "most deprived (0-10%)", "least deprived (90-100%)")
    add(e28, "active days, first 28 days (still enrolled at day 28)", "q28")
    pd.DataFrame(rows).to_csv(out / "stage6_deprivation_gap.csv", index=False)

    # Logistic check: does the IMD effect shrink once engagement is held constant?
    d = df.assign(active_weeks=df["active_days"] / 7)
    or_raw = np.exp(LogisticRegression(max_iter=5000).fit(d[["imd_ord"]], d["success"]).coef_[0][0])
    or_adj = np.exp(LogisticRegression(max_iter=5000).fit(d[["imd_ord", "active_weeks"]], d["success"]).coef_[0][0])
    return {"imd_odds_ratio_per_band_raw": or_raw, "imd_odds_ratio_per_band_given_active_weeks": or_adj}


def fairness(fitted, test, sets, out):
    m = fitted[("C: Both", "XGBoost")]
    t = test.assign(pred=m.predict(test[sets["C: Both"]]))
    rows = []
    for col in ["imd_band", "gender", "disability"]:
        for g, x in t.groupby(col):
            pos, neg = x[x.success == 1], x[x.success == 0]
            rows.append({"variable": col, "group": g, "n": len(x), "accuracy": (x.pred == x.success).mean(),
                         "missed_successes_rate": (pos.pred == 0).mean() if len(pos) else np.nan,
                         "missed_at_risk_rate": (neg.pred == 1).mean() if len(neg) else np.nan})
    pd.DataFrame(rows).to_csv(out / "fairness_error_rates.csv", index=False)


def main(clean, out):
    clean, out = Path(clean), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(clean / "oulad_clean.csv")
    background_all = Path(clean / "features_background.txt").read_text().split()
    behavior = Path(clean / "features_behavior.txt").read_text().split()
    # Drop one region dummy so the logistic regression is not over-specified
    background = [c for c in background_all if c != "region_East_Anglian_Region"]
    sets = {"A: Background": background, "B: Behavior": behavior, "C: Both": background + behavior}

    # Step 10: 80/20 stratified split, plus 2014J as a "future" presentation
    train, test = train_test_split(df, test_size=0.2, stratify=df["success"], random_state=SEED)
    future_train, future_test = df[df.code_presentation != "2014J"], df[df.code_presentation == "2014J"]
    pd.DataFrame({"set": ["train", "test", "future_train", "future_test_2014J"],
                  "rows": [len(train), len(test), len(future_train), len(future_test)]}) \
        .to_csv(out / "split_sizes.csv", index=False)

    stage1(df, ["gender", "age_band", "highest_education", "imd_band", "region", "disability"], behavior, out)
    fitted = stage2(train, test, future_train, future_test, sets, out)
    stage3(fitted, train, test, sets, background, out)
    fairness(fitted, test, sets, out)
    k = stage4(df, behavior, out)
    stage5(clean, df, background, out)
    gap = stage6(df, clean, out)
    (out / "summary.json").write_text(json.dumps({"clusters_k": k, **gap}, indent=2, default=float))


if __name__ == "__main__":
    main(*sys.argv[1:3])
