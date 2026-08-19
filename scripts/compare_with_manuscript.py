# -*- coding: utf-8 -*-
"""PHASE 9: old manuscript / legacy CSVs vs. the reproduced results.

    python scripts/compare_with_manuscript.py

Writes results/tables/old_vs_new_comparison.csv and .md.
This script only REPORTS differences; it never tunes anything to close them.
"""
from __future__ import annotations

import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KO2EN = {"태백": "Taebaek", "대관령": "Daegwallyeong", "백령도": "Baengnyeongdo",
         "인천": "Incheon", "강화": "Ganghwa", "동두천": "Dongducheon", "파주": "Paju",
         "철원": "Cheorwon", "춘천": "Chuncheon", "인제": "Inje", "속초": "Sokcho",
         "서울": "Seoul"}


def legacy_frames():
    d = os.path.join(ROOT, "reference", "legacy_results")
    ov = pd.read_csv(os.path.join(d, "model_comparison_by_region.csv"), encoding="utf-8-sig")
    on = pd.read_csv(os.path.join(d, "onset_model_comparison_by_region.csv"), encoding="utf-8-sig")
    for f in (ov, on):
        f["station"] = f["region"].map(KO2EN)
    return ov, on


def main() -> int:
    res = os.path.join(ROOT, "results")
    new = pd.read_csv(os.path.join(res, "metrics", "station_metrics.csv"))
    claims = pd.read_csv(os.path.join(ROOT, "reference", "manuscript_v5_claims.csv"))
    leg_ov, leg_on = legacy_frames()

    new_f1 = new.pivot_table(index=["task", "station"], columns="model", values="f1")
    rows = []

    def push(item, task, station, model, ms, legacy, repro):
        rows.append({
            "item": item, "task": task, "station": station, "model": model,
            "manuscript_v5": ms,
            "legacy_result_csv": legacy,
            "reproduced": None if repro is None else round(float(repro), 4),
            "diff_repro_minus_manuscript": None if (repro is None or ms is None)
                                           else round(float(repro) - float(ms), 4),
            "diff_legacy_minus_manuscript": None if (legacy is None or ms is None)
                                            else round(float(legacy) - float(ms), 4),
        })

    # per-station overall F1
    t3 = claims[(claims["source"] == "Table3") & (~claims["station"].isin(["MEAN", "SD"]))]
    for _, r in t3.iterrows():
        st, mdl = r["station"], r["model"]
        lg = leg_ov[(leg_ov["station"] == st) & (leg_ov["model"] == mdl)]["f1"]
        rp = new_f1.loc[("overall", st), mdl] if ("overall", st) in new_f1.index else None
        push("Table 3 F1", "overall", st, mdl, float(r["value"]),
             float(lg.iloc[0]) if len(lg) else None, rp)

    # means
    for mdl in ["Persistence", "XGBoost", "1D-CNN", "LSTM"]:
        ms = float(claims[(claims["source"] == "Table3") & (claims["station"] == "MEAN")
                          & (claims["model"] == mdl)]["value"].iloc[0])
        lg = leg_ov[leg_ov["model"] == mdl]["f1"].mean()
        rp = new_f1.loc["overall"][mdl].mean()
        push("Table 3 mean F1", "overall", "MEAN(12)", mdl, ms, lg, rp)

    for mdl in ["XGBoost-Onset", "1D-CNN-Onset", "LSTM-Onset", "No-Onset Baseline"]:
        sel = claims[(claims["source"] == "Table4") & (claims["station"] == "MEAN")
                     & (claims["model"] == mdl) & (claims["metric"] == "f1")]
        ms = float(sel["value"].iloc[0]) if len(sel) else None
        lg = leg_on[leg_on["model"] == mdl]["f1"].mean()
        rp = new_f1.loc["onset"][mdl].mean() if mdl in new_f1.loc["onset"].columns else None
        push("Table 4 mean F1", "onset", "MEAN(12)", mdl, ms, lg, rp)

    new_pr = new.pivot_table(index=["task", "station"], columns="model", values="pr_auc")
    for mdl in ["XGBoost-Onset", "1D-CNN-Onset", "LSTM-Onset"]:
        sel = claims[(claims["source"] == "Table4") & (claims["metric"] == "pr_auc")
                     & (claims["model"] == mdl)]
        ms = float(sel["value"].iloc[0]) if len(sel) else None
        lg = leg_on[leg_on["model"] == mdl]["pr_auc"].mean()
        rp = new_pr.loc["onset"][mdl].mean()
        push("Table 4 mean PR-AUC", "onset", "MEAN(12)", mdl, ms, lg, rp)

    out = pd.DataFrame(rows)

    def status(r):
        if r["reproduced"] is None:
            return "E: cannot reproduce"
        if r["legacy_result_csv"] is None:
            return "C: no legacy source found"
        if abs(r["diff_legacy_minus_manuscript"]) <= 0.005:
            return ("A: manuscript matches a stored result"
                    if abs(r["diff_repro_minus_manuscript"]) <= 0.02
                    else "B: manuscript traceable, but reproduction differs")
        return "D/E: manuscript value has NO matching stored result"

    out["provenance_status"] = out.apply(status, axis=1)

    t = os.path.join(res, "tables")
    os.makedirs(t, exist_ok=True)
    out.to_csv(os.path.join(t, "old_vs_new_comparison.csv"), index=False)

    with open(os.path.join(t, "old_vs_new_comparison.md"), "w", encoding="utf-8") as fh:
        fh.write("# Old manuscript vs. legacy result files vs. reproduced results\n\n")
        fh.write("`manuscript_v5` = value printed in Atmosphere_eng_v5.docx  \n")
        fh.write("`legacy_result_csv` = value found in the pre-existing result CSVs  \n")
        fh.write("`reproduced` = value produced by this repository\n\n")
        fh.write(out.to_markdown(index=False))
        fh.write("\n")

    print(out.groupby("provenance_status").size().to_string())
    print(f"\nWrote {t}/old_vs_new_comparison.csv|md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
