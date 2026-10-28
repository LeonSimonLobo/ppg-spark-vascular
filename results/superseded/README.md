# Superseded and Archived Results Index

This directory archives historical results and diagnostic artifacts from the initial between-subject investigation (`PLAN.md`). All entries here are preserved intact without modification for complete provenance and pre-registration integrity.

Under **Amendment A4** and the adoption of **`PLAN_W.md`**, the research question was reframed from between-subject demographic modeling to acute within-subject hemodynamic tracking.

---

## Inventory of Archived Results

| File / Item | Initial Scope | Status | Superseded / Cancelled By | Reason & Outcome |
|---|---|---|---|---|
| **`detector_agreement.md`** | Validation beat agreement under frozen Elgendi detector with [0, 60) lag window | **Superseded** | Amendment A3 | Initial agreement pass rate of 68.93% fell below the 80% pre-registered threshold. Analysis revealed the [0, 60) search window was one-sided and unable to represent physiological timing where ABP leads PPG. |
| **`ptt_sign_check.md`** | Synthetic unit test and empirical validation of cross-correlation sign convention | **Superseded** | Amendment A3 | Verified mathematical sign convention ($lag < 0$ corresponds to ABP leading PPG) and identified 56.88% boundary hits at lag 0 in the one-sided window. |
| **`detector_diagnostics.md`** | Exploratory diagnostic on validation segments | **Superseded** | Amendment A3 | Identified severe boundary clustering at lag 0 and beat count distributions informing Amendment A3. Labeled strictly exploratory. |
| **`detector_agreement_A3.md`** | Validation beat agreement under symmetric [-37, 37) lag window (Amendment A3) | **Superseded / Blocked** | Amendment A4 & PLAN_W.md | Validation agreement under the corrected symmetric window was 44.09% (< 80% threshold). This triggered the pre-registered detector failure rule, blocking between-subject morphology ladder steps. |
| **`validation_baselines.md`** | Steps 0a (age) and 0 (age, sex, BMI) logistic regression on validation split | **Superseded** | Amendment A4 & PLAN_W.md | Between-subject baselines yielded AUROC 0.5332 (Step 0a) and 0.5283 (Step 0). Demographic predictive power was effectively near chance. |
| **`baseline_sanity.md`** | Subject-level correlation between age and median SBP on validation subjects | **Superseded** | Amendment A4 | Demonstrated absent demographic-SBP relation ($r_s = +0.0145$, AUROC = 0.4327) in the VitalDB surgical cohort due to intraoperative hemodynamic interventions decoupling operative BP from chronic demographic factors. |
| **Test Cancellation under A4** | Steps 0a and 0 evaluation on held-out `VitalDB_CalFree_Test_Subset.mat` | **Cancelled** | Amendment A4 (`fcc38f0`) | Scoring Steps 0a/0 alone would have exhausted the held-out test cohort without testing any morphology hypothesis. CalFree was reserved pristine (zero test segments touched) for the within-subject study in `PLAN_W.md`. |

---

## Active Protocol
All ongoing and forward analyses are governed exclusively by **[`PLAN_W.md`](../../PLAN_W.md)**.
