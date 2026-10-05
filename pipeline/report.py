"""The paper's PRESENTATION layer: Tables 1-4 + the full paper figure
suite, assembled entirely by
READING artifacts other stages already produced. This module never re-fits
MaxEnt, never re-runs screening, and never calls any ``*.run()``/
``*.fit_*()`` that would recompute a heavy stage -- those artifacts already
exist on disk from the earlier stages (config/QA/HWM/predictors/screening/MaxEnt
fit-eval/model-comparison/robustness/sensitivity/equity). The few functions
here that DO touch a raster (``evaluate.gfi_baseline()``, the honest-OOS
ROC/PR curve, the mean/uncertainty map) are pure, deterministic, cheap
DERIVED reads/statistics over already-fit models -- never a re-fit, never a
re-run of screening/sensitivity/robustness/model-comparison. Every value
returned below traces to a named artifact file (or to a cheap, seeded,
tested pipeline function), so this module stays fast and
reproducible.

**Tables** (``table1()``..``table4()``, ``tables()``):
  - Table 1 -- flood-observation inventory (``hwm.raw_counts()`` /
    ``hwm.build_presence()``).
  - **Table 2 -- predictor stack** (``table2()``). ``PREDICTOR_CATALOG``
    below holds the Family/Source/Type columns of the original draft's
    21-variable predictor table. ``table2()`` JOINS that catalog with the LIVE
    bivariate effect sizes (``screen.BIVARIATE_REPORT_PATH``) and the LIVE
    screening retained/dropped result (``screen.REPORT_PATH``'s ``dropped``
    column -- the real, current on-disk output of the last
    ``screen.multicollinearity()`` run, never re-run here) to show the
    TRUE canonical stack: 17 modeled predictors, not the draft's 21.
    The catalog also carries 3 entries the draft's table omitted
    (``tri``, ``tpi`` -- real raw layers this pipeline acquired and
    screened out; ``median_income`` -- acquired, screened IN, then dropped
    at the modeling stage for a structural ACS coverage gap) so the
    "full considered stack" is honestly 24, not 21. See ``table2()``'s own
    docstring and its returned ``df.attrs["note"]``.
  - Table 3 -- variable importance (``maxent.VARIABLE_IMPORTANCE_CSV``).
    Takes an optional ``top_n`` (paper caption: "top 10"); default
    returns the full 17 rows.
  - Table 4 -- ascertainment/sensitivity arms (``sensitivity.SUMMARY_CSV``).
  - ``tables()`` writes CSV + LaTeX (booktabs, ``escape=True``) for all four
    under ``reports/tables/`` and returns ``{"table1":..., ..., "table4":
    ...}``, plus three NICE-TO-HAVE tables (model-comparison metrics, equity
    correlations, external directional comparison) added to the SAME dict
    under their own keys whenever their source is present.

**Figures** (``figures()``): renders every paper figure that has a real,
on-disk data source, plus 4 nice-to-have results figures, to
``reports/figures/`` -- see the SAVE SPEC below. ``figures()`` itself never
raises just because one optional source is missing; each ``_fig_*`` builder
checks its own source and prints ``"[report] skipping <fig>: ... not
found"`` instead of fabricating anything.

  NECESSARY (the figures the paper uses):
    1. ``study_area_hwm``            -- AOI + 286 HW/EOW presence points
                                         (+ NHD streams if available).
    2. ``susceptibility_maps``       -- ensemble mean + uncertainty
                                         (``flood_avg``/``flood_stddev``).
    3. ``jackknife_importance``      -- classic MaxEnt jackknife
                                         (without/only-variable bars +
                                         with-all-variables line).
    4. ``internal_validation_curves``-- ROC + PR on the honest 83-point
                                         holdout (``evaluate.oos_scores``
                                         against the TRAIN-ONLY eval
                                         raster -- never ``FINAL_RASTER``).
    5. ``gfi_comparison``            -- GFI terrain-only baseline AUC vs.
                                         MaxEnt's honest OOS AUC, plus the
                                         hit-rate-vs-area-flagged sweep.
    6. ``sensitivity_summary``       -- Delta-AUC tornado across ALL 7
                                         canonical arms, outfall-distance-
                                         matched proper bound highlighted.
    7. ``susceptibility_maps_comparison`` -- 2x2 panel: MaxEnt (reference,
       ``evaluate.EVAL_RASTER`` -- the SAME train-only eval raster used
       everywhere else in this module) + Logistic Regression + Random
       Forest + XGBoost, one shared colormap/colorbar + common vmin/vmax,
       each panel titled with its ``metrics_comparison.csv`` test AUC.
       READS ``model_compare.RASTER_DIR``'s per-model rasters -- this
       function never fits/predicts anything (report.py stays READ-ONLY);
       those rasters are produced upstream by
       ``model_compare.susceptibility_rasters()`` (the notebook's Stage 4,
       right after ``model_compare.run()``). Gracefully SKIPPED-WITH-LOG
       (never the stale legacy
       ``data/processed/model_comparison/susceptibility_maps_comparison.png``
       pre-pipeline artifact) if those rasters have not been produced yet.

  NICE-TO-HAVE (added only if their source CSV exists):
    8. ``model_comparison_auc``  -- 6-model AUC bar (+ Nadeau-Bengio note).
    9. ``spatial_cv_folds``      -- spatial-CV per-fold AUC spread.
    10. ``equity_income_quintile`` -- susceptibility by income quintile.
    11. ``calibration_auprc``    -- AUPRC + Brier score by model.

**SAVE SPEC** (``_save``, applies to every figure without exception):
  ``<name>.png`` at ``dpi=300`` AND ``<name>.pdf`` (vector), both
  ``bbox_inches="tight"``, under ``config.REPO/"reports"/"figures"``.
  ``figures()`` returns the list of PNG paths written; each has a same-stem
  ``.pdf`` sibling.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless-safe -- no DISPLAY needed under pytest/CI.
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import rioxarray
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from pipeline import config, equity, evaluate, hwm, maxent, model_compare, predictors, qa, robustness, screen, sensitivity

REPORTS_DIR = config.REPO / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"

FIGURE_DPI = 300   # publication resolution -- every _save() call uses this for the PNG;
                   # the PDF sibling is vector and carries no DPI concept at all.

CHANCE_AUC = 0.5   # ROC-AUC of a non-discriminating classifier -- a property of the metric
                   # itself, not a pipeline-observed artifact value; the one literal in this
                   # module that is deliberately NOT sourced from a CSV.

# Table 4's presentation order: baseline first, then the three background-
# control ("ascertainment-bias") arms in the pipeline's own established
# narrative order (pipeline.sensitivity's module docstring: developed ->
# stormwater-buffer LOWER bound -> outfall-distance-matched PROPER/upper
# bound), then the three remaining robustness arms (resolution,
# output-format, event-pooling). Checked
# against the CSV's actual arm set at read time (table4() asserts equality)
# so this list can never silently drop or misrepresent a real arm.
ARM_ORDER = [
    "baseline",
    "bg_target_group_developed",
    "bg_target_group_stormwater_buffer",
    "bg_outfall_distance_matched",
    "resolution_30m",
    "output_format_logistic",
    "event_pooling_top4",
]

# Short, paper-ready display labels for ARM_ORDER's raw ids -- used ONLY by
# the sensitivity_summary figure (the tornado plot's y-axis), never by
# table4() itself (whose "Arm" column stays the real, joinable arm id).
ARM_DISPLAY_LABELS = {
    "baseline": "Baseline (random background)",
    "bg_target_group_developed": "Background: NLCD developed",
    "bg_target_group_stormwater_buffer": "Background: stormwater buffer",
    "bg_outfall_distance_matched": "Background: outfall-distance matched",
    "resolution_30m": "Resolution: 30 m",
    "output_format_logistic": "Output format: logistic",
    "event_pooling_top4": "Events: top-4 only",
}

DELTA_AUC_COL = "ΔAUC"   # table4()'s own delta_auc column name (post-rename) -- held in
                         # one place so every consumer below references the identical string.


# ============================================================================
# Table 1 -- flood-observation inventory.
# ============================================================================

def table1() -> pd.DataFrame:
    """One row per observation class (High-Water Marks; End-of-Water /
    extent-of-water points) plus a Total row, each with its raw
    (pre-thinning) count and its retained-after-30-m-thinning count --
    read live off ``hwm.raw_counts()``/``hwm.build_presence()``, never
    hardcoded.
    """
    raw = hwm.raw_counts()
    presence = hwm.build_presence()
    retained = presence["type"].value_counts()

    hw_raw, eow_raw = int(raw["HW_raw"]), int(raw["EOW_raw"])
    hw_retained, eow_retained = int(retained.get("HW", 0)), int(retained.get("EOW", 0))

    df = pd.DataFrame([
        {"Class": "High-Water Marks (HW)",
         "Raw count": hw_raw, "Retained after 30 m thinning": hw_retained},
        {"Class": "End-of-Water / extent-of-water (EOW)",
         "Raw count": eow_raw, "Retained after 30 m thinning": eow_retained},
        {"Class": "Total",
         "Raw count": hw_raw + eow_raw, "Retained after 30 m thinning": hw_retained + eow_retained},
    ])

    df.attrs["source"] = "pipeline.hwm.raw_counts() + pipeline.hwm.build_presence()"
    df.attrs["raw_hw"] = hw_raw
    df.attrs["raw_eow"] = eow_raw
    df.attrs["retained_hw"] = hw_retained
    df.attrs["retained_eow"] = eow_retained
    # EOW points are USED in the canonical modeling set, never
    # discarded -- config.canonical_samples() applies no type-based filter.
    df.attrs["eow_status"] = "used"
    return df


# ============================================================================
# Table 2 -- predictor stack.
# ============================================================================

# Family/Source/Type for the 21 variables of the original draft's predictor
# table, keyed by this pipeline's internal snake_case predictor name. PLUS three
# entries that table omitted even though this canonical
# pipeline actually acquired/screened them -- "tri"/"tpi" (real terrain
# layers, dropped at multicollinearity screening -- see PLANNED_
# MULTICOLLINEARITY_DROPS in pipeline/predictors.py) and "median_income"
# (acquired + screened IN, dropped only at the modeling stage). Order here
# is display order (Family-grouped, matching the paper's own row order,
# with the 3 non-paper rows inserted into their natural family slot) --
# table2() iterates this dict in insertion order.
PREDICTOR_CATALOG = {
    # Terrain
    "dem":                      {"family": "Terrain",    "source": "USGS 3DEP 10 m",    "type": "Cont."},
    "slope":                    {"family": "Terrain",    "source": "Derived",           "type": "Cont."},
    "curvature":                {"family": "Terrain",    "source": "Derived",           "type": "Cont."},
    "tri":                      {"family": "Terrain",    "source": "Derived",           "type": "Cont."},
    "tpi":                      {"family": "Terrain",    "source": "Derived",           "type": "Cont."},
    # Hydrology
    "flow_accumulation":        {"family": "Hydrology",  "source": "Derived (D8)",      "type": "Cont."},
    "twi":                      {"family": "Hydrology",  "source": "Derived (D8)",      "type": "Cont."},
    "spi":                      {"family": "Hydrology",  "source": "Derived (D8)",      "type": "Cont."},
    "hand":                     {"family": "Hydrology",  "source": "Derived (D8)",      "type": "Cont."},
    "distance_to_streams":      {"family": "Hydrology",  "source": "NHD HR",            "type": "Cont."},
    "drainage_density":         {"family": "Hydrology",  "source": "NHD HR",            "type": "Cont."},
    # Soils
    "ksat":                     {"family": "Soils",      "source": "POLARIS",           "type": "Cont."},
    "available_water_storage":  {"family": "Soils",      "source": "gNATSGO",           "type": "Cont."},
    "depth_to_restriction":     {"family": "Soils",      "source": "gNATSGO",           "type": "Cont."},
    "hydrologic_soil_group":    {"family": "Soils",      "source": "SSURGO/SDA",        "type": "Cat."},
    "curve_number":             {"family": "Soils",      "source": "Derived (TR-55)",   "type": "Cont."},
    # Land cover
    "nlcd_landcover":           {"family": "Land cover", "source": "NLCD 2021",         "type": "Cat."},
    "impervious_pct":           {"family": "Land cover", "source": "NLCD 2021",         "type": "Cont."},
    # Climate -- never acquired in this canonical rebuild (see
    # pipeline.predictors module docstring, "deviation (a)"): no raster for
    # either layer ever existed under data/processed/predictors/.
    "precip_annual_mean":       {"family": "Climate",    "source": "gridMET 2014-2023", "type": "Cont."},
    "precip_rx1day":            {"family": "Climate",    "source": "gridMET 2014-2023", "type": "Cont."},
    # Stormwater
    "stormwater_density":       {"family": "Stormwater", "source": "City of Manhattan",  "type": "Cont."},
    "distance_to_outfall":      {"family": "Stormwater", "source": "City of Manhattan",  "type": "Cont."},
    # Census
    "population_density":       {"family": "Census",     "source": "ACS 2018-2022",      "type": "Cont."},
    "median_income":            {"family": "Census",     "source": "ACS 2018-2022",      "type": "Cont."},
}

# The live screen.multicollinearity() drop set this table asserts against
# (table2() re-checks this every call against the REAL on-disk CSV -- this
# tuple is only the EXPECTED value for that assertion, never itself the
# source of the "dropped" decision).
_SCREENED_OUT_EXPECTED = frozenset({"flow_accumulation", "spi", "tri", "tpi"})

# Never acquired in this canonical rebuild at all (module docstring above).
_NOT_ACQUIRED = ("precip_annual_mean", "precip_rx1day")


def _screening_reason(var: str, mc: pd.DataFrame) -> str:
    """A faithful, LIVE-numbered (never hardcoded) one-line rationale for
    why ``var`` was dropped at multicollinearity screening -- pulls the
    actual pairwise Pearson r straight off ``mc`` (``screen.REPORT_PATH``'s
    own correlation matrix, on the MODELING-transformed values screen.py's
    Stage 1/2 actually used) rather than restating a number that could go
    stale. Mirrors the rationale already documented in
    ``screen._KNOWN_PAIR_RATIONALE``/``_choose_vif_drop``'s docstrings.
    """
    if var == "tri":
        r = mc.loc["tri", "slope"]
        return (f"screening Stage 1 Pearson (r={r:.3f} with slope; slope retained as the "
                 f"standard hydrologic predictor)")
    if var == "tpi":
        r = mc.loc["tpi", "curvature"]
        return (f"screening Stage 1 Pearson (r={r:.3f} with curvature; curvature retained as "
                 f"the more direct flow-convergence measure)")
    if var == "spi":
        r = mc.loc["spi", "flow_accumulation"]
        return f"screening Stage 1 Pearson (r={r:.3f} with flow_accumulation, post-log-transform)"
    if var == "flow_accumulation":
        r = mc.loc["flow_accumulation", "twi"]
        return (f"screening Stage 2 VIF (shares TWI's post-log scale, pairwise r={r:.3f} under "
                 f"the Stage-1 threshold but VIF-inflated; TWI retained as the standard "
                 f"topographic wetness index)")
    return "multicollinearity screening"   # defensive fallback; never expected to fire.


def table2() -> pd.DataFrame:
    """The full CONSIDERED predictor stack (24 rows: the paper's own 21 plus
    ``tri``/``tpi``/``median_income``, which its table omits), joined with
    the LIVE bivariate effect size (``screen.BIVARIATE_REPORT_PATH``) and
    the LIVE screening retained/dropped result (``screen.REPORT_PATH``'s
    ``dropped`` column) to produce the columns: Family, Variable, Source,
    Type, ``Effect (MW r / V)``, Sig, Status.

    Status is one of:
      - ``"Retained (modeled)"`` -- one of the 17 canonical MaxEnt inputs.
      - ``"Dropped -- <live screening reason>"`` -- one of the 4
        multicollinearity-screening drops (tri/tpi/flow_accumulation/spi),
        reason text built from the REAL on-disk correlation matrix
        (``_screening_reason``), never a hardcoded number.
      - ``"Dropped -- modeling stage (...)"`` -- ``median_income`` only:
        survived screening (Pearson/VIF), dropped afterward for a
        structural ACS coverage gap (``predictors.MODELING_DROPS``).
      - ``"Dropped -- not acquired (...)"`` -- ``precip_annual_mean``/
        ``precip_rx1day``: the paper's table lists them, but this canonical
        rebuild never acquired that gridMET layer at all.

    Effect/Sig are ``"N/A"``/``"n/a"`` for the 6 rows with no bivariate
    characterization on disk (the 4 screening drops -- bivariate() only
    ever ran on the SCREENED stack -- plus the 2 never-acquired precip
    rows); every other row (the 17 modeled + median_income, all 18 of
    which DO appear in ``predictor_bivariate_characterization.csv``) gets
    its real Mann-Whitney rank-biserial r (continuous) or Cramer's V
    (categorical, prefixed ``"V="``) plus a ``"**"`` marker wherever
    ``p_bonferroni < 0.05`` (matching the paper's own significance
    convention), else ``""``.

    Raises ``AssertionError`` if the REAL on-disk screening result no
    longer drops exactly ``{tri, tpi, flow_accumulation, spi}`` -- a loud
    signal that this table (and its docstring) need updating, never a
    silent drift.
    """
    biv = pd.read_csv(screen.BIVARIATE_REPORT_PATH).set_index("predictor")
    mc = pd.read_csv(screen.REPORT_PATH, index_col=0)

    screened_dropped = set(mc.index[mc["dropped"].astype(bool)])
    assert screened_dropped == set(_SCREENED_OUT_EXPECTED), (
        f"[report] table2: live screen at {screen.REPORT_PATH} dropped {sorted(screened_dropped)}, "
        f"expected exactly {sorted(_SCREENED_OUT_EXPECTED)} -- the canonical screening result has "
        f"changed; update pipeline.report.table2()/its docstring to match"
    )

    rows = []
    for var, meta in PREDICTOR_CATALOG.items():
        if var in _NOT_ACQUIRED:
            status = ("Dropped -- not acquired (gridMET precipitation layer out of scope for "
                       "this canonical rebuild; see pipeline.predictors module docstring)")
            effect, sig = "N/A", "n/a"
        elif var in screened_dropped:
            status = f"Dropped -- {_screening_reason(var, mc)}"
            effect, sig = "N/A", "n/a"
        else:
            row = biv.loc[var]
            effect = (f"V={row['effect']:.3f}" if meta["type"] == "Cat."
                      else f"{row['effect']:+.3f}")
            sig = "**" if bool(row["significant"]) else ""
            if var in predictors.MODELING_DROPS:
                status = ("Dropped -- modeling stage (ACS median-income estimate missing for "
                           "some AOI block groups; a structural coverage gap, not a "
                           "multicollinearity drop)")
            else:
                status = "Retained (modeled)"
        rows.append({
            "Family": meta["family"], "Variable": var, "Source": meta["source"],
            "Type": meta["type"], "Effect (MW r / V)": effect, "Sig": sig, "Status": status,
        })

    df = pd.DataFrame(rows)
    n_modeled = int((df["Status"] == "Retained (modeled)").sum())
    assert n_modeled == 17, f"[report] table2: expected 17 modeled predictors, got {n_modeled}"

    df.attrs["source"] = (
        f"PREDICTOR_CATALOG (the original draft's 21-variable predictor table) "
        f"joined with {screen.BIVARIATE_REPORT_PATH} (live effect sizes) and "
        f"{screen.REPORT_PATH} (live screening retained/dropped)"
    )
    df.attrs["n_modeled"] = n_modeled
    df.attrs["n_considered"] = len(df)
    df.attrs["note"] = (
        f"The original draft's predictor table listed 21 variables and omitted "
        f"median_income. The canonical, empirically screened stack MaxEnt "
        f"actually models is {n_modeled} predictors (Status == 'Retained (modeled)'): "
        f"tri/tpi/flow_accumulation/spi are dropped at multicollinearity screening, "
        f"median_income is dropped afterward at the modeling stage (ACS coverage gap), and "
        f"precip_annual_mean/precip_rx1day were never acquired in this pipeline. "
        f"This table's {len(df)} rows are the FULL considered stack (the draft's 21 plus "
        f"tri/tpi/median_income)."
    )
    return df


# ============================================================================
# Table 3 -- variable importance.
# ============================================================================

def table3(top_n: int | None = None) -> pd.DataFrame:
    """Predictor x {gain_only, unique_contribution, permutation_importance_
    mean(+sd), percent_contribution_mean(+sd), mw_rank_biserial}, read from
    ``maxent.VARIABLE_IMPORTANCE_CSV`` and sorted so the top predictor
    (by jackknife unique contribution -- this pipeline's own established
    primary importance metric) is row 0. ``distance_to_outfall`` ranks #1.

    ``top_n``: if given, returns only the top ``top_n`` rows (the
    paper's own Table 3 caption says "top 10 predictors" -- pass
    ``top_n=10`` to reproduce that exact subset). Default ``None`` keeps
    the FULL 17-row table.
    """
    df = pd.read_csv(maxent.VARIABLE_IMPORTANCE_CSV)
    df = df.sort_values("unique_contribution", ascending=False).reset_index(drop=True)
    source = str(maxent.VARIABLE_IMPORTANCE_CSV)
    top_predictor = df.iloc[0]["predictor"]

    if top_n is not None:
        df = df.head(top_n).reset_index(drop=True)

    df.attrs["source"] = source
    df.attrs["top_predictor"] = top_predictor
    df.attrs["top_n"] = top_n
    return df


# ============================================================================
# Table 4 -- ascertainment / sensitivity arms.
# ============================================================================

def table4() -> pd.DataFrame:
    """The canonical sensitivity arms (``sensitivity.SUMMARY_CSV`` --
    ``data/processed/maxent/sensitivity/sensitivity_summary.csv``, NEVER
    the stale pre-pipeline ``data/processed/sensitivity/`` legacy file),
    reordered via ``ARM_ORDER`` and relabeled to the paper's column names:
    ``arm``->Arm, ``description``->Description, ``auc``->AUC,
    ``baseline_auc``->"Matched-baseline AUC", ``delta_auc``->"ΔAUC".
    ``eval_id``/``baseline_eval_id`` (provenance) pass through unrenamed.
    """
    df = pd.read_csv(sensitivity.SUMMARY_CSV)

    on_disk = set(df["arm"])
    expected = set(ARM_ORDER)
    assert on_disk == expected, (
        f"[report] table4: {sensitivity.SUMMARY_CSV} arms {sorted(on_disk)} do not match "
        f"the expected canonical arm set {sorted(expected)} -- update pipeline.report.ARM_ORDER"
    )

    df = df.set_index("arm").loc[ARM_ORDER].reset_index()
    df = df.rename(columns={
        "arm": "Arm",
        "description": "Description",
        "auc": "AUC",
        "baseline_auc": "Matched-baseline AUC",
        "delta_auc": DELTA_AUC_COL,
    })
    df.attrs["source"] = str(sensitivity.SUMMARY_CSV)
    return df


# ============================================================================
# tables() -- CSV + LaTeX save spec, shared by every table in this module.
# ============================================================================

def _save_table(df: pd.DataFrame, name: str) -> tuple[Path, Path]:
    """Writes ``<name>.csv`` and ``<name>.tex`` (booktabs -- pandas 3's
    ``to_latex`` emits ``\\toprule``/``\\midrule``/``\\bottomrule`` by
    default -- LaTeX-escaped, ``NaN`` rendered as ``"--"``) under
    ``TABLES_DIR``. The ONE save path every table in this module goes
    through, mirroring ``_save``'s role for figures below.
    """
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = TABLES_DIR / f"{name}.csv"
    tex_path = TABLES_DIR / f"{name}.tex"
    df.to_csv(csv_path, index=False)
    tex_path.write_text(df.to_latex(index=False, escape=True, na_rep="--"), encoding="utf-8")
    return csv_path, tex_path


def _external_directional_comparison_table() -> pd.DataFrame | None:
    """A compact "at a glance" external directional-consistency table:
    FEMA (2 zone classes) + NWS (5 categories) + Sentinel-1, each filtered
    down to its already-computed per-threshold sweep's row NEAREST the
    FINAL model's own current MaxSS threshold (``evaluate.
    final_maxss_threshold()`` -- the SAME threshold ``evaluate.
    external_comparisons()`` itself binarizes at), instead of the full
    sweep. Nearest- rather than exact-match is deliberate: the FINAL model
    is a 10-replicate bootstrap ensemble fit with no fixed CLI seed (see
    ``maxent.fit_final``'s docstring), so its own MaxSS drifts slightly
    run-to-run -- an exact-match would spuriously skip this table every
    time that jitter nudges the live threshold off the sweep CSVs'
    (already-cached) threshold grid. NEVER independent validation
    -- carries ``evaluate.DIRECTIONAL_FRAMING``/``URBAN_PLUVIAL_SCOPE_
    CAVEAT`` in ``df.attrs``, matching every other directional-consistency
    result in this codebase. Returns ``None`` (with a skip print) if none
    of the three source CSVs exist.
    """
    fema_p = evaluate.VALIDATION_DIR / "fema_spatial_comparison.csv"
    nws_p = evaluate.VALIDATION_DIR / "nws_spatial_comparison.csv"
    s1_p = evaluate.VALIDATION_DIR / "s1_spatial_comparison.csv"
    if not (fema_p.exists() or nws_p.exists() or s1_p.exists()):
        print(f"[report] skipping external_directional_comparison table: none of "
              f"{fema_p.name}/{nws_p.name}/{s1_p.name} found under {evaluate.VALIDATION_DIR}")
        return None

    threshold = evaluate.final_maxss_threshold()

    def _headline_rows(csv_path, label_prefix, label_col=None):
        if not csv_path.exists():
            print(f"[report] external_directional_comparison: {csv_path} not found -- "
                  f"skipping {label_prefix}")
            return []
        sweep = pd.read_csv(csv_path)
        out_rows = []
        groups = sweep.groupby(label_col) if label_col else [(None, sweep)]
        for key, g in groups:
            nearest_idx = (g["threshold"] - threshold).abs().idxmin()
            r = g.loc[nearest_idx]
            product = f"{label_prefix} {key}" if label_col else label_prefix
            out_rows.append({"product": product, "threshold": float(r["threshold"]),
                              "jaccard": float(r["jaccard"]), "hit_rate": float(r["hit_rate"]),
                              "precision": float(r["precision"]), "f1": float(r["f1"])})
        return out_rows

    rows = []
    rows += _headline_rows(fema_p, "FEMA", "fema_zone")
    rows += _headline_rows(nws_p, "NWS", "nws_category")
    rows += _headline_rows(s1_p, "Sentinel-1 SAR")

    if not rows:
        print("[report] external_directional_comparison: no rows assembled -- skipping table")
        return None

    out = pd.DataFrame(rows)
    out.attrs["kind"] = evaluate.DIRECTIONAL_KIND
    out.attrs["framing"] = evaluate.DIRECTIONAL_FRAMING
    out.attrs["scope_caveat"] = evaluate.URBAN_PLUVIAL_SCOPE_CAVEAT
    out.attrs["threshold"] = threshold
    return out


def tables() -> dict:
    """Writes table1..table4 (CSV + LaTeX under ``reports/tables/``) and
    returns ``{"table1": ..., "table2": ..., "table3": ..., "table4":
    ...}``. Also emits three NICE-TO-HAVE tables under the SAME CSV+LaTeX
    save spec -- model-comparison metrics, equity correlations, and the
    external directional-consistency headline -- each ADDED to the
    returned dict under its own key ONLY if its source is present (never a
    KeyError if a nice-to-have's source is missing; a clear skip print
    instead).

    The equity-correlations nice-to-have reuses ``equity.correlations()``'s
    real Spearman/Holm logic against the ALREADY-CACHED
    ``data/processed/equity/equity_summary_all_bgs.csv`` (passed in via
    ``correlations(table=...)``) rather than calling ``equity.run()``
    itself, which would trigger a fresh ~2-5 minute live ACS network pull
    -- the exact same "never re-run a heavy stage" discipline this whole
    module is built on.
    """
    out = {}

    t1 = table1(); _save_table(t1, "table1"); out["table1"] = t1
    t2 = table2(); _save_table(t2, "table2"); out["table2"] = t2
    t3 = table3(); _save_table(t3, "table3"); out["table3"] = t3
    t4 = table4(); _save_table(t4, "table4"); out["table4"] = t4

    mc_path = model_compare.OUT_DIR / "metrics_comparison.csv"
    if mc_path.exists():
        tmc = pd.read_csv(mc_path)
        _save_table(tmc, "model_comparison_metrics")
        out["model_comparison_metrics"] = tmc
    else:
        print(f"[report] skipping model_comparison_metrics table: {mc_path} not found")

    eq_path = config.DATA / "equity" / "equity_summary_all_bgs.csv"
    if eq_path.exists():
        eq_table = pd.read_csv(eq_path)
        corr = equity.correlations(table=eq_table)
        _save_table(corr, "equity_correlations")
        out["equity_correlations"] = corr
    else:
        print(f"[report] skipping equity_correlations table: {eq_path} not found")

    ext = _external_directional_comparison_table()
    if ext is not None:
        _save_table(ext, "external_directional_comparison")
        out["external_directional_comparison"] = ext

    return out


# ============================================================================
# figures() -- non-interactive PNG+PDF pairs. Save spec lives in ONE place.
# ============================================================================

def _save(fig, name: str) -> Path:
    """Writes ``<name>.png`` (``dpi=FIGURE_DPI`` == 300) AND ``<name>.pdf``
    (vector) under ``FIGURES_DIR``, both ``bbox_inches="tight"`` -- the ONE
    save path every figure builder in this module goes through, so the
    publication-resolution save spec can never be applied inconsistently.
    Always closes ``fig`` (Agg-backend-safe). Returns the PNG ``Path`` (the
    PDF path is implied: same stem, ``.pdf`` suffix, same directory).
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    png_path = FIGURES_DIR / f"{name}.png"
    pdf_path = FIGURES_DIR / f"{name}.pdf"
    fig.savefig(png_path, dpi=FIGURE_DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path


def _clean_axes(*axes):
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


# ----------------------------------------------------------------------
# 1. study_area_hwm
# ----------------------------------------------------------------------

NAMED_RIVERS = ("Kansas River", "Big Blue River", "Wildcat Creek")


def _kansas_outline():
    """State outline for the locator inset, dissolved from the TIGER block groups
    already downloaded for the equity analysis. Returns None if absent."""
    z = config.REPO / "data" / "raw" / "tiger" / "tl_2022_20_bg.zip"
    if not z.exists():
        return None
    return gpd.read_file(f"zip://{z}").to_crs(26914).dissolve()


def _hillshade(dem, azimuth=315.0, altitude=45.0):
    """Standard Horn hillshade, 0-1. Written out rather than pulled from a
    dependency so the figure does not acquire one for a single call."""
    z = dem.astype("float64")
    dy, dx = np.gradient(z)
    slope = np.pi / 2.0 - np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)
    az, alt = np.radians(360.0 - azimuth + 90.0), np.radians(altitude)
    shaded = (np.sin(alt) * np.sin(slope)
              + np.cos(alt) * np.cos(slope) * np.cos(az - aspect))
    return (shaded + 1) / 2


def _scale_bar(ax, length_m=5000, label="5 km"):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    x = x0 + 0.06 * (x1 - x0)
    y = y0 + 0.06 * (y1 - y0)
    ax.plot([x, x + length_m], [y, y], color="#212121", linewidth=2.4,
            solid_capstyle="butt", zorder=6)
    ax.text(x + length_m / 2, y + 0.018 * (y1 - y0), label, ha="center", va="bottom",
            fontsize=7.5, color="#212121", zorder=6)


def _fig_study_area_hwm():
    """Study-area locator.

    Rebuilt 2026-08-18. The previous version drew every NHD flowline in the
    domain, which buried the three watercourses the paper actually discusses in
    a mesh of unnamed tributaries, and it carried no hillshade and no inset even
    though the caption promised both. Only the Kansas River, the Big Blue River
    and Wildcat Creek are drawn now, each labelled, over a hillshade of the same
    10 m DEM the model uses.
    """
    aoi_path = config.DATA / "aoi" / "manhattan_city_limits.gpkg"
    presence_path = config.DATA / "canonical_presence.csv"
    if not aoi_path.exists() or not presence_path.exists():
        print(f"[report] skipping study_area_hwm: {aoi_path} or {presence_path} not found")
        return None

    city = gpd.read_file(aoi_path)
    # The marks drawn are the ones the model actually fits, not every thinned
    # record. canonical_presence.csv holds 286 rows; nine of those share a 10 m
    # pixel with another mark and are dropped before fitting, so plotting the CSV
    # put a legend count in the figure that no number in the paper matched.
    samples = config.canonical_samples()
    pres_gdf = gpd.GeoDataFrame(
        samples, geometry=gpd.points_from_xy(samples["x"], samples["y"]), crs=city.crs,
    )

    fig, ax = plt.subplots(figsize=(6.4, 5.4))

    # --- hillshade the caption has always claimed ------------------------
    dem_path = config.DATA / "predictors" / "dem.tif"
    if dem_path.exists():
        dem = rioxarray.open_rasterio(dem_path, masked=True).squeeze()
        hs = _hillshade(np.nan_to_num(dem.values, nan=float(np.nanmedian(dem.values))))
        west, south, east, north = dem.rio.bounds()
        ax.imshow(hs, cmap="Greys_r", vmin=0.25, vmax=1.0, alpha=0.55,
                  extent=(west, east, south, north), origin="upper", zorder=0)

    city.plot(ax=ax, facecolor="#FFFFFF", edgecolor="#37474F", linewidth=1.3,
              alpha=0.28, zorder=2)
    city.boundary.plot(ax=ax, edgecolor="#37474F", linewidth=1.3, zorder=3)

    # --- only the named watercourses -------------------------------------
    nhd_path = config.DATA / "nhd" / "flowlines_hr.gpkg"
    if nhd_path.exists():
        flow = gpd.read_file(nhd_path)
        named = flow[flow["gnis_name"].isin(NAMED_RIVERS)]
        for name, width in zip(NAMED_RIVERS, (2.6, 2.2, 1.6)):
            seg = named[named["gnis_name"] == name]
            if seg.empty:
                continue
            seg.plot(ax=ax, color="#2E7FA8", linewidth=width, zorder=4)
            longest = seg.geometry.length.idxmax()
            pt = seg.loc[longest].geometry.interpolate(0.5, normalized=True)
            ax.annotate(name, xy=(pt.x, pt.y), fontsize=8, style="italic",
                        color="#1B5E77", zorder=6, ha="center",
                        path_effects=[pe.withStroke(linewidth=2.6, foreground="white")])

    # --- the marks --------------------------------------------------------
    for t, color, marker, lab in (("HW", "#C62828", "o", "High water mark"),
                                  ("EOW", "#EF8A00", "^", "Edge of water")):
        sub = pres_gdf[pres_gdf["type"] == t]
        ax.scatter(sub.geometry.x, sub.geometry.y, s=26, c=color, marker=marker,
                   edgecolor="white", linewidth=0.6, zorder=5,
                   label=f"{lab} ($n={len(sub)}$)")

    # Frame on the city and the marks rather than the whole routed DEM, which is
    # far wider than the study area and left the scale bar stranded in a margin.
    xmin, ymin, xmax, ymax = gpd.GeoSeries(
        list(city.geometry) + list(pres_gdf.geometry), crs=city.crs).total_bounds
    padx, pady = 0.07 * (xmax - xmin), 0.07 * (ymax - ymin)
    ax.set_xlim(xmin - padx, xmax + padx)
    ax.set_ylim(ymin - pady, ymax + pady)
    ax.set_aspect("equal")

    ax.set_xticks([]); ax.set_yticks([])
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_color("#B0BEC5")
    leg = ax.legend(fontsize=8, loc="lower right", framealpha=0.94, borderpad=0.7)
    leg.get_frame().set_edgecolor("#B0BEC5")

    # north arrow and scale
    ax.annotate("N", xy=(0.955, 0.93), xytext=(0.955, 0.86), xycoords="axes fraction",
                ha="center", va="center", fontsize=10, fontweight="bold", color="#212121",
                arrowprops=dict(arrowstyle="-|>", color="#212121", linewidth=1.4))
    _scale_bar(ax)

    # --- locator inset ----------------------------------------------------
    ks = _kansas_outline()
    if ks is not None:
        iax = ax.inset_axes([0.015, 0.70, 0.30, 0.28])
        ks.boundary.plot(ax=iax, color="#546E7A", linewidth=0.8)
        ks.plot(ax=iax, facecolor="#ECEFF1", zorder=0)
        cx, cy = float(city.geometry.centroid.x.iloc[0]), float(city.geometry.centroid.y.iloc[0])
        iax.plot(cx, cy, marker="*", markersize=9, color="#C62828",
                 markeredgecolor="white", markeredgewidth=0.5, zorder=3)
        iax.set_xticks([]); iax.set_yticks([])
        iax.set_facecolor("white")
        for s in iax.spines.values():
            s.set_color("#B0BEC5")
        iax.set_title("Kansas", fontsize=7, pad=2, color="#37474F")

    plt.tight_layout()
    return _save(fig, "study_area_hwm")


# ----------------------------------------------------------------------
# 2. susceptibility_maps
# ----------------------------------------------------------------------

def _fig_susceptibility_maps():
    mean_path = evaluate.FINAL_RASTER
    if not mean_path.exists():
        print(f"[report] skipping susceptibility_maps: {mean_path} not found")
        return None

    sd_path = mean_path.parent / "flood_stddev.tif"
    have_sd = sd_path.exists()
    if not have_sd:
        print(f"[report] susceptibility_maps: {sd_path} not found -- rendering mean-only (1 panel)")

    mean_da = rioxarray.open_rasterio(mean_path).squeeze("band", drop=True)
    left, bottom, right, top = mean_da.rio.bounds()
    extent = [left, right, bottom, top]

    presence_path = config.DATA / "canonical_presence.csv"
    presence = pd.read_csv(presence_path) if presence_path.exists() else None

    ncols = 2 if have_sd else 1
    fig, axes = plt.subplots(1, ncols, figsize=(9.6 if have_sd else 5.2, 4.6))
    axes = np.atleast_1d(axes)

    im0 = axes[0].imshow(mean_da.values, extent=extent, cmap="YlOrRd", origin="upper")
    axes[0].set_title("(a) Ensemble mean susceptibility" if have_sd else "Ensemble mean susceptibility",
                       fontsize=9)
    fig.colorbar(im0, ax=axes[0], shrink=0.75, label="Cloglog susceptibility")

    if have_sd:
        sd_da = rioxarray.open_rasterio(sd_path).squeeze("band", drop=True)
        im1 = axes[1].imshow(sd_da.values, extent=extent, cmap="viridis", origin="upper")
        axes[1].set_title("(b) Ensemble uncertainty (std. dev.)", fontsize=9)
        fig.colorbar(im1, ax=axes[1], shrink=0.75, label="Std. dev. (10 bootstrap replicates)")

    if presence is not None:
        for ax in axes:
            ax.scatter(presence["x"], presence["y"], s=3, c="black", alpha=0.45, linewidth=0)

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    return _save(fig, "susceptibility_maps")


# ----------------------------------------------------------------------
# 3. jackknife_importance
# ----------------------------------------------------------------------

def _fig_jackknife_importance(t3: pd.DataFrame):
    jk_path = maxent.JACKKNIFE_GAINS_CSV
    if not jk_path.exists():
        print(f"[report] skipping jackknife_importance: {jk_path} not found")
        return None

    jk = pd.read_csv(jk_path).set_index("predictor")
    order = [p for p in t3["predictor"] if p in jk.index]
    jk = jk.loc[order]
    gain_all = float(jk["gain_all"].iloc[0])   # identical across rows -- the full-model reference.

    ordered = jk.iloc[::-1]   # row 0 (top predictor) plotted LAST -> appears at the chart's TOP.
    y = np.arange(len(ordered))

    fig, ax = plt.subplots(figsize=(4.6, max(4.2, 0.34 * len(ordered) + 1)))
    ax.barh(y - 0.19, ordered["gain_without"], height=0.38, color="#90A4AE", label="Without variable")
    ax.barh(y + 0.19, ordered["gain_only"], height=0.38, color="#1565C0", label="With only variable")
    ax.axvline(gain_all, color="#E53935", linestyle="--", linewidth=1.3,
               label=f"With all variables ({gain_all:.2f})")
    ax.set_yticks(y); ax.set_yticklabels(ordered.index, fontsize=7.5)
    ax.set_xlabel("Regularized training gain", fontsize=9)
    ax.set_title("Jackknife of regularized training gain", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="lower right")
    _clean_axes(ax)
    plt.tight_layout()
    return _save(fig, "jackknife_importance")


# ----------------------------------------------------------------------
# 4. internal_validation_curves
# ----------------------------------------------------------------------

def _fig_internal_validation_curves():
    if not evaluate.EVAL_RASTER.exists():
        print(f"[report] skipping internal_validation_curves: {evaluate.EVAL_RASTER} not found")
        return None

    samples = config.canonical_samples()
    train_idx, test_idx = config.split(samples)
    qa.assert_eval_model_excludes_test(train_idx, test_idx)   # Invariant 2, before any scoring.

    test_pts = samples.iloc[test_idx][["x", "y"]]
    # Score on the SAME matched test set the six-model table uses -- the 83 test
    # presence points against the 30% test partition of the presence-excluded
    # background pool -- not a fresh full-size background draw. Both are valid,
    # but AUPRC and its no-skill baseline both move with the presence:background
    # ratio, so drawing a different background here would print an AUPRC that
    # disagrees with the one in the model-comparison table for no reason a
    # reader could see.
    background_pool = model_compare.build_background(
        n=model_compare.DEFAULT_BACKGROUND_N, seed=config.GLOBAL_SEED, presence=samples)
    _bg_train_idx, bg_test_idx = config.split(background_pool)
    background = background_pool.iloc[bg_test_idx][["x", "y"]]
    y_true, y_score = evaluate.oos_scores(evaluate.EVAL_RASTER, test_pts, background)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)
    prevalence = float(np.mean(y_true))

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
    ax = axes[0]
    ax.plot(fpr, tpr, color="#1565C0", linewidth=1.8, label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("False positive rate", fontsize=8); ax.set_ylabel("True positive rate", fontsize=8)
    ax.set_title("(a) ROC curve", fontsize=9)
    ax.legend(fontsize=7.5, loc="lower right")

    ax = axes[1]
    ax.plot(rec, prec, color="#2E7D32", linewidth=1.8, label=f"PR (AUPRC = {auprc:.3f})")
    ax.axhline(prevalence, color="k", linestyle="--", linewidth=0.8, alpha=0.6,
               label=f"No-skill baseline = prevalence ({prevalence:.3f})")
    ax.set_xlabel("Recall", fontsize=8); ax.set_ylabel("Precision", fontsize=8)
    ax.set_title("(b) Precision–recall curve", fontsize=9)
    ax.legend(fontsize=7.5, loc="upper right")

    for ax_ in axes:
        ax_.set_xlim(-0.02, 1.02); ax_.set_ylim(-0.02, 1.02)
        ax_.tick_params(labelsize=7.5)
    _clean_axes(*axes)
    plt.tight_layout()
    return _save(fig, "internal_validation_curves")


# ----------------------------------------------------------------------
# 5. gfi_comparison
# ----------------------------------------------------------------------

def _fig_gfi_comparison():
    if not (evaluate.FLOW_ACCUMULATION_TIF.exists() and evaluate.SLOPE_TIF.exists()):
        print(f"[report] skipping gfi_comparison: {evaluate.FLOW_ACCUMULATION_TIF} or "
              f"{evaluate.SLOPE_TIF} not found")
        return None

    gfi_result = evaluate.gfi_baseline()
    gfi_auc = gfi_result["auc"]
    maxent_auc = float(table4().set_index("Arm").loc["baseline", "AUC"])

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
    ax = axes[0]
    bars = ax.bar(["GFI\n(terrain-only)", "MaxEnt\n(honest OOS)"], [gfi_auc, maxent_auc],
                  color=["#90A4AE", "#1565C0"], width=0.6)
    ax.axhline(CHANCE_AUC, color="black", linestyle="--", linewidth=1,
               label=f"Chance (AUC = {CHANCE_AUC})")
    # Value labels sit INSIDE each bar (white, near the top, va="top") rather
    # than floating just above it -- GFI's own bar (~0.47) sits right next to
    # the AUC=0.5 chance line, so an "above the bar" label would visually
    # collide with that dashed line; anchoring inside the bar's own colored
    # region never can, regardless of how close a bar's height lands to 0.5.
    for b, v in zip(bars, (gfi_auc, maxent_auc)):
        ax.text(b.get_x() + b.get_width() / 2, v - 0.025, f"{v:.3f}",
                ha="center", va="top", fontsize=8.5, fontweight="bold", color="white")
    ax.set_ylim(0, 1.08); ax.set_ylabel("AUC", fontsize=8)
    ax.set_title("(a) Discrimination gap", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
    ax.tick_params(labelsize=7.5)

    ax = axes[1]
    csv_path = evaluate.VALIDATION_DIR / "gfi_comparison.csv"
    if csv_path.exists():
        sweep = pd.read_csv(csv_path)
        for model, color in (("GFI", "#90A4AE"), ("MaxEnt", "#1565C0")):
            sub = sweep[sweep["model"] == model].sort_values("pred_area_pct")
            ax.plot(sub["pred_area_pct"], sub["hit_rate"], color=color, linewidth=1.6,
                    marker="o", markersize=2.5, label=model)
        ax.set_xlabel("Area flagged (% of AOI)", fontsize=8)
        ax.set_ylabel("Hit rate (presence captured)", fontsize=8)
        ax.set_title("(b) Hit rate vs. area flagged", fontsize=9)
        ax.legend(fontsize=7.5)
        ax.tick_params(labelsize=7.5)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, f"{csv_path.name}\nnot found", ha="center", va="center", fontsize=8)

    _clean_axes(axes[0], axes[1])
    plt.tight_layout()
    return _save(fig, "gfi_comparison")


# ----------------------------------------------------------------------
# 6. sensitivity_summary
# ----------------------------------------------------------------------

def _fig_sensitivity_summary(t4: pd.DataFrame):
    highlight = "bg_outfall_distance_matched"
    sub = t4.set_index("Arm").loc[ARM_ORDER]   # ARM_ORDER's own presentation order.
    ordered = sub.iloc[::-1]   # baseline plotted LAST -> appears at the chart's TOP.
    y = np.arange(len(ordered))

    colors = ["#9E9E9E" if a == "baseline" else "#E53935" if a == highlight else "#1565C0"
              for a in ordered.index]
    labels = [ARM_DISPLAY_LABELS.get(a, a) for a in ordered.index]

    fig, ax = plt.subplots(figsize=(5.6, 0.5 * len(ordered) + 1.4))
    ax.barh(y, ordered[DELTA_AUC_COL], color=colors)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7.5)
    ax.margins(x=0.15)

    # Value labels anchored to a FIXED axes-fraction column just past the right
    # spine (x=1.03 in axes coords, y in DATA coords via get_yaxis_transform())
    # -- deliberately NOT placed at each bar's own data-coordinate tip: the
    # outfall-distance-matched arm's bar (delta ~= -0.053) dwarfs every other
    # arm's (+/-0.005 or smaller), so a tip-anchored negative label would land
    # on top of the y-axis tick labels themselves. A fixed right-hand column
    # never collides with either the bars or the left-side tick labels,
    # regardless of how lopsided the arms' deltas are.
    trans = ax.get_yaxis_transform()
    for yi, (arm, row) in zip(y, ordered.iterrows()):
        d, auc = row[DELTA_AUC_COL], row["AUC"]
        ax.text(1.03, yi, f"{d:+.3f} (AUC={auc:.3f})", transform=trans,
                va="center", ha="left", fontsize=6.5,
                fontweight="bold" if arm == highlight else "normal")

    ax.set_xlabel("ΔAUC relative to matched baseline", fontsize=8.5)
    ax.tick_params(labelsize=7.5)
    _clean_axes(ax)
    plt.tight_layout()
    return _save(fig, "sensitivity_summary")


# ----------------------------------------------------------------------
# 7. susceptibility_maps_comparison -- 4-model susceptibility MAP panel.
# READ-ONLY: renders whatever model_compare.susceptibility_rasters() (run
# upstream, in the notebook/pipeline driver, NEVER here) already wrote to
# model_compare.RASTER_DIR, plus evaluate.EVAL_RASTER directly for MaxEnt's
# own panel (``maxent.fit_eval()``'s train-only evaluation raster -- the SAME raster
# model_compare.run()'s MaxEnt row and _fig_internal_validation_curves
# already use, so all 4 panels are apples-to-apples train-only-model maps).
# ----------------------------------------------------------------------

# Panel order: label -> model_compare.RASTER_DIR filename stem (None for
# MaxEnt, whose raster is evaluate.EVAL_RASTER directly, not one of
# model_compare's own per-model outputs). One method family per panel --
# MaxEnt (presence-background ME) / linear / bagging / boosting.
# All SIX compared models, grouped by family: the presence-background reference
# first, then the two smooth-boundary models, then the four tree ensembles. The
# panel list previously stopped at four, so the figure showed a subset of the
# models the paper compares -- an arbitrary layout choice a reader could not
# distinguish from a deliberate exclusion.
_COMPARISON_PANELS = (
    ("MaxEnt (reference)", None),
    ("Logistic Regression", "logistic_regression"),
    ("Random Forest", "random_forest"),
    ("XGBoost", "xgboost"),
    ("LightGBM", "lightgbm"),
    ("CatBoost", "catboost"),
)


def _fig_susceptibility_maps_comparison():
    raster_paths = {
        label: (evaluate.EVAL_RASTER if stem is None else model_compare.RASTER_DIR / f"{stem}.tif")
        for label, stem in _COMPARISON_PANELS
    }
    missing = [str(p) for p in raster_paths.values() if not p.exists()]
    if missing:
        print(f"[report] skipping susceptibility_maps_comparison: missing raster(s) {missing} -- "
              f"run model_compare.susceptibility_rasters(train_idx, test_idx, background) first "
              f"(model_compare.run() alone only writes per-model METRICS, never these prediction "
              f"rasters)")
        return None

    metrics_path = model_compare.OUT_DIR / "metrics_comparison.csv"
    auc_by_model = {}
    if metrics_path.exists():
        auc_by_model = pd.read_csv(metrics_path).set_index("model")["auc"].to_dict()
    else:
        print(f"[report] susceptibility_maps_comparison: {metrics_path} not found -- panel "
              f"titles will omit each model's test AUC")

    das = {label: rioxarray.open_rasterio(p).squeeze("band", drop=True)
           for label, p in raster_paths.items()}
    all_vals = np.concatenate([da.values[np.isfinite(da.values)] for da in das.values()])
    vmin, vmax = float(all_vals.min()), float(all_vals.max())

    # figsize's aspect ratio is matched to the raster's own (width > height)
    # aspect -- `imshow`'s default `aspect="equal"` otherwise letterboxes
    # every panel inside a too-tall grid cell, opening up a large dead gap
    # between the two rows (constrained_layout alone does not fix this;
    # the CELL aspect ratio has to be right to begin with).
    ref_da = next(iter(das.values()))
    h, w = ref_da.shape
    ncols = 3
    nrows = int(np.ceil(len(_COMPARISON_PANELS) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(12.6, (12.6 / ncols) * (h / w) * nrows + 1.1),
                             constrained_layout=True)
    im = None
    for ax, (label, _stem) in zip(axes.ravel(), _COMPARISON_PANELS):
        da = das[label]
        left, bottom, right, top = da.rio.bounds()
        im = ax.imshow(da.values, extent=[left, right, bottom, top], cmap="YlOrRd",
                        vmin=vmin, vmax=vmax, origin="upper")
        auc = auc_by_model.get(label)
        title = f"{label}\n(test AUC = {auc:.3f})" if auc is not None else label
        ax.set_title(title, fontsize=9.5)
        ax.set_xticks([]); ax.set_yticks([])

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.78, label="Susceptibility")
    return _save(fig, "susceptibility_maps_comparison")


# ----------------------------------------------------------------------
# Nice-to-have 8: model_comparison_auc
# ----------------------------------------------------------------------

def _fig_model_comparison_auc():
    csv_path = model_compare.OUT_DIR / "metrics_comparison.csv"
    if not csv_path.exists():
        print(f"[report] skipping model_comparison_auc: {csv_path} not found")
        return None

    df = pd.read_csv(csv_path).sort_values("auc")
    note = ""
    sig_path = robustness.OUT_DIR / "loeo_significance_test.csv"
    if sig_path.exists():
        sig = pd.read_csv(sig_path).iloc[0]
        note = (f"MaxEnt vs {sig['second_model']}: Nadeau-Bengio p={sig['p_value']:.2f} "
                f"({str(sig['verdict']).replace('_', ' ')})")

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    colors = ["#E53935" if "MaxEnt" in m else "#1565C0" for m in df["model"]]
    ax.barh(df["model"], df["auc"], color=colors)
    for y, v in enumerate(df["auc"]):
        ax.text(v + 0.0005, y, f"{v:.3f}", va="center", fontsize=7.5)
    ax.set_xlim(min(0.95, float(df["auc"].min()) - 0.01), 1.0)
    ax.set_xlabel("AUC (identical matched test set)", fontsize=8.5)
    title = "Model comparison -- MaxEnt vs. 5 ML comparators"
    ax.set_title(f"{title}\n{note}" if note else title, fontsize=9, fontweight="bold")
    ax.tick_params(labelsize=7.5)
    _clean_axes(ax)
    plt.tight_layout()
    return _save(fig, "model_comparison_auc")


# ----------------------------------------------------------------------
# Nice-to-have 9: spatial_cv_folds
# ----------------------------------------------------------------------

def _fig_spatial_cv_folds():
    csv_path = robustness.OUT_DIR / "spatial_cv_per_fold.csv"
    if not csv_path.exists():
        print(f"[report] skipping spatial_cv_folds: {csv_path} not found")
        return None

    df = pd.read_csv(csv_path).sort_values("auc").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(5.6, max(3.6, 0.28 * len(df) + 1)))
    sizes = np.clip(df["n_test"].to_numpy(dtype=float) * 3, 15, 200)
    ax.scatter(df["auc"], range(len(df)), s=sizes, c="#1565C0", alpha=0.75,
               edgecolor="black", linewidth=0.4)
    mean_auc = float(df["auc"].mean())
    ax.axvline(mean_auc, color="#E53935", linestyle="--", linewidth=1.2, label=f"mean = {mean_auc:.3f}")
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([f"fold {int(f)} (n={int(n)})" for f, n in zip(df["fold"], df["n_test"])],
                        fontsize=7)
    ax.set_xlabel("AUC", fontsize=8.5)
    ax.legend(fontsize=7.5)
    ax.tick_params(labelsize=7.5)
    _clean_axes(ax)
    plt.tight_layout()
    return _save(fig, "spatial_cv_folds")


def _fig_spatial_cv_buffering():
    """Two panels: how far does spatial dependence
    reach, and what does enforcing that gap cost?

    (a) Moran's I of the test residuals in 250 m distance bands, with the bands
        that are not significantly positive drawn hollow, and the estimated
        autocorrelation range marked.
    (b) Mean blocked-CV AUC against buffer width, with the fold standard
        deviation as the error band.

    Both are text-only in the manuscript otherwise, and the pair is the whole
    argument: the block SIZE clears the measured range, but the block EDGES do
    not, so the blocked estimate is an interval rather than a point.
    """
    corr_path = robustness.CORRELOGRAM_CSV
    buf_path = robustness.BUFFERED_CV_CSV
    if not (corr_path.exists() and buf_path.exists()):
        print(f"[report] skipping spatial_cv_buffering: need {corr_path.name} and {buf_path.name}")
        return None

    corr = pd.read_csv(corr_path).dropna(subset=["morans_i"])
    buf = pd.read_csv(buf_path).sort_values("buffer_m")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.3))

    # -- (a) correlogram --------------------------------------------------
    mid = (corr["lo_m"] + corr["hi_m"]) / 2.0
    sig = corr["significant"].to_numpy(dtype=bool)
    ax1.axhline(0.0, color="#9E9E9E", linewidth=0.8, zorder=1)
    ax1.plot(mid, corr["morans_i"], color="#1565C0", linewidth=1.2, zorder=2)
    ax1.scatter(mid[sig], corr["morans_i"][sig], s=26, c="#1565C0",
                edgecolor="black", linewidth=0.4, zorder=3, label="$p<0.05$")
    ax1.scatter(mid[~sig], corr["morans_i"][~sig], s=26, facecolor="white",
                edgecolor="#1565C0", linewidth=0.9, zorder=3, label="not significant")

    # the range is the lower edge of the first non-significant band
    nonsig = corr[~corr["significant"]]
    if len(nonsig):
        rng = float(nonsig["lo_m"].iloc[0])
        ax1.axvline(rng, color="#E53935", linestyle="--", linewidth=1.1, zorder=2)
        ax1.annotate(f"range $\\approx$ {rng:.0f} m", xy=(rng, ax1.get_ylim()[1]),
                     xytext=(4, -8), textcoords="offset points",
                     fontsize=7.5, color="#E53935", va="top")
    ax1.set_xlabel("Separation distance (m)", fontsize=8.5)
    ax1.set_ylabel("Moran's $I$ of test residuals", fontsize=8.5)
    ax1.set_title("(a) Empirical autocorrelation range", fontsize=9, fontweight="bold")
    ax1.legend(fontsize=7, frameon=False)
    ax1.tick_params(labelsize=7.5)

    # -- (b) buffer sweep -------------------------------------------------
    x = buf["buffer_m"].to_numpy(dtype=float)
    y = buf["mean_auc"].to_numpy(dtype=float)
    sd = buf["sd_auc"].to_numpy(dtype=float)
    ax2.fill_between(x, y - sd, y + sd, color="#1565C0", alpha=0.15,
                     label="$\\pm$1 SD over folds")
    ax2.plot(x, y, marker="o", markersize=5, color="#1565C0", linewidth=1.4,
             markeredgecolor="black", markeredgewidth=0.4)
    for xi, yi in zip(x, y):
        ax2.annotate(f"{yi:.3f}", xy=(xi, yi), xytext=(0, 7), textcoords="offset points",
                     ha="center", fontsize=7)
    if len(nonsig):
        ax2.axvline(rng, color="#E53935", linestyle="--", linewidth=1.1,
                    label="autocorrelation range")
    ax2.set_xlabel("Train–test buffer (m)", fontsize=8.5)   # en dash: "--" renders literally here
    ax2.set_ylabel("Mean blocked-CV AUC", fontsize=8.5)
    ax2.set_title("(b) Cost of enforcing the gap", fontsize=9, fontweight="bold")
    ax2.legend(fontsize=7, frameon=False, loc="lower left")
    ax2.tick_params(labelsize=7.5)

    _clean_axes(ax1, ax2)
    plt.tight_layout()
    return _save(fig, "spatial_cv_buffering")


# ----------------------------------------------------------------------
# Nice-to-have 10: equity_income_quintile
# ----------------------------------------------------------------------

def _fig_equity_income_quintile():
    csv_path = config.DATA / "equity" / "income_quintile_susceptibility.csv"
    if not csv_path.exists():
        print(f"[report] skipping equity_income_quintile: {csv_path} not found")
        return None

    df = pd.read_csv(csv_path)
    quintiles = sorted(df["income_q"].unique())
    sample_kinds = list(df["sample"].unique())
    x = np.arange(len(quintiles))
    width = 0.8 / max(len(sample_kinds), 1)

    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    for i, kind in enumerate(sample_kinds):
        sub = df[df["sample"] == kind].set_index("income_q").reindex(quintiles)
        offset = (i - (len(sample_kinds) - 1) / 2) * width
        yerr = np.vstack([
            (sub["mean"] - sub["q25"]).clip(lower=0).to_numpy(),
            (sub["q75"] - sub["mean"]).clip(lower=0).to_numpy(),
        ])
        ax.bar(x + offset, sub["mean"], width=width, yerr=yerr, capsize=2,
               label=str(kind).replace("_", " "))

    ax.set_xticks(x); ax.set_xticklabels(quintiles, fontsize=8)
    ax.set_ylabel("Mean susceptibility (IQR whiskers)", fontsize=8.5)
    ax.set_xlabel("Block-group income quintile (Q1 = lowest income)", fontsize=8.5)
    ax.set_title("Susceptibility by income quintile", fontsize=9.5, fontweight="bold")
    ax.legend(fontsize=7.5)
    ax.tick_params(labelsize=7.5)
    _clean_axes(ax)
    plt.tight_layout()
    return _save(fig, "equity_income_quintile")


# ----------------------------------------------------------------------
# Nice-to-have 11: calibration_auprc
# ----------------------------------------------------------------------

def _fig_calibration_auprc():
    csv_path = robustness.OUT_DIR / "imbalance_metrics.csv"
    if not csv_path.exists():
        print(f"[report] skipping calibration_auprc: {csv_path} not found")
        return None

    df = pd.read_csv(csv_path)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))

    by_auprc = df.sort_values("auprc")
    colors = ["#E53935" if "MaxEnt" in m else "#1565C0" for m in by_auprc["model"]]
    axes[0].barh(by_auprc["model"], by_auprc["auprc"], color=colors)
    axes[0].set_xlabel("AUPRC", fontsize=8.5)
    axes[0].set_title("(a) Precision-recall AUC", fontsize=9)

    by_brier = df.sort_values("brier", ascending=False)
    colors2 = ["#E53935" if "MaxEnt" in m else "#2E7D32" for m in by_brier["model"]]
    axes[1].barh(by_brier["model"], by_brier["brier"], color=colors2)
    axes[1].set_xlabel("Brier score (lower = better calibrated)", fontsize=8.5)
    axes[1].set_title("(b) Calibration (Brier score)", fontsize=9)

    for ax in axes:
        ax.tick_params(labelsize=7.5)
    _clean_axes(*axes)
    plt.suptitle("Calibration and precision-recall discrimination by model", fontsize=9.5, fontweight="bold")
    plt.tight_layout()
    return _save(fig, "calibration_auprc")


# ----------------------------------------------------------------------
# 11b. Predictor stack panel figure.
#
# Ported from the legacy scripts/make_fig02_predictors.py, which is superseded
# by this. That script hardcoded a 24-layer panel list including the two gridMET
# precipitation layers this pipeline never acquires -- it would raise on the
# canonical stack -- and hardcoded the removed set as {tri, tpi, median_income},
# which predates the log-transform re-screen that also drops spi and
# flow_accumulation. Here BOTH the panel list and the removed set are derived
# from the data: panels are whatever rasters exist under predictors/, and a
# layer is flagged "not used" iff it is absent from maxent.modeling_layers().
# ----------------------------------------------------------------------

PREDICTOR_FAMILY_COLOURS = {"terrain": "#2e6b3e", "soil": "#8a5a2b", "anthro": "#274b8a"}

PREDICTOR_PANEL_ORDER = [
    ("dem", "terrain"), ("slope", "terrain"), ("curvature", "terrain"),
    ("twi", "terrain"), ("spi", "terrain"), ("hand", "terrain"),
    ("flow_accumulation", "terrain"), ("distance_to_streams", "terrain"),
    ("drainage_density", "terrain"), ("tri", "terrain"), ("tpi", "terrain"),
    ("ksat", "soil"), ("available_water_storage", "soil"),
    ("depth_to_restriction", "soil"), ("hydrologic_soil_group", "soil"),
    ("curve_number", "soil"), ("nlcd_landcover", "soil"),
    ("impervious_pct", "soil"), ("precip_annual_mean", "soil"),
    ("precip_rx1day", "soil"),
    ("population_density", "anthro"), ("stormwater_density", "anthro"),
    ("distance_to_outfall", "anthro"), ("median_income", "anthro"),
]

PREDICTOR_PANEL_TITLES = {
    "dem": "Elevation (m)", "slope": "Slope (°)", "curvature": "Curvature",
    "twi": "Wetness index (TWI)", "spi": "Stream power index", "hand": "HAND (m)",
    "flow_accumulation": "Flow accumulation",
    "distance_to_streams": "Distance to streams (m)",
    "drainage_density": "Drainage density", "tri": "Ruggedness (TRI)",
    "tpi": "Position (TPI)", "ksat": "K$_{sat}$ (log$_{10}$ cm h$^{-1}$)",
    "available_water_storage": "Avail. water storage (mm)",
    "depth_to_restriction": "Depth to restriction (cm)",
    "hydrologic_soil_group": "Hydrologic soil group", "curve_number": "Curve number",
    "nlcd_landcover": "Land cover (NLCD)", "impervious_pct": "Impervious surface (%)",
    "precip_annual_mean": "Mean annual precip. (mm)",
    "precip_rx1day": "Max 1-day precip. (mm)",
    "population_density": "Population density (km$^{-2}$)",
    "stormwater_density": "Stormwater density",
    "distance_to_outfall": "Distance to outfall (m)",
    "median_income": "Median income (\\$)",
}

_PANEL_FAM_CMAP = {"terrain": "YlGnBu", "soil": "YlOrBr", "anthro": "magma"}
_PANEL_SPECIAL_CMAP = {"dem": "gist_earth"}
_PANEL_DIVERGING = {"curvature", "tpi"}
_HSG_COLOURS = ["#2c7bb6", "#abd9e9", "#fdae61", "#d7191c"]


def _panel_tick(v):
    return "0" if abs(v) < 1e-6 else f'{float(f"{v:.3g}"):g}'


def _fig_predictor_stack(downsample=3, ncols=4):
    """Every predictor raster on disk, family-grouped, with the layers that do
    not survive to the final model flagged in red."""
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    from mpl_toolkits.axes_grid1 import ImageGrid
    import rasterio

    pred_dir = config.DATA / "predictors"
    if not pred_dir.exists() or not any(pred_dir.glob("*.tif")):
        print(f"[report] skipping predictor_stack: {pred_dir} not populated")
        return None

    # Layers acquired for the equity overlay are not candidate predictors and
    # have no place in a predictor-stack figure (maxent.EQUITY_ONLY_LAYERS).
    on_disk = {p.stem for p in pred_dir.glob("*.tif")} - set(maxent.EQUITY_ONLY_LAYERS)
    panels = [(n, f) for n, f in PREDICTOR_PANEL_ORDER if n in on_disk]
    extra = sorted(on_disk - {n for n, _ in PREDICTOR_PANEL_ORDER})
    panels += [(n, "anthro") for n in extra]     # never silently omit a layer
    kept = set(maxent.modeling_layers())
    dropped = {n for n, _ in panels if n not in kept}

    def _load(name):
        with rasterio.open(pred_dir / f"{name}.tif") as src:
            arr = src.read(1).astype("float64")
            nodata = src.nodata
        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)
        return arr[::downsample, ::downsample]

    # Four columns is the compromise between panel size and page share: the
    # figure comes out at aspect 1.13, so as a [t] float it takes roughly the
    # top two thirds of the measure and leaves the rest for text, while the
    # panels stay large enough to read. figsize derives from ncols so the
    # panels keep their size whatever the column count.
    nrows = int(np.ceil(len(panels) / ncols))
    fig = plt.figure(figsize=(3.0 * ncols, 1.77 * nrows))
    grid = ImageGrid(fig, [0.02, 0.05, 0.96, 0.93], nrows_ncols=(nrows, ncols),
                     axes_pad=(0.58, 0.34), aspect=False, cbar_mode="each",
                     cbar_location="right", cbar_size="5%", cbar_pad=0.04)

    for i, (name, fam) in enumerate(panels):
        ax, cax = grid[i], grid.cbar_axes[i]
        arr = _load(name)
        is_dropped = name in dropped
        if name in screen.CATEGORICAL:
            vals = np.unique(arr[np.isfinite(arr)])
            cmap = (ListedColormap(_HSG_COLOURS[:len(vals)])
                    if name == "hydrologic_soil_group"
                    else ListedColormap(plt.cm.tab20(np.linspace(0, 1, len(vals)))))
            # Explicit float-typed mapping rather than np.vectorize: vectorize
            # infers its output dtype from the FIRST element, so a raster whose
            # first pixel is a valid class code gets an int dtype and then raises
            # on the NaN nodata cells (the legacy script only survived this
            # because its first pixel happened to be nodata).
            idx = np.full(arr.shape, np.nan, dtype="float64")
            for j, v in enumerate(vals):
                idx[arr == v] = j
            ax.imshow(idx, cmap=cmap, interpolation="nearest", aspect="auto")
            cax.set_visible(False)
        else:
            if name in _PANEL_DIVERGING:
                lim = np.nanpercentile(np.abs(arr), 98)
                vmin, vmax, cmap, ticks = -lim, lim, "RdBu_r", [-lim, 0, lim]
            else:
                vmin, vmax = np.nanpercentile(arr, [2, 98])
                cmap = _PANEL_SPECIAL_CMAP.get(name, _PANEL_FAM_CMAP[fam])
                ticks = [vmin, vmax]
            im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax,
                           interpolation="nearest", aspect="auto")
            cb = fig.colorbar(im, cax=cax, ticks=ticks)
            cax.set_yticklabels([_panel_tick(t) for t in ticks], fontsize=6)
            cb.outline.set_linewidth(0.3)
            cax.tick_params(length=1.5, width=0.3)

        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#c0392b" if is_dropped else "0.6")
            # 21 panels on one page means each is printed small; at 1.6 pt the
            # frame that marks a screened-out layer disappeared on the page.
            sp.set_linewidth(3.2 if is_dropped else 0.5)
            sp.set_zorder(6)
        ax.set_title(PREDICTOR_PANEL_TITLES.get(name, name.replace("_", " ")),
                     fontsize=7.0, color=("#c0392b" if is_dropped else "black"), pad=2.5)
        ax.add_patch(Rectangle((0.03, 0.87), 0.075, 0.10, transform=ax.transAxes,
                               facecolor=PREDICTOR_FAMILY_COLOURS[fam],
                               edgecolor="white", linewidth=0.7, zorder=5))

    for j in range(len(panels), nrows * ncols):      # hide unused slots
        grid[j].set_visible(False)
        grid.cbar_axes[j].set_visible(False)

    handles = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor=PREDICTOR_FAMILY_COLOURS["terrain"],
               markersize=8, label="Terrain / hydro-geomorphic"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=PREDICTOR_FAMILY_COLOURS["soil"],
               markersize=8, label="Soil / land cover"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=PREDICTOR_FAMILY_COLOURS["anthro"],
               markersize=8, label="Anthropogenic / stormwater"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="w",
               markeredgecolor="#c0392b", markeredgewidth=3.2, markersize=8,
               label="Not used in the final model"),
    ]
    # A single row beneath the grid, which is where it reads best: stacking it
    # into an empty panel slot shrank it to a quarter of the width and the type
    # went with it. One row of four entries can be set larger and still fit.
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9.5,
               frameon=False, bbox_to_anchor=(0.5, 0.008))

    print(f"[report] predictor_stack: {len(panels)} panels, "
          f"{len(dropped)} flagged as not used ({', '.join(sorted(dropped))})")
    return _save(fig, "predictor_stack")


# ----------------------------------------------------------------------
# 12-13. Cross-model variable importance (main-text pair).
#
# Ported from the legacy scripts/make_fig03_importance.py and
# make_fig04_importance_permodel.py, which are superseded by these. Both read
# model_compare.variable_importance_comparison()'s canonical output instead of
# the stale 21-predictor CSV those scripts assumed, and both derive the
# predictor count from the data rather than hardcoding it -- the legacy pair
# hardcoded "of 21" in the colourbar and rank scale, and read MaxEnt column
# names ("perm_mean"/"perm_sd") that the canonical
# variable_importance_combined.csv does not use.
# ----------------------------------------------------------------------

PREDICTOR_LABELS = {
    "distance_to_outfall": "Distance to outfall",
    "available_water_storage": "Avail. water storage",
    "dem": "Elevation (DEM)",
    "depth_to_restriction": "Depth to restriction",
    "nlcd_landcover": "Land cover (NLCD)",
    "ksat": "K$_{sat}$",
    "hydrologic_soil_group": "Hydrologic soil group",
    "drainage_density": "Drainage density",
    "population_density": "Population density",
    "distance_to_streams": "Distance to streams",
    "curve_number": "Curve number",
    "impervious_pct": "Impervious surface %",
    "stormwater_density": "Stormwater density",
    "slope": "Slope",
    "twi": "TWI",
    "curvature": "Curvature",
    "hand": "HAND",
}

# (column, short label) for the five models with a comparable importance measure.
IMPORTANCE_COLUMNS = [
    ("maxent_jk_norm", "MaxEnt"),
    ("rf_perm_mean", "RF"),
    ("xgb_gain_norm", "XGB"),
    ("lgb_gain_norm", "LightGBM"),
    ("cat_gain_norm", "CatBoost"),
]

HIGHLIGHT_PREDICTOR = "distance_to_outfall"


def _importance_frame():
    """model_compare's canonical cross-model importance table, or None if it
    has not been built yet (printed skip, matching this module's contract)."""
    path = model_compare.OUT_DIR / "importance_comparison.csv"
    if not path.exists():
        print(f"[report] skipping importance figures: {path} not found "
              f"(run model_compare.variable_importance_comparison())")
        return None
    return pd.read_csv(path)


def _label(p):
    return PREDICTOR_LABELS.get(p, p.replace("_", " ").capitalize())


def _fig_importance_cross_model(top_n=12):
    """(a) MaxEnt permutation importance for the leading predictors;
    (b) importance-rank heatmap across the five comparable models."""
    imp = _importance_frame()
    if imp is None:
        return None
    vic_path = config.DATA / "maxent" / "variable_importance_combined.csv"
    if not vic_path.exists():
        print(f"[report] skipping importance_cross_model: {vic_path} not found")
        return None

    mx = pd.read_csv(vic_path).sort_values(
        "permutation_importance_mean", ascending=False).reset_index(drop=True)
    n_pred = len(imp)
    top = mx.head(top_n)["predictor"].tolist()

    ranks = (imp.set_index("predictor")[[c for c, _ in IMPORTANCE_COLUMNS]]
             .rank(ascending=False, method="min").reindex(top))

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.4, 5.6),
                                    gridspec_kw={"width_ratios": [1.05, 1.0]})

    sub = mx.head(top_n).iloc[::-1]
    colors = ["#c0392b" if p == HIGHLIGHT_PREDICTOR else "#4472a4"
              for p in sub["predictor"]]
    axA.barh([_label(p) for p in sub["predictor"]],
             sub["permutation_importance_mean"],
             xerr=sub["permutation_importance_sd"], color=colors,
             edgecolor="white", linewidth=0.5,
             error_kw=dict(lw=0.7, ecolor="0.4"))
    axA.set_xlabel("MaxEnt permutation importance (%)", fontsize=9.5)
    axA.tick_params(labelsize=8.5)
    axA.set_title("(a) MaxEnt permutation importance", fontsize=10.5,
                  fontweight="bold", loc="left")
    axA.margins(y=0.01)
    _clean_axes(axA)

    M = ranks.to_numpy()
    im = axB.imshow(M, cmap="YlGnBu", vmin=1, vmax=n_pred, aspect="auto")
    axB.set_xticks(range(len(IMPORTANCE_COLUMNS)))
    axB.set_xticklabels([n for _, n in IMPORTANCE_COLUMNS], fontsize=8.5)
    axB.set_yticks(range(len(top)))
    axB.set_yticklabels([_label(p) for p in top], fontsize=8.5)
    midpoint = n_pred / 2
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            axB.text(j, i, f"{int(M[i, j])}", ha="center", va="center", fontsize=8,
                     color="0.12" if M[i, j] <= midpoint else "white")
    axB.set_title("(b) Importance rank across models (1 = top)", fontsize=10.5,
                  fontweight="bold", loc="left")
    cb = fig.colorbar(im, ax=axB, fraction=0.045, pad=0.03)
    cb.set_label(f"rank (of {n_pred} predictors)", fontsize=8)
    cb.ax.tick_params(labelsize=7.5)

    fig.tight_layout()
    return _save(fig, "importance_cross_model")


def _fig_importance_per_model():
    """Full per-predictor importance profile for each of the five comparable
    models, each panel sorted by its own importance and normalised to a share."""
    imp = _importance_frame()
    if imp is None:
        return None

    panels = [("maxent_jk_norm", "MaxEnt (jackknife)"),
              ("rf_perm_mean", "Random forest (permutation)"),
              ("xgb_gain_norm", "XGBoost (gain)"),
              ("lgb_gain_norm", "LightGBM (gain)"),
              ("cat_gain_norm", "CatBoost (gain)")]

    fig, axes = plt.subplots(1, 5, figsize=(17, 6.2))
    for ax, (col, title) in zip(axes, panels):
        s = imp[["predictor", col]].copy()
        total = s[col].clip(lower=0).sum()
        s[col] = s[col].clip(lower=0) / total if total > 0 else 0.0
        s = s.sort_values(col, ascending=True)
        colors = ["#c0392b" if p == HIGHLIGHT_PREDICTOR else "#5b7fa6"
                  for p in s["predictor"]]
        ax.barh([_label(p) for p in s["predictor"]], s[col],
                color=colors, edgecolor="white", linewidth=0.4)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Importance (fraction)", fontsize=8.5)
        ax.tick_params(axis="y", labelsize=7.3)
        ax.tick_params(axis="x", labelsize=7.5)
        _clean_axes(ax)

    fig.tight_layout()
    return _save(fig, "importance_per_model")


def figures() -> list:
    """Writes every paper figure that has a real data source (plus 4
    nice-to-have results figures) under ``FIGURES_DIR`` as PNG+PDF pairs
    (see ``_save``) and returns the list of PNG paths written -- each has
    a same-stem ``.pdf`` sibling. Never fabricates: any missing source is
    a printed skip, never a raised error and never a stand-in image.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    paths = []

    def _add(p):
        if p is not None:
            paths.append(p)

    _add(_fig_study_area_hwm())
    _add(_fig_susceptibility_maps())
    _add(_fig_predictor_stack())
    _add(_fig_jackknife_importance(table3()))
    _add(_fig_importance_cross_model())
    _add(_fig_importance_per_model())
    _add(_fig_response_curves())
    _add(_fig_shap())
    _add(_fig_internal_validation_curves())
    _add(_fig_gfi_comparison())
    _add(_fig_sensitivity_summary(table4()))
    _add(_fig_susceptibility_maps_comparison())

    _add(_fig_model_comparison_auc())
    _add(_fig_spatial_cv_folds())
    _add(_fig_spatial_cv_buffering())
    _add(_fig_equity_income_quintile())
    _add(_fig_calibration_auprc())

    return paths


# ----------------------------------------------------------------------
# 14. MaxEnt response curves -- direction and shape for the leading predictors.
#
# Jackknife and permutation importance rank predictors; neither says which way
# the effect runs or whether it is linear. SHAP is the usual post-hoc answer in
# this literature, but MaxEnt emits the marginal response directly, so the
# fitted model is read rather than approximated.
# ----------------------------------------------------------------------

# Category labels for the two nominal predictors, so response-curve panels read
# as land-cover classes rather than raw legend codes. NLCD 2021 legend; HSG is
# the A-D hydrologic soil group encoded 1-4 by predictors.acquire_hsg.
NLCD_CLASS_LABELS = {
    11: "Open water", 21: "Devel. open", 22: "Devel. low", 23: "Devel. med",
    24: "Devel. high", 31: "Barren", 41: "Decid. forest", 42: "Everg. forest",
    43: "Mixed forest", 52: "Shrub", 71: "Grassland", 81: "Pasture",
    82: "Crops", 90: "Woody wetland", 95: "Herb. wetland",
}
HSG_CLASS_LABELS = {1: "A", 2: "B", 3: "C", 4: "D"}


def _category_label(pred, code):
    if pred == "nlcd_landcover":
        return NLCD_CLASS_LABELS.get(int(code), str(int(code)))
    if pred == "hydrologic_soil_group":
        return HSG_CLASS_LABELS.get(int(code), str(int(code)))
    return str(int(code))


# x-axis label for each response-curve panel (the panel title names the
# predictor; the axis gives the quantity and its unit). Distance to outfall is
# plotted in km so the axis matches the text ("the first 2 km").
RESPONSE_XLABELS = {
    "distance_to_outfall": "Distance to outfall (km)",
    "available_water_storage": "Available water storage (mm)",
    "depth_to_restriction": "Depth to restriction (cm)",
    "dem": "Elevation (m)",
    "ksat": "K$_{sat}$ (log$_{10}$ cm h$^{-1}$)",
    "nlcd_landcover": "NLCD 2021 class",
    "hydrologic_soil_group": "Hydrologic soil group",
}
_RESPONSE_X_SCALE = {"distance_to_outfall": 1e-3}   # m -> km


def _maxent_training_ranges(lambdas_path):
    """Per-predictor (min, max) of the training data, read from a MaxEnt
    ``.lambdas`` file. Its linear-feature lines are ``name, lambda, min, max``;
    outside that range MaxEnt clamps the prediction, so the response curves it
    writes (which run about 10% past each end) are flat there and, for
    distances, extend below zero."""
    ranges = {}
    for line in Path(lambdas_path).read_text().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 4 and parts[0].replace("_", "").isalnum():
            ranges[parts[0]] = (float(parts[2]), float(parts[3]))
    return ranges


def _fig_response_curves(top_n=6):
    """Marginal response of predicted susceptibility to each of the leading
    predictors, other predictors held at their means, shown only over the
    range of each predictor in the training data."""
    curves_csv = maxent.RESPONSE_DIR / "response_curves.csv"
    if not curves_csv.exists():
        print(f"[report] skipping response_curves: {curves_csv} not found "
              f"(run maxent.response_curves())")
        return None
    vic_path = config.DATA / "maxent" / "variable_importance_combined.csv"
    if not vic_path.exists():
        print(f"[report] skipping response_curves: {vic_path} not found")
        return None

    curves = pd.read_csv(curves_csv)
    order = (pd.read_csv(vic_path)
             .sort_values("permutation_importance_mean", ascending=False)["predictor"]
             .tolist())
    order = [p for p in order if p in set(curves["predictor"])][:top_n]
    lambdas_path = maxent.RESPONSE_DIR / "flood.lambdas"
    ranges = _maxent_training_ranges(lambdas_path) if lambdas_path.exists() else {}
    if not ranges:
        print(f"[report] response_curves: {lambdas_path} missing or unreadable -- "
              f"curves are drawn over MaxEnt's full (extrapolated) range")

    ncols = 3
    nrows = int(np.ceil(len(order) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(9.0, 2.6 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, pred in zip(axes, order):
        sub = curves[curves["predictor"] == pred].sort_values("x")
        colour = "#c0392b" if pred == HIGHLIGHT_PREDICTOR else "#4472a4"
        if pred in screen.CATEGORICAL:
            # Nominal class codes have no ordering, so a line joining them would
            # imply a gradient between arbitrary legend values that does not exist.
            grouped = sub.groupby(sub["x"].round().astype(int))["y"].mean()
            names = [_category_label(pred, c) for c in grouped.index]
            ax.bar(names, grouped.values, color=colour, width=0.75)
            ax.tick_params(axis="x", labelrotation=90, labelsize=6.5)
        else:
            if pred in ranges:
                lo, hi = ranges[pred]
                sub = sub[(sub["x"] >= lo) & (sub["x"] <= hi)]
            ax.plot(sub["x"] * _RESPONSE_X_SCALE.get(pred, 1.0), sub["y"],
                    color=colour, linewidth=1.8)
        ax.set_title(_label(pred), fontsize=9,
                     color=colour if pred == HIGHLIGHT_PREDICTOR else "black")
        ax.set_xlabel(RESPONSE_XLABELS.get(pred, PREDICTOR_PANEL_TITLES.get(pred, "")),
                      fontsize=8)
        ax.set_ylabel("Susceptibility", fontsize=8)
        ax.tick_params(labelsize=7.5)
        _clean_axes(ax)
    for ax in axes[len(order):]:
        ax.set_visible(False)

    fig.tight_layout()
    return _save(fig, "response_curves")


# ----------------------------------------------------------------------
# 15. SHAP attributions for the tree ensembles.
#
# An independent check on the importance ranking: SHAP decomposes each
# individual prediction rather than measuring model-level gain, so agreement
# with the jackknife/permutation ordering is corroboration by a different
# route rather than a restatement. MaxEnt is absent by design -- see
# model_compare.shap_importance's docstring.
# ----------------------------------------------------------------------

def _fig_shap(top_n=8):
    """(a) mean |SHAP| share per predictor for each model;
    (b) the direction each predictor pushes, as a rank correlation.

    MaxEnt is drawn in orange and the four tree ensembles in blues: its
    attributions come from an exact closed form (maxent.exact_shap) rather than
    TreeExplainer, so the visual separation is deliberate."""
    path = model_compare.SHAP_CSV
    if not path.exists():
        print(f"[report] skipping shap: {path} not found "
              f"(run model_compare.shap_importance())")
        return None
    df = pd.read_csv(path)

    order = (df.groupby("predictor")["share"].mean()
             .sort_values(ascending=False).head(top_n).index.tolist())
    models = [m for m in model_compare.SHAP_MODELS if m in set(df["model"])]

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 0.52 * top_n + 2.2))
    y = np.arange(len(order))
    h = 0.8 / max(len(models), 1)
    # one colour per model -- MUST be at least len(SHAP_MODELS) long, or two
    # models silently share a colour and the legend becomes unreadable
    _TREE_BLUES = ["#4472a4", "#5b9bd5", "#8FAADC", "#2E5A88"]
    colour = {m: _TREE_BLUES[i % len(_TREE_BLUES)]
              for i, m in enumerate(m for m in models if m != "MaxEnt")}
    colour["MaxEnt"] = "#d1813a"   # orange vs blue: safest dichromat contrast
    assert len({colour[m] for m in models}) == len(models),         f"palette collision across {models}"

    ax = axes[0]
    for k, m in enumerate(models):
        sub = df[df["model"] == m].set_index("predictor").reindex(order)
        ax.barh(y + k * h - 0.4 + h / 2, sub["share"], height=h,
                label=m, color=colour[m], edgecolor="white", linewidth=0.3)
    ax.set_yticks(y); ax.set_yticklabels([_label(p) for p in order], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP| (share of model total)", fontsize=9)
    ax.set_title("(a) Attribution magnitude", fontsize=10, fontweight="bold", loc="left")
    ax.legend(fontsize=7.5)
    _clean_axes(ax)

    ax = axes[1]
    for k, m in enumerate(models):
        sub = df[df["model"] == m].set_index("predictor").reindex(order)
        ax.barh(y + k * h - 0.4 + h / 2, sub["direction_corr"], height=h,
                color=colour[m], edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="0.3", linewidth=0.8)
    ax.set_yticks(y); ax.set_yticklabels([])
    ax.invert_yaxis()   # must match panel (a): the two panels share row labels
    ax.set_xlabel("Spearman(predictor value, its SHAP value)", fontsize=9)
    ax.set_title("(b) Direction of effect", fontsize=10, fontweight="bold", loc="left")
    ax.set_xlim(-1, 1)
    _clean_axes(ax)

    fig.tight_layout()
    return _save(fig, "shap_importance")
