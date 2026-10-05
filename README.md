# Urban flood susceptibility from high-water marks, Manhattan, Kansas

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23171162.svg)](https://doi.org/10.5281/zenodo.23171162)

This repository holds the code and result tables for the paper

> Monavarian, A., Abadifard, S., Heatherman, W. J., McGinty, H. K., and Sharda, V. *Infrastructure-Aware Urban Flood Susceptibility from Multi-Event High-Water Marks: A Multi-Model Comparison.* Submitted to *Journal of Hydrology*.

The pipeline maps urban flood susceptibility for Manhattan, Kansas, with MaxEnt. The presence data are 277 high-water marks that the City of Manhattan surveyed after 11 floods between 2007 and 2024. The 17 predictors cover terrain, flow routing, soils, land cover, population and the City's stormwater network. The pipeline also

- compares MaxEnt with five machine-learning models on the same held-out points,
- tests transfer to new areas (2 km spatial blocks) and to new floods (leave-one-event-out cross-validation),
- runs the ablation, sensitivity and equity analyses reported in the paper, and
- compares the mapped susceptibility with the FEMA floodplain, NWS inundation maps and a Sentinel-1 flood extent.

## Data availability

Two inputs belong to the City of Manhattan and are **not** in this repository.

| Folder the code reads | Content |
|---|---|
| `data/raw/hwm/` | high-water-mark survey files, one folder per flood |
| `data/raw/infrastructure/` | the as-built stormwater geodatabase, `extracted/commondata/mhk_stormwater_export.gdb` |

Requests for these data should be directed to the City of Manhattan, Kansas, Public Works Department. For the same reason, the tables that list the mark locations (the presence-point tables and MaxEnt's sample files) have been left out. All figures are included.

The other inputs are public, and the code downloads them on first run.

- USGS 3DEP 10 m elevation (`py3dep`), routed with WhiteboxTools
- NHD High Resolution flowlines (`pynhd`)
- gNATSGO soils (Microsoft Planetary Computer), NRCS Soil Data Access and POLARIS soil conductivity
- NLCD 2021 land cover and imperviousness (`pygeohydro`)
- Census ACS 5-year estimates and TIGER city limits, roads and block groups (needs a free Census API key, see Setup)
- Sentinel-1 RTC scenes (Microsoft Planetary Computer), for the external comparison

Two comparison layers have to be downloaded by hand, reprojected to EPSG:26914 and saved as GeoPackages. They are used only for the external comparisons.

- The FEMA National Flood Hazard Layer for the study area, as `data/raw/fema/flood_hazard_zones.gpkg`, with the fields `FLD_ZONE` and `ZONE_SUBTY`.
- The NWS AHPS flood inundation maps for Wildcat Creek at Scenic Drive (gauge MWCK1), as `data/raw/nws/fim_<category>.gpkg` for the categories `low`, `near`, `minor`, `moderate` and `major`.

## What is here

| Path | Content |
|---|---|
| `pipeline/` | the analysis, one module per stage (table below) |
| `notebooks/00_canonical_pipeline.ipynb` | runs every stage in order |
| `tests/` | regression tests |
| `data/processed/` | result tables (CSV) from the run reported in the paper |
| `reports/tables/` | the paper's tables (CSV and LaTeX) |
| `reports/figures/` | the figures (PNG and PDF), including all of the paper's |
| `env/` | the tested environment |

| Module | What it does |
|---|---|
| `config.py` | seed, the 277-point modeling set, train/test split, spatial and event folds, Java and MaxEnt paths |
| `hwm.py` | parses the survey files, reprojects them to EPSG:26914 and thins them to one point per 30 m |
| `predictors.py` | downloads the predictors, aligns them to the 10 m grid, routes flow and checks the stack |
| `screen.py` | multicollinearity screen (Pearson, then VIF) and the presence-versus-background contrast of each predictor |
| `maxent.py` | MaxEnt runs, AICc tuning, the final and evaluation models, the jackknife and the response curves |
| `evaluate.py` | held-out AUC and thresholds, the terrain-only GFI baseline and the external comparisons |
| `model_compare.py` | the five machine-learning models on the same test set, variable importance and SHAP |
| `robustness.py` | spatial-block and leave-one-event-out validation, ablation and significance tests |
| `sensitivity.py` | background, resolution, output-format and event-pooling sensitivity |
| `equity.py` | the Census block-group overlay and its corrected correlations |
| `report.py` | builds the paper's tables and figures from the saved results |
| `qa.py` | small checks shared by the stages |

Comments in the code sometimes mention "the original notebook". These are the project's first analysis notebooks, which this pipeline replaces. They are not included.

## Setup

The tested environment is Python 3.14.5 with the package versions pinned in `env/requirements.lock.txt`. `env/RUNTIME.md` lists the components.

```bash
py -3.14 -m venv flood_env
flood_env/Scripts/python -m pip install -r env/requirements.lock.txt
```

Three more things are needed.

- **MaxEnt 3.4.4.** Download `maxent.jar` from the [MaxEnt website](https://biodiversityinformatics.amnh.org/open_source/maxent/) and save it as `tools/maxent/maxent.jar`, or point the `FLOOD_MAXENT_JAR` environment variable to it.
- **Java 17.** Put `java` on `PATH`, or point `FLOOD_JAVA` to the executable.
- **A Census API key.** Request one at <https://api.census.gov/data/key_signup.html> and save it in a file named `.census_api_key` in the repository root. Git ignores this file.

## Running the analysis

1. Copy the City's two datasets into `data/raw/hwm/` and `data/raw/infrastructure/`, and the FEMA and NWS layers into `data/raw/fema/` and `data/raw/nws/`.
2. Run `notebooks/00_canonical_pipeline.ipynb` from top to bottom. The first run downloads the public layers. The notebook then rebuilds everything under `data/processed/` and `reports/`. A full run fits several hundred MaxEnt models.

## Reproducibility

The random seed is `pipeline.config.GLOBAL_SEED = 42`. MaxEnt 3.4.4 has no seed of its own, so the published map, which is the mean of ten bootstrap replicates, changes slightly between runs. The AUCs reported in the paper come from single fits and reproduce to about four decimal places. XGBoost results also depend on the XGBoost version (see `env/RUNTIME.md`).

## Tests

```bash
flood_env/Scripts/python -m pytest -q
```

Most tests run the pipeline on the study data and check the values reported in the paper. Without the City's data, the 227 unit tests that use synthetic data pass, 115 tests that need the study data fail at the first missing input, and the other 48 are skipped. Some tests download public layers when they run.

## Citation

Please cite the paper above. To cite the code itself, use the archived release.

> Monavarian, A., Abadifard, S., Heatherman, W. J., McGinty, H. K., and Sharda, V. (2026). *Code for: Infrastructure-Aware Urban Flood Susceptibility from Multi-Event High-Water Marks: A Multi-Model Comparison* (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.23171162

## License

The code is released under the MIT License (see `LICENSE`). The license does not cover the City of Manhattan's data.

## Contact

Alireza Monavarian, Kansas State University, alirezam@ksu.edu
