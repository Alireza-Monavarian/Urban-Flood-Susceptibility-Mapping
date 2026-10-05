# Tested runtime

The numbers reported in the paper come from this environment. If any component changes, results can move.

## Recreate it

```bash
py -3.14 -m venv flood_env
flood_env/Scripts/python -m pip install -r env/requirements.lock.txt
```

On macOS or Linux, use `python3.14 -m venv flood_env` and `flood_env/bin/python`.

[`requirements.lock.txt`](requirements.lock.txt) lists all 203 packages with exact versions, taken from `pip freeze --all` of the tested interpreter. [`../environment.yml`](../environment.yml) is a conda recipe with mostly unpinned versions and is not the tested path.

## Components

| Component | Version | Notes |
|---|---|---|
| Python | 3.14.5 | virtual environment, packages from the lock file |
| numpy / pandas / scipy | 2.4.6 / 3.0.3 / 1.17.1 | |
| scikit-learn | 1.9.0 | logistic regression, random forest |
| xgboost | 3.3.0 | results depend on this version, see below |
| lightgbm / catboost | 4.6.0 / 1.2.10 | |
| shap | 0.52.0 | `TreeExplainer` for the four tree ensembles |
| statsmodels | 0.14.6 | variance inflation factors |
| geopandas / rasterio / rioxarray | 1.1.3 / 1.5.0 / 0.22.0 | |
| xarray / shapely / pyproj | 2026.4.0 / 2.1.2 / 3.7.2 | |
| matplotlib | 3.10.9 | |
| Java | Temurin 17.0.19+10 | `java` on `PATH`, or set `FLOOD_JAVA`, or place a JDK under `tools/` |
| MaxEnt | 3.4.4 | `tools/maxent/maxent.jar`, or set `FLOOD_MAXENT_JAR` |
| WhiteboxTools (`whitebox`) | 2.3.6 | D8 flow routing; downloads its binary on first use |

The random seed for everything the pipeline controls is `pipeline.config.GLOBAL_SEED = 42`.

## What does not reproduce exactly

MaxEnt 3.4.4 has no random seed. The published susceptibility map is the mean of a ten-replicate bootstrap, so it differs slightly from run to run. Single fits are stable to about 1e-5 in AUC, because MaxEnt still draws its own background without a seed. The tests therefore pin those values with a small tolerance.

XGBoost results depend on the library version. Under xgboost 3.3.0 the XGBoost AUPRC is 0.771, and the Nadeau–Bengio p-value for MaxEnt against XGBoost is 0.23. An earlier, unrecorded version gave 0.809 and 0.53. The conclusion is the same in both cases: the difference is within cross-validation noise. Use the lock file to reproduce the reported values.
