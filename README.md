# Predicting Internet Connectivity from Satellite Imagery

Predicting internet connectivity access across India using satellite imagery and geospatial foundation models, with a geographic out-of-distribution evaluation on Northeast India.

## Problem

Reliable maps of internet access are critical for infrastructure planning, yet ground-truth connectivity data is sparse in developing regions. This project investigates whether satellite imagery -- processed through geospatial foundation models (Prithvi-EO-2.0) and classical ML baselines -- can predict internet connectivity at the sub-national level. A key challenge is **geographic distribution shift**: models are trained on most of India but evaluated on the geographically and socioeconomically distinct Northeast region.

## Data

| Source | Description |
|--------|-------------|
| [Harmonized Landsat Sentinel-2 (HLS)](https://hls.gsfc.nasa.gov/) | 6-band satellite imagery, 224x224 px patches at 30 m resolution (6,970 patches) |
| [Ookla Speedtest Intelligence](https://www.ookla.com/ookla-for-good/open-data) | Internet speed-test tile data (Q1 2019), used to construct connectivity labels |
| Engineered features | Nighttime lights (VIIRS), population density (WorldPop), road density (OSM), elevation/slope (SRTM) |

**Label construction:** Two aggregate regression targets are derived from Ookla sub-tiles per patch:
- `coverage_fraction` -- spatial coverage (proportion of zoom-16 tiles with data)
- `log_mean_tests` -- log-transformed mean test density

Binary connectivity labels are also used for baseline comparison.

**Train/test split:** Geographic hold-out -- Northeast India (lon > 89, lat > 21) is reserved as the test set; remaining India is split 80/20 for train/val with stratification.

## Methods

The project compares three tiers of models:

### Tier 1: Classical Baselines (Notebooks 02-03)
- **XGBoost** on engineered socioeconomic features (nightlights, population, roads, elevation)
- **MOSAIKS** random kitchen-sink features + Ridge regression
- **ResNet-18** (ImageNet-pretrained) + logistic/linear head
- **DINOv2** (ViT-L/14, frozen) + logistic/linear head

### Tier 2: Geospatial Foundation Model (Notebooks 04-05)
- **Prithvi-EO-2.0-300M** (frozen encoder, 1024-d embeddings)
  - Linear probe: `LayerNorm(1024) -> Dropout(0.1) -> Linear(1024, 1)`
  - MLP head: `LayerNorm(1024) -> Linear(1024, 256) -> GELU -> Dropout(0.2) -> Linear(256, 128) -> GELU -> Dropout(0.2) -> Linear(128, 1)`

### Tier 3: Feature Fusion (Notebook 06)
- **XGBoost + Prithvi embeddings** -- combines engineered socioeconomic features with frozen Prithvi-EO embeddings, with ablation over feature subsets

### Qualitative Analysis (Notebook 07)
- Spatial error maps, regional breakdowns, satellite overlay case studies

## Notebooks

Run notebooks in order on **Google Colab with GPU** (T4 minimum). Each notebook is self-contained: it installs dependencies, clones the repo, and syncs data from Google Drive.


1. `01_data_exploration.ipynb`: download data and create train inputs 
2. `02_binary_baselines.ipynb`: five binary classification baselines (XGBoost, MOSAIKS, ResNet-18, DINOv2, Prithvi linear) 
3. `03_aggregate_baselines.ipynb`: same five models retrained on aggregate regression targets 
4. `04_prithvi_linear_probe.ipynb`: Prithvi-EO-2.0 frozen encoder + linear regression head 
5. `05_prithvi_mlp_head.ipynb`: Prithvi-EO-2.0 frozen encoder + 2-layer MLP regression head 
6. `06_xgboost_prithvi_fusion.ipynb`: XGBoost combining engineered features + Prithvi embeddings, with ablation 
7. `07_qualitative_analysis.ipynb`: spatial error analysis, regional summaries, satellite overlay case studies 

## Evaluation

**Primary metrics** (regression): MAE, RMSE, Spearman rho on continuous targets.

**Secondary metrics** (binary task): PR-AUC, ROC-AUC, F1 at val-optimal threshold -- regression predictions are binarised and compared against binary ground-truth labels.

All metrics are reported on the **Northeast India geographic hold-out test set**.

## Repository Structure

```
satellite-internet-prediction/
├── notebooks/          # Jupyter notebooks (01-07), run on Colab
├── data/
│   ├── raw/            # GeoTIFF patches, Ookla CSV (not committed)
│   └── processed/      # Patch metadata CSV (not committed)
├── outputs/
│   ├── results/        # Metrics CSVs per model
│   ├── models/         # Trained model artifacts (.pkl, .json)
│   ├── figures/        # Generated plots (not committed)
│   ├── features/       # Cached Prithvi embeddings (not committed)
│   └── maps/           # Spatial visualisations
├── .gitignore
└── README.md
```

## Reproducibility

1. Open any notebook in Google Colab (GPU runtime).
2. The first cell installs all dependencies (`terratorch`, `rasterio`, `scikit-learn`, etc.).
3. Cell 0.2 clones this repository; Cell 0.3 syncs patch data from Google Drive.
4. Run all cells sequentially. Results are saved to `outputs/`.

**Seeds:** All notebooks set `random.seed(42)`, `np.random.seed(42)`, `torch.manual_seed(42)` at the top for reproducibility.

## Key References

- Jakubik, J. et al. (2024). *Prithvi-EO-2.0.* IBM/NASA geospatial foundation model.
- Ookla for Good. (2019). *Speedtest Intelligence open data.*
- Rolf, E. et al. (2021). *A generalizable and accessible approach to machine learning with global satellite imagery.* Nature Communications.
