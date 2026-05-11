# plasmodesmata_dacapo

End-to-end pipeline for predicting plasmodesmata in FIB-SEM leaf-gall datasets, post-processing them into instance segmentations, and analyzing their spatial distribution on cell surfaces.

Datasets: `jrc_22ak351-leaf-2l`, `2lb`, `3m`, `3mb`, `3r`, `3rb`.

## Pipeline overview

1. **Preprocessing** — annotations, cell-segmentation masks, ROI splits.
2. **Training / validation / test** — DaCapo runs scored on validation crops.
3. **Whole-dataset prediction** — affinities + LSDs predicted on full volumes.
4. **Mutex watershed (MWS)** — affinities turned into instance segmentations.
5. **Postprocessing** — alignment, relabeling, and `cellmap-analyze` measurements.
6. **Analysis** — mesh remeshing, geodesic distances, clustering, mollweide projections.

Each stage's working directory is below. Outputs live under `/nrs/cellmap/ackermand/...` and are not tracked in this repo.

---

## 1. Preprocessing — [preprocessing/](preprocessing/)

Trained on 8 nm; evaluated on the native 4 nm downsampled to 8 nm.

Per-dataset YAMLs live in [preprocessing/annotations/processing_yamls/](preprocessing/annotations/processing_yamls/) and are run via `annotation-processing-utils` cluster submission:

```
python /groups/cellmap/cellmap/ackermand/Programming/annotation-processing-utils/annotation_processing_utils/cli/cluster_submission.py
```

Other helpers in this dir:
- `create_masks.py`, `fix_cell_segmentations.py`, `relabel_cell_masks_for_annotating.py` — mask preparation.
- `image_data_interface.py` — tensorstore-backed zarr/n5 reader.
- Per-dataset `*.py` and `*.yaml` driver scripts.

## 2. Training / validation / test — [validation_and_test/](validation_and_test/)

DaCapo training + inference + scoring:
- `run_validation_inference.py` — runs validation inference across a range of runs/regions.
- `predict_with_write_size.py` — prediction with a specified write size (the default did not work).
- `get_best.py` — picks best iterations per run.
- `yamls/` — combined metric YAMLs per run/dataset.

## 3. Whole-dataset prediction — [whole_datasets/prediction_yamls/](whole_datasets/prediction_yamls/)

The actual inference is launched from the `cellmap_experiments` env in `ml_experiments`, not from this repo — this dir only stores the configs and submission commands.

```
bsub -P cellmap -n 4 -o .../predictions/<date>_<dataset>.out python \
  /groups/scicompsoft/home/ackermand/Programming/ml_experiments/scripts/submit.py \
  predict -p <date>_<dataset>.yaml -w 100
```

Two generations of configs:
- `2025-02-15_*.yaml` — first round (4 nm `b` datasets and matching).
- `2025-09-15_*.yaml` — second round on 8 nm datasets, routed to `gpu_h200`.

Submission commands recorded in [whole_datasets/prediction_yamls/submissions.md](whole_datasets/prediction_yamls/submissions.md).

## 4. Mutex watershed — [whole_datasets/mws/](whole_datasets/mws/)

Per-dataset `rusty_mws.PostProcessor` driver scripts. Activate the `rusty_mws` conda env (it requires numpy 2, kept separate from the dacapo env). Submission commands in [whole_datasets/mws/submissions.md](whole_datasets/mws/submissions.md).

## 5. Postprocessing — [postprocessing/](postprocessing/)

Segmentation cleanup and measurement scaffolding:
- `align_segmentations.py` / `align_segmentations.ipynb` — align MWS output to reference frames.
- `zarr_util.py` — multiscale zarr helpers used by preprocessing too.
- `utils.py` / `utils/` — rotations, ROI helpers, etc.
- `match_connecting_cells.ipynb`, `fit_lines_to_segmentations.ipynb`, `cylinder_fitting.ipynb`, `evaluate_best_run_metrics.ipynb` — measurement and evaluation notebooks.
- `postprocessing.ipynb` — top-level driver notebook.

Per-cell and per-plasmodesma measurements (cell COMs, plasmodesmata COMs, two-nearest-cell assignment) are produced via `cellmap-analyze`'s connected-components/watershed pipeline; results land under `/nrs/cellmap/ackermand/cellmap/analysisResults/leaf-gall/<dataset>/`.

## 6. Analysis — [analysis/](analysis/)

Spatial-distribution analysis on top of the per-cell measurements:

**Mesh remeshing & geodesic distances** (see [analysis/README.md](analysis/README.md)):
- `remesh.py` — legacy incremental insertion (deprecated).
- `remesh_new.py` — batch insertion (4-pass: project, allocate edges, allocate interior, retriangulate).
- `remesh_and_measure_dask.py` — **production**: batch insertion + Dask-parallel geodesic distance matrices, with mesh-cleaning robustness (`clean_mesh_for_geodesic`) that handles degenerate-face failures seen on cells like #84.
- `parallel_remesh.py`, `parallel_remesh_flexible.py` — earlier parallel wrappers.

**Clustering & visualization:**
- `cluster_cells_by_pattern.py` — PCA + KMeans on per-cell geodesic distance distributions; produces mollweide projections and interactive Plotly HTMLs (PNGs written externally to keep the HTML small).
- `measure_clustering.py` — measurement-driven clustering primitives, importable from notebooks.
- `weighted_graph_analysis.py` — graph analysis on cell-connection weights.
- `visualize_geodesic_distance.py` — interactive geodesic-distance viewer.
- `cell_assignment_plots.py` / `.ipynb` — per-cell plasmodesmata assignment diagnostics.
- `generate_neuroglancer_annotations_ngsidekick.py` — Neuroglancer links with cell centers + plasmodesmata connections.

**Top-level helpers:**
- [image_similarity.py](image_similarity.py) — CLIP-based + radial-average image similarity for plasmodesmata cube comparison.
- [move_processed_data_for_new_splitting.py](move_processed_data_for_new_splitting.py) — moves validation outputs into test paths when ROI splits change (move calls are commented out; uncomment when running).

## Environments

- **`plasmodesmata_dacapo`** — preprocessing, training, validation, analysis (this repo's `pyproject.toml`).
- **`cellmap_experiments`** — used for the actual whole-dataset prediction launches from `ml_experiments`.
- **`rusty_mws`** — mutex watershed (needs numpy 2; kept separate so it doesn't conflict with the dacapo dask stack).
- **`annotation_processing_utils_mws`** — used when running MWS via `annotation_processing_utils`.

Extra pinned constraint for the analysis env in [installation_analysis_constraints.txt](installation_analysis_constraints.txt) (`pyacvd==0.2.10`).
