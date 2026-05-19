"""
Generate per-dataset Neuroglancer states + an index HTML page for the
plasmodesmata training data. Models the structure of
`nuclear_pores_dacapo/preprocessing/create_training_data_neuroglancer.py`.

Per dataset, the state shows:
  - raw EM at the training resolution (8 nm)
  - gt_cylinders: the rasterized cylindrical plasmodesmata annotations used
    as training labels
  - plasmodesmata: the cleaned predicted-then-postprocessed segmentation
    (where available)
  - cell_fixed: the cell-mask segmentation used downstream (where available)
  - prediction_mask: the dilated cell-mask the model is restricted to during
    inference (per-dataset dilation iterations)
  - validation_test_rois / training_validation_test_rois: in-state axis-
    aligned bounding-box annotation layers parsed from each dataset's
    training_validation_test_roi_info.yaml

Writes:
  preprocessing/training_data_neuroglancer/<dataset>/state.json
  preprocessing/training_data_neuroglancer/index.html

URL mapping:
  /nrs/cellmap/X          -> https://cellmap-vm1.int.janelia.org/nrs/X
  /groups/cellmap/cellmap/X -> https://cellmap-vm1.int.janelia.org/prfs/X
"""

import json
import os
import yaml

BASE_DIR = "/groups/cellmap/cellmap/ackermand/Programming/plasmodesmata_dacapo"
ANNOTATIONS_DIR = os.path.join(BASE_DIR, "preprocessing/annotations")

NG_OUTPUT_DIR = os.path.join(BASE_DIR, "preprocessing/training_data_neuroglancer")
GT_CYLINDERS_ZARR = (
    "/nrs/cellmap/ackermand/cellmap/plasmodesmata/annotations_as_cylinders.zarr"
)
LEAFGALL_ZARR_BASE = "/nrs/cellmap/ackermand/cellmap/leaf-gall"
PREDICTION_MASKS_ZARR = f"{LEAFGALL_ZARR_BASE}/prediction_masks.zarr"

VM_URL = "https://cellmap-vm1.int.janelia.org"
NG_VIEWER = "https://neuroglancer-demo.appspot.com"

TRAIN_RES_NM = 8.0

# Per-dataset raw EM container + array scale used by the training/inference
# pipeline. Mirrors the prediction YAMLs in whole_datasets/prediction_yamls/.
# `scale` is the multiscale level inside the container that lands at 8 nm.
DATASETS = {
    "jrc_22ak351-leaf-2l":  {"scale": "s0", "dilation_iterations": 5},
    "jrc_22ak351-leaf-2lb": {"scale": "s1", "dilation_iterations": 8},
    "jrc_22ak351-leaf-3m":  {"scale": "s0", "dilation_iterations": 4},
    "jrc_22ak351-leaf-3mb": {"scale": "s1", "dilation_iterations": 8},
    "jrc_22ak351-leaf-3r":  {"scale": "s0", "dilation_iterations": 5},
    "jrc_22ak351-leaf-3rb": {"scale": "s1", "dilation_iterations": 8},
}


def nrs_to_url(path):
    return path.replace("/nrs/cellmap/", f"{VM_URL}/nrs/")


def groups_to_url(path):
    return path.replace("/groups/cellmap/cellmap/", f"{VM_URL}/prfs/")


def make_grayscale_shader(cmin=0, cmax=255):
    return (
        f"#uicontrol invlerp normalized(range=[{cmin}, {cmax}], "
        f"window=[{cmin}, {cmax}])\n"
        f"void main() {{ emitGrayscale(normalized()); }}"
    )


def parse_range(s):
    a, b = (int(x) for x in str(s).split("-"))
    return (min(a, b), max(a, b))


def roi_annotation_layer(name, rois, color):
    """Build an inline NG annotation layer of axis-aligned bounding boxes
    from a list of {roi_name, x, y, z} dicts. Coordinates are in nm in the
    YAML; NG voxel size is in m so we divide by 1e-9 * TRAIN_RES_NM."""
    if not rois:
        return None
    annotations = []
    for roi in rois:
        xa, xb = parse_range(roi["x"])
        ya, yb = parse_range(roi["y"])
        za, zb = parse_range(roi["z"])
        # convert nm -> voxels at training resolution
        annotations.append({
            "type": "axis_aligned_bounding_box",
            "id": str(roi.get("roi_name", len(annotations))),
            "pointA": [xa / TRAIN_RES_NM, ya / TRAIN_RES_NM, za / TRAIN_RES_NM],
            "pointB": [xb / TRAIN_RES_NM, yb / TRAIN_RES_NM, zb / TRAIN_RES_NM],
            "description": str(roi.get("roi_name", "")),
        })
    return {
        "type": "annotation",
        "name": name,
        "annotations": annotations,
        "annotationColor": color,
        "visible": True,
    }


def load_rois(dataset):
    yaml_path = os.path.join(
        ANNOTATIONS_DIR, dataset, "training_validation_test_roi_info.yaml"
    )
    if not os.path.isfile(yaml_path):
        return None
    with open(yaml_path) as f:
        return yaml.safe_load(f)


def make_state(dataset, dilation_iterations, raw_scale):
    layers = []

    raw_path = f"/nrs/cellmap/data/{dataset}/{dataset}.zarr/recon-1/em/fibsem-uint8"
    layers.append({
        "type": "image",
        "source": [{"url": f"zarr://{nrs_to_url(raw_path)}/{raw_scale}"}],
        "name": "raw",
        "shader": make_grayscale_shader(),
    })

    gt_path = f"{GT_CYLINDERS_ZARR}/{dataset}/s0"
    if os.path.isdir(gt_path):
        layers.append({
            "type": "segmentation",
            "source": [{"url": f"zarr://{nrs_to_url(gt_path)}"}],
            "name": "gt_cylinders",
        })

    plasmodesmata_path = f"{LEAFGALL_ZARR_BASE}/{dataset}.zarr/plasmodesmata_cleaned/s0"
    if os.path.isdir(plasmodesmata_path):
        layers.append({
            "type": "segmentation",
            "source": [{"url": f"zarr://{nrs_to_url(plasmodesmata_path)}"}],
            "name": "plasmodesmata_cleaned",
            "visible": False,
        })

    cell_path = f"{LEAFGALL_ZARR_BASE}/{dataset}.zarr/cell_fixed/s0"
    if os.path.isdir(cell_path):
        layers.append({
            "type": "segmentation",
            "source": [{"url": f"zarr://{nrs_to_url(cell_path)}"}],
            "name": "cell_fixed",
            "visible": False,
        })

    mask_path = (
        f"{PREDICTION_MASKS_ZARR}/dilation_iterations_{dilation_iterations}_{dataset}/s0"
    )
    if os.path.isdir(mask_path):
        layers.append({
            "type": "segmentation",
            "source": [{"url": f"zarr://{nrs_to_url(mask_path)}"}],
            "name": f"prediction_mask_dilation_{dilation_iterations}",
            "visible": False,
        })

    rois = load_rois(dataset)
    if rois:
        rois_to_split = rois.get("rois_to_split", {})
        vt_layer = roi_annotation_layer(
            "validation_test_rois",
            rois_to_split.get("validation_test", []),
            "#ff4444",
        )
        if vt_layer:
            layers.append(vt_layer)
        tvt_layer = roi_annotation_layer(
            "training_validation_test_rois",
            rois_to_split.get("training_validation_test", []),
            "#44b0ff",
        )
        if tvt_layer:
            layers.append(tvt_layer)

    return {
        "layers": layers,
        "dimensions": {
            "z": [TRAIN_RES_NM * 1e-9, "m"],
            "y": [TRAIN_RES_NM * 1e-9, "m"],
            "x": [TRAIN_RES_NM * 1e-9, "m"],
        },
        "showSlices": False,
    }


def count_annotation_csvs(dataset):
    d = os.path.join(ANNOTATIONS_DIR, dataset)
    if not os.path.isdir(d):
        return 0
    return sum(
        1
        for f in os.listdir(d)
        if f.startswith("annotations_") and f.endswith(".csv")
    )


def count_rois(dataset):
    rois = load_rois(dataset)
    if not rois:
        return (0, 0)
    rts = rois.get("rois_to_split", {})
    return (
        len(rts.get("validation_test", [])),
        len(rts.get("training_validation_test", [])),
    )


def make_html(datasets):
    rows = ""
    for ds in datasets:
        cfg = DATASETS[ds]
        state_url = groups_to_url(f"{NG_OUTPUT_DIR}/{ds}/state.json")
        ng_url = f"{NG_VIEWER}/#!{state_url}"

        layers_present = ["raw"]
        if os.path.isdir(f"{GT_CYLINDERS_ZARR}/{ds}/s0"):
            layers_present.append("gt")
        if os.path.isdir(f"{LEAFGALL_ZARR_BASE}/{ds}.zarr/plasmodesmata_cleaned/s0"):
            layers_present.append("plasmodesmata")
        if os.path.isdir(f"{LEAFGALL_ZARR_BASE}/{ds}.zarr/cell_fixed/s0"):
            layers_present.append("cell")
        if os.path.isdir(
            f"{PREDICTION_MASKS_ZARR}/dilation_iterations_{cfg['dilation_iterations']}_{ds}/s0"
        ):
            layers_present.append(f"mask(dil={cfg['dilation_iterations']})")
        n_vt, n_tvt = count_rois(ds)
        if n_vt or n_tvt:
            layers_present.append(f"rois(vt={n_vt},tvt={n_tvt})")
        layers_str = ", ".join(layers_present)

        n_csvs = count_annotation_csvs(ds)
        annot_dir = os.path.join(ANNOTATIONS_DIR, ds)
        annot_dir_url = groups_to_url(annot_dir)
        annot_cell = (
            f'<a href="{annot_dir_url}" target="_blank">{n_csvs} CSV(s)</a>'
            if n_csvs else '<span class="no-csv">n/a</span>'
        )

        gt_path = f"{GT_CYLINDERS_ZARR}/{ds}/s0"
        gt_cell = (
            f'<div class="path-cell" title="{gt_path}">{gt_path}</div>'
            if os.path.isdir(gt_path)
            else '<span class="no-csv">n/a</span>'
        )

        rows += f"""
            <tr>
                <td class="dataset-cell"><a href="{ng_url}" target="_blank">{ds}</a></td>
                <td class="layers-cell">{layers_str}</td>
                <td class="dl-cell">{annot_cell}</td>
                <td class="dl-cell">{gt_cell}</td>
            </tr>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Plasmodesmata - Training Data Neuroglancer Links</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #2d8f5f 0%, #6dd58c 100%);
            min-height: 100vh;
            padding: 32px 20px;
            margin: 0;
            color: #2d3748;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        header {{
            background: white;
            border-radius: 12px;
            padding: 32px 36px;
            margin-bottom: 24px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.15);
        }}
        h1 {{
            color: #1a202c;
            margin: 0 0 8px 0;
            font-size: 1.9em;
            letter-spacing: -0.5px;
        }}
        .info {{ color: #718096; font-size: 0.95em; line-height: 1.5; }}
        .legend {{
            color: #4a5568;
            font-size: 0.9em;
            line-height: 1.6;
            padding-left: 22px;
            margin: 8px 0 16px 0;
        }}
        .legend code {{
            background: #edf2f7;
            padding: 1px 6px;
            border-radius: 3px;
            font-family: 'SF Mono', Monaco, Consolas, monospace;
            font-size: 0.92em;
            color: #2d3748;
        }}
        table {{
            width: 100%;
            background: white;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.15);
            border-collapse: collapse;
            overflow: hidden;
        }}
        th {{
            background: #f7fafc;
            padding: 14px 24px;
            text-align: left;
            font-weight: 600;
            color: #4a5568;
            font-size: 0.85em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            border-bottom: 2px solid #edf2f7;
            border-right: 1px solid #e2e8f0;
        }}
        th:last-child {{ border-right: none; }}
        td {{
            padding: 16px 24px;
            border-top: 1px solid #edf2f7;
            border-right: 1px solid #edf2f7;
            vertical-align: middle;
        }}
        td:last-child {{ border-right: none; }}
        tr:hover td {{ background: #f7fafc; }}
        a {{ color: #2d8f5f; text-decoration: none; font-weight: 500; }}
        a:hover {{ text-decoration: underline; }}
        .dataset-cell a {{ font-size: 1.05em; font-weight: 600; color: #1e6a44; }}
        .layers-cell {{
            font-family: 'SF Mono', Monaco, Consolas, monospace;
            font-size: 0.82em;
            color: #4a5568;
        }}
        .dl-cell {{
            min-width: 280px;
            font-family: 'SF Mono', Monaco, Consolas, monospace;
            font-size: 0.85em;
        }}
        .path-cell {{
            font-size: 0.78em;
            color: #4a5568;
            word-break: break-all;
            line-height: 1.4;
            padding: 4px 8px;
            background: #f7fafc;
            border-radius: 4px;
            border: 1px solid #e2e8f0;
        }}
        .no-csv {{
            color: #cbd5e0;
            font-style: italic;
            font-size: 0.85em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Plasmodesmata - Training Data</h1>
            <p class="info">Per-dataset Neuroglancer links and source pointers for the leaf-gall plasmodesmata training data. Six datasets, all viewed at the 8 nm training grid (the 4 nm <code>b</code> volumes are viewed via their <code>s1</code> level).</p>
            <ul class="legend">
                <li><code>raw</code> — EM fed to the model: <code>recon-1/em/fibsem-uint8/s0</code> for 8 nm datasets, <code>/s1</code> for the 4 nm <code>b</code> variants.</li>
                <li><code>gt_cylinders</code> — rasterized cylindrical ground-truth: each annotation line drawn as a cylinder. This is what the model was trained against. Source: <code>annotations_as_cylinders.zarr/&lt;dataset&gt;</code>.</li>
                <li><code>plasmodesmata_cleaned</code> — post-MWS / postprocessed plasmodesmata instance segmentation (8 nm datasets only).</li>
                <li><code>cell_fixed</code> — proofread cell-mask segmentation used for downstream geodesic analysis (8 nm datasets only).</li>
                <li><code>prediction_mask_dilation_N</code> — dilated cell mask restricting model inference. <code>N</code> is the dilation-iterations setting used in <code>whole_datasets/prediction_yamls/2025-09-15_&lt;dataset&gt;.yaml</code>.</li>
                <li><code>validation_test_rois</code> / <code>training_validation_test_rois</code> — ROI bounding boxes pulled from each dataset's <code>training_validation_test_roi_info.yaml</code>.</li>
            </ul>
        </header>
        <table>
            <thead>
                <tr>
                    <th>Neuroglancer link</th>
                    <th>Layers present</th>
                    <th>Annotation CSVs</th>
                    <th>GT cylinders zarr</th>
                </tr>
            </thead>
            <tbody>{rows}
            </tbody>
        </table>
    </div>
</body>
</html>
"""


def main():
    os.makedirs(NG_OUTPUT_DIR, exist_ok=True)
    datasets = list(DATASETS.keys())

    for ds in datasets:
        cfg = DATASETS[ds]
        ds_dir = os.path.join(NG_OUTPUT_DIR, ds)
        os.makedirs(ds_dir, exist_ok=True)
        state = make_state(ds, cfg["dilation_iterations"], cfg["scale"])
        with open(os.path.join(ds_dir, "state.json"), "w") as f:
            json.dump(state, f, indent=2)
        layer_names = [layer["name"] for layer in state["layers"]]
        print(f"  {ds}: {layer_names}")

    html = make_html(datasets)
    with open(os.path.join(NG_OUTPUT_DIR, "index.html"), "w") as f:
        f.write(html)

    index_url = groups_to_url(f"{NG_OUTPUT_DIR}/index.html")
    print(f"\nIndex page: {index_url}")
    print(f"Generated links for {len(datasets)} datasets.")


if __name__ == "__main__":
    main()
