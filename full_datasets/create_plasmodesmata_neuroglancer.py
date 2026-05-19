"""
Generate per-dataset Neuroglancer states + an index HTML page for the
plasmodesmata final whole-dataset results. Models the structure of
`nuclear_pores_dacapo/full_datasets/create_np_original_resolution_neuroglancer.py`.

Per dataset, the state shows:
  - raw EM (multiscale parent group)
  - plasmodesmata: post-MWS / postprocessed instance segmentation. Resolved
    from /nrs/cellmap/data/<dataset>/<dataset>.zarr/recon-1/labels/inference/
    segmentations/pd when present (the OpenOrganelle-style final location;
    currently the b datasets), falling back to leaf-gall/<dataset>.zarr/
    plasmodesmata_cleaned for the non-b datasets that haven't been moved
    yet.
  - cell_fixed: proofread cell-mask segmentation + precomputed multires meshes
    (the segmentation zarr and the precomputed mesh dir are bundled in the
    same layer via two sources)

The HTML index lists each dataset with download links for the measurement
CSVs produced by cellmap-analyze:
  cell_fixed.csv, plasmodesmata.csv, plasmodesmata_lines.csv,
  plasmodesmata_lines_assigned_to_2_nearest_cells.csv

Writes:
  full_datasets/plasmodesmata_neuroglancer/<dataset>/state.json
  full_datasets/plasmodesmata_neuroglancer/index.html

URL mapping:
  /nrs/cellmap/X          -> https://cellmap-vm1.int.janelia.org/nrs/X
  /groups/cellmap/cellmap/X -> https://cellmap-vm1.int.janelia.org/prfs/X
"""

import json
import os

BASE_DIR = "/groups/cellmap/cellmap/ackermand/Programming/plasmodesmata_dacapo"
NG_OUTPUT_DIR = os.path.join(BASE_DIR, "full_datasets/plasmodesmata_neuroglancer")

LEAFGALL_ZARR_BASE = "/nrs/cellmap/ackermand/cellmap/leaf-gall"
MEASURE_BASE = "/nrs/cellmap/ackermand/cellmap/analysisResults/leaf-gall"
MESH_BASE = "/nrs/cellmap/ackermand/new_meshes/meshes/single_resolution/leaf-gall"

VM_URL = "https://cellmap-vm1.int.janelia.org"
NG_VIEWER = "https://neuroglancer-demo.appspot.com"

DATASETS = [
    "jrc_22ak351-leaf-2l",
    "jrc_22ak351-leaf-2lb",
    "jrc_22ak351-leaf-3m",
    "jrc_22ak351-leaf-3mb",
    "jrc_22ak351-leaf-3r",
    "jrc_22ak351-leaf-3rb",
]

CSV_NAMES = [
    "cell_fixed.csv",
    "plasmodesmata.csv",
    "plasmodesmata_lines.csv",
    "plasmodesmata_lines_assigned_to_2_nearest_cells.csv",
]


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


def find_plasmodesmata_zarr(dataset):
    """Return the multiscale-parent path to the plasmodesmata segmentation,
    or None. Prefers the OpenOrganelle-style final location (used currently
    for the b datasets) and falls back to the leaf-gall working path used
    by the 8 nm datasets."""
    candidates = [
        f"/nrs/cellmap/data/{dataset}/{dataset}.zarr/recon-1/labels/inference/segmentations/pd",
        f"{LEAFGALL_ZARR_BASE}/{dataset}.zarr/plasmodesmata_cleaned",
    ]
    for p in candidates:
        if os.path.isdir(p):
            return p
    return None


def find_cell_zarr(dataset):
    """Same pattern for the cell segmentation: prefer the OO final location
    (segmentations/cell), fall back to leaf-gall/<dataset>.zarr/cell_fixed."""
    candidates = [
        f"/nrs/cellmap/data/{dataset}/{dataset}.zarr/recon-1/labels/inference/segmentations/cell",
        f"{LEAFGALL_ZARR_BASE}/{dataset}.zarr/cell_fixed",
    ]
    for p in candidates:
        if os.path.isdir(p):
            return p
    return None


def make_state(dataset):
    layers = []

    raw_path = f"/nrs/cellmap/data/{dataset}/{dataset}.zarr/recon-1/em/fibsem-uint8"
    layers.append({
        "type": "image",
        "source": [{"url": f"zarr://{nrs_to_url(raw_path)}"}],
        "name": "raw",
        "shader": make_grayscale_shader(),
    })

    plasmodesmata_path = find_plasmodesmata_zarr(dataset)
    if plasmodesmata_path:
        layers.append({
            "type": "segmentation",
            "source": [{"url": f"zarr://{nrs_to_url(plasmodesmata_path)}"}],
            "name": "plasmodesmata",
        })

    cell_path = find_cell_zarr(dataset)
    cell_mesh_path = f"{MESH_BASE}/{dataset}/cell_fixed_neuroglancer/meshes"
    if cell_path:
        cell_sources = [{"url": f"zarr://{nrs_to_url(cell_path)}"}]
        if os.path.isdir(cell_mesh_path):
            cell_sources.append({"url": f"precomputed://{nrs_to_url(cell_mesh_path)}"})
        layers.append({
            "type": "segmentation",
            "source": cell_sources,
            "name": "cell",
            "visible": False,
        })

    return {
        "layers": layers,
        "dimensions": {
            "z": [8e-9, "m"],
            "y": [8e-9, "m"],
            "x": [8e-9, "m"],
        },
        "showSlices": False,
    }


def csv_link(dataset, name):
    path = f"{MEASURE_BASE}/{dataset}/{name}"
    if not os.path.isfile(path):
        return '<span class="no-csv">n/a</span>'
    url = nrs_to_url(path)
    return f'<a href="{url}" download class="download-btn">⬇ {name}</a>'


def make_html(datasets):
    rows = ""
    for ds in datasets:
        state_url = groups_to_url(f"{NG_OUTPUT_DIR}/{ds}/state.json")
        ng_url = f"{NG_VIEWER}/#!{state_url}"

        layers_present = ["raw"]
        if find_plasmodesmata_zarr(ds):
            layers_present.append("plasmodesmata")
        if find_cell_zarr(ds):
            cell_chip = "cell"
            if os.path.isdir(f"{MESH_BASE}/{ds}/cell_fixed_neuroglancer/meshes"):
                cell_chip += "+meshes"
            layers_present.append(cell_chip)
        layers_str = ", ".join(layers_present)

        csvs_html = " ".join(csv_link(ds, name) for name in CSV_NAMES)

        rows += f"""
            <tr>
                <td class="dataset-cell"><a href="{ng_url}" target="_blank">{ds}</a></td>
                <td class="layers-cell">{layers_str}</td>
                <td class="csv-cell">{csvs_html}</td>
            </tr>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Plasmodesmata - Whole-Dataset Results</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1d6f4a 0%, #4bbf86 100%);
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
        a {{ color: #1d6f4a; text-decoration: none; font-weight: 500; }}
        a:hover {{ text-decoration: underline; }}
        .dataset-cell a {{ font-size: 1.05em; font-weight: 600; color: #154f33; }}
        .layers-cell {{
            font-family: 'SF Mono', Monaco, Consolas, monospace;
            font-size: 0.82em;
            color: #4a5568;
        }}
        .csv-cell {{
            min-width: 360px;
            line-height: 1.9;
        }}
        .download-btn {{
            display: inline-block;
            padding: 4px 10px;
            background: #1d6f4a;
            color: white !important;
            border-radius: 5px;
            font-size: 0.78em;
            font-weight: 500;
            text-decoration: none !important;
            margin-right: 4px;
            transition: background 0.15s ease;
            white-space: nowrap;
        }}
        .download-btn:hover {{ background: #154f33; }}
        .no-csv {{
            color: #cbd5e0;
            font-style: italic;
            font-size: 0.78em;
            margin-right: 4px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Plasmodesmata - Whole-Dataset Results</h1>
            <p class="info">Per-dataset Neuroglancer states overlaying the final plasmodesmata and cell segmentations on raw EM, plus download links to the per-dataset measurement CSVs produced by <code>cellmap-analyze</code>. Modeled on <code>nuclear_pores_dacapo/full_datasets/create_np_original_resolution_neuroglancer.py</code>.</p>
            <ul class="legend">
                <li><code>raw</code> — EM, multiscale parent group.</li>
                <li><code>plasmodesmata</code> — post-MWS / postprocessed plasmodesmata instance segmentation. Resolved from <code>/nrs/cellmap/data/&lt;dataset&gt;/&lt;dataset&gt;.zarr/recon-1/labels/inference/segmentations/pd</code> when present (OpenOrganelle-style final location, currently the b datasets); otherwise from <code>leaf-gall/&lt;dataset&gt;.zarr/plasmodesmata_cleaned</code> (the non-b datasets that haven't been moved yet).</li>
                <li><code>cell</code> — proofread cell-mask segmentation. Resolved from the OpenOrganelle path <code>/nrs/cellmap/data/&lt;dataset&gt;/&lt;dataset&gt;.zarr/recon-1/labels/inference/segmentations/cell</code> when present, falling back to <code>leaf-gall/&lt;dataset&gt;.zarr/cell_fixed</code>. When precomputed multires meshes are available (8 nm datasets only, at <code>new_meshes/.../cell_fixed_neuroglancer/meshes</code>) they are bundled into the same layer via a second source.</li>
            </ul>
            <p class="info"><strong>Downloads.</strong> Each row's CSVs come from <code>/nrs/cellmap/ackermand/cellmap/analysisResults/leaf-gall/&lt;dataset&gt;/</code>. <code>cell_fixed.csv</code> + <code>plasmodesmata_lines_assigned_to_2_nearest_cells.csv</code> are only produced for the 8 nm datasets (cell-aware analysis).</p>
        </header>
        <table>
            <thead>
                <tr>
                    <th>Neuroglancer link</th>
                    <th>Layers present</th>
                    <th>Measurement CSVs</th>
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

    for ds in DATASETS:
        ds_dir = os.path.join(NG_OUTPUT_DIR, ds)
        os.makedirs(ds_dir, exist_ok=True)
        state = make_state(ds)
        with open(os.path.join(ds_dir, "state.json"), "w") as f:
            json.dump(state, f, indent=2)
        layer_names = [layer["name"] for layer in state["layers"]]
        print(f"  {ds}: {layer_names}")

    html = make_html(DATASETS)
    with open(os.path.join(NG_OUTPUT_DIR, "index.html"), "w") as f:
        f.write(html)

    index_url = groups_to_url(f"{NG_OUTPUT_DIR}/index.html")
    print(f"\nIndex page: {index_url}")
    print(f"Generated links for {len(DATASETS)} datasets.")


if __name__ == "__main__":
    main()
