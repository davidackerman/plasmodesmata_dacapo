#!/usr/bin/env python3
"""
Fix all k=2 HTML files for all datasets.
"""
import pickle
import os
import json
import numpy as np
import plotly.graph_objects as go
from sklearn_extra.cluster import KMedoids

# Load existing results
cache_file = "measurement_results/cell_clustering/data/cell_clustering_results.pkl"
print("Loading clustering results...")

with open(cache_file, 'rb') as f:
    results_dict = pickle.load(f)

# Extract variables
cell_ids = results_dict["cell_ids"]
cell_datasets = results_dict["datasets"]
X_scaled = results_dict["X_scaled"]
X_pca = results_dict["X_pca"]
all_results = results_dict["all_results"]

# Cluster with k=2
print("Clustering with k=2...")
km = KMedoids(n_clusters=2, metric='euclidean', random_state=42)
labels_k2 = km.fit_predict(X_scaled)

# Get unique datasets
datasets = sorted(set(cell_datasets))
print(f"Found {len(datasets)} datasets: {datasets}\n")

# Process each dataset
for dataset in datasets:
    print(f"Processing dataset: {dataset}")

    # Filter to cells from this dataset
    dataset_mask = np.array([ds == dataset for ds in cell_datasets])
    dataset_indices = np.where(dataset_mask)[0]

    if len(dataset_indices) == 0:
        print(f"  No cells found for {dataset}, skipping\n")
        continue

    print(f"  Found {len(dataset_indices)} cells")

    dataset_X_pca = X_pca[dataset_mask]
    dataset_cluster_labels = labels_k2[dataset_mask]
    dataset_cell_ids = [cell_ids[i] for i in dataset_indices]
    dataset_cell_datasets = [cell_datasets[i] for i in dataset_indices]

    # Directory setup
    figures_dir = f"measurement_results/cell_clustering/figures/k2/{dataset.split('-')[-1]}"
    projections_dir = os.path.join(figures_dir, "projections")

    # Build projection_images dict by reading the existing projections directory
    projection_images = {}
    if os.path.exists(projections_dir):
        for i, idx in enumerate(dataset_indices):
            cell_id = cell_ids[idx]
            img_filename = f"mollweide_{cell_id}.png"
            img_path = os.path.join(projections_dir, img_filename)

            if os.path.exists(img_path):
                projection_images[i] = {
                    "img_path": img_filename,
                    "cell_id": int(cell_id),
                    "dataset": dataset,
                    "cluster": int(dataset_cluster_labels[i]),
                    "pc1": float(dataset_X_pca[i, 0]),
                    "pc2": float(dataset_X_pca[i, 1]),
                }

    print(f"  Found {len(projection_images)} existing projection images")

    # Create hover text
    hover_text = []
    for i in range(len(dataset_cell_ids)):
        hover_text.append(
            f"Cell {dataset_cell_ids[i]}<br>"
            f"Dataset: {dataset_cell_datasets[i].split('-')[-1]}<br>"
            f"Cluster: {dataset_cluster_labels[i]}<br>"
            f"PC1: {dataset_X_pca[i, 0]:.2f}<br>"
            f"PC2: {dataset_X_pca[i, 1]:.2f}"
        )

    # Create Plotly figure with FIXED array serialization
    fig = go.Figure(
        data=[
            go.Scatter(
                x=dataset_X_pca[:, 0].tolist(),  # Convert to list!
                y=dataset_X_pca[:, 1].tolist(),  # Convert to list!
                mode="markers",
                marker=dict(
                    size=8,
                    color=dataset_cluster_labels.tolist(),  # Convert to list!
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Cluster"),
                    line=dict(width=1, color="DarkSlateGray"),
                ),
                text=hover_text,
                hovertemplate="<b>%{text}</b><extra></extra>",
                customdata=list(range(len(dataset_cell_ids))),
            )
        ]
    )

    fig.update_layout(
        title=f"Interactive PCA with Mollweide Projections (k=2)<br>"
        f"<sub>Hover over points to view mollweide projection</sub>",
        xaxis_title="PC1",
        yaxis_title="PC2",
        width=700,
        height=700,
        hovermode="closest",
    )

    # Convert to JSON
    fig_json = fig.to_json()

    # Create HTML
    html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Interactive PCA with Mollweide Projections</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            text-align: center;
            margin-bottom: 20px;
        }}
        #container {{
            display: flex;
            gap: 20px;
            max-width: 1600px;
            margin: 0 auto;
        }}
        #plotly-div {{
            flex: 1;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        #projection-display {{
            flex: 1;
            padding: 20px;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }}
        #projection-img {{
            max-width: 100%;
            height: auto;
            border-radius: 5px;
        }}
        .instruction {{
            color: #666;
            font-style: italic;
            margin-bottom: 10px;
            font-size: 14px;
        }}
        #projection-info {{
            margin-top: 15px;
            font-size: 16px;
            font-weight: 500;
        }}
    </style>
</head>
<body>
    <h1>Interactive PCA Visualization with Mollweide Projections (k=2)</h1>
    <div id="container">
        <div id="plotly-div"></div>
        <div id="projection-display">
            <p class="instruction">Hover over any point in the PCA plot to view its mollweide projection</p>
            <img id="projection-img" src="" alt="Mollweide projection will appear here" style="display:none;">
            <p id="projection-info"></p>
        </div>
    </div>

    <script>
        var projectionData = {json.dumps(projection_images)};

        // Create the plotly figure
        var figData = {fig_json};
        Plotly.newPlot('plotly-div', figData.data, figData.layout);

        // Add hover event listener
        var plotlyDiv = document.getElementById('plotly-div');
        plotlyDiv.on('plotly_hover', function(data) {{
            var pointIndex = data.points[0].customdata;

            if (projectionData.hasOwnProperty(pointIndex)) {{
                var projInfo = projectionData[pointIndex];
                var img = document.getElementById('projection-img');
                var info = document.getElementById('projection-info');

                img.src = 'projections/' + projInfo.img_path;
                img.style.display = 'block';
                info.innerHTML = '<b>Cell ' + projInfo.cell_id + '</b> from dataset ' +
                                projInfo.dataset.split('-').pop() +
                                ' (Cluster ' + projInfo.cluster + ')';
            }} else {{
                var info = document.getElementById('projection-info');
                info.innerHTML = '<i>No projection available for this cell.</i>';
            }}
        }});

        // Optional: Also support click for "pinning" a projection
        plotlyDiv.on('plotly_click', function(data) {{
            var pointIndex = data.points[0].customdata;

            if (projectionData.hasOwnProperty(pointIndex)) {{
                var projInfo = projectionData[pointIndex];
                var img = document.getElementById('projection-img');
                var info = document.getElementById('projection-info');

                img.src = 'projections/' + projInfo.img_path;
                img.style.display = 'block';
                info.innerHTML = '<b>Cell ' + projInfo.cell_id + '</b> from dataset ' +
                                projInfo.dataset.split('-').pop() +
                                ' (Cluster ' + projInfo.cluster + ') <i>(pinned)</i>';
            }}
        }});
    </script>
</body>
</html>
"""

    # Save HTML file
    html_path = os.path.join(figures_dir, "interactive_pca_mollweide_k2.html")
    with open(html_path, "w") as f:
        f.write(html_template)

    print(f"  Saved to {html_path}")
    print(f"  Included data for {len(projection_images)} cells\n")

print("All HTML files fixed successfully!")
