# Interactive Mollweide Projection Visualization

## Overview

The `cluster_cells_by_pattern.py` script now includes interactive HTML visualizations that combine PCA clustering with mollweide projections of plasmodesmata distributions on cell surfaces.

## What's New

### 1. **Mollweide Projections**
Each cell's plasmodesmata are projected onto a 2D mollweide map by:
- Computing the center of mass (COM) of the cell mesh
- Converting plasmodesmata positions to spherical coordinates relative to the COM
- Mapping to longitude/latitude coordinates
- Displaying on a mollweide projection (commonly used for showing entire spheres)

### 2. **Interactive PCA Visualization**
For each k value (2, 3, 4), the script generates:
- **Overall visualization**: `interactive_pca_mollweide_k{k}.html`
  - Shows all cells in PCA space colored by cluster
  - Generates mollweide projections for **ALL** cells
  - **Hover** over any point to see its mollweide projection in the right panel
  - **Click** to "pin" a projection (keeps it displayed while you explore other cells)
  - Side-by-side layout: PCA plot on left, mollweide projection on right

- **Per-dataset visualizations**: `{dataset}/interactive_pca_mollweide_k{k}.html`
  - Same as above but filtered to one dataset at a time
  - Useful for seeing dataset-specific patterns

## How to Use

1. **Run the clustering script**:
   ```bash
   cd /groups/cellmap/cellmap/ackermand/Programming/plasmodesmata_dacapo/analysis
   python cluster_cells_by_pattern.py
   ```

2. **Open the HTML files**:
   - Overall: `measurement_results/cell_clustering/figures/k2/interactive_pca_mollweide_k2.html`
   - Per dataset: `measurement_results/cell_clustering/figures/k2/2l/interactive_pca_mollweide_k2.html`

3. **Interact with the visualization**:
   - **Hover** over any point to instantly see its mollweide projection in the right panel
   - **Click** to pin a projection (useful for comparing multiple cells)
   - Hover also shows cell ID, dataset, cluster, and PCA coordinates
   - The projection shows the spatial distribution of plasmodesmata on the cell surface

## What the Mollweide Projection Shows

The mollweide projection treats each cell as approximately spherical and shows:
- **Red dots**: Individual plasmodesmata locations
- **Spatial patterns**: Clustering, uniform distribution, or other patterns
- **Global view**: The entire cell surface "unwrapped" into 2D

### Interpreting Patterns
- **Clustered**: Plasmodesmata concentrated in certain regions
- **Uniform**: Even distribution across the surface
- **Banded/Striped**: Organized in latitudinal or longitudinal patterns
- **Polar concentration**: Higher density near top/bottom of the projection

## Projection Generation

The script now generates mollweide projections for **ALL** cells in the dataset:
- **Parallel processing** - Uses Dask to generate projections in parallel for much faster performance
- **External PNG files** - Images are saved as separate PNG files (not embedded in HTML)
- **Fast loading** - HTML files are small (<1 MB), images load on-demand
- **Offline viewing** - All files are local, no internet connection needed
- No sampling needed - every cell gets a projection

## Files Generated

For each k value (2, 3, 4):
```
measurement_results/cell_clustering/figures/
├── k2/
│   ├── interactive_pca_mollweide_k2.html       # All datasets combined
│   ├── projections/                             # PNG images for all cells
│   │   ├── mollweide_0.png
│   │   ├── mollweide_1.png
│   │   └── ... (one per cell)
│   ├── 2l/
│   │   ├── interactive_pca_mollweide_k2.html   # leaf-2l only
│   │   └── projections/                         # PNG images for 2l cells
│   ├── 3r/
│   │   ├── interactive_pca_mollweide_k2.html   # leaf-3r only
│   │   └── projections/                         # PNG images for 3r cells
│   └── 3m/
│       ├── interactive_pca_mollweide_k2.html   # leaf-3m only
│       └── projections/                         # PNG images for 3m cells
├── k3/ (same structure)
└── k4/ (same structure)
```

## Technical Details

### Data Sources
- **Mesh data**: `/nrs/cellmap/ackermand/new_meshes/meshes/single_resolution/leaf-gall/{dataset}/cell_fixed/meshes/{cell_id}.ply`
- **Plasmodesmata positions**: `/nrs/cellmap/ackermand/cellmap/analysisResults/leaf-gall/{dataset}/geodesic_distances/{cell_id}_distribution.pkl`
  - Uses `updated_vertices` and `plasmodesmata_indices` from the pickle file

### Coordinate Transformation
```python
# Center around COM
centered_pd = plasmodesmata_positions - com

# Spherical coordinates
r = norm(centered_pd)
theta = arctan2(y, x)      # azimuth [0, 2π]
phi = arccos(z/r)          # polar angle from z-axis [0, π]

# Mollweide coordinates
longitude = theta           # [-π, π]
latitude = π/2 - phi       # [-π/2, π/2]
```

### HTML Structure
Each interactive visualization includes:
- Plotly.js for interactive PCA scatter plot
- Base64-encoded PNG images of mollweide projections
- JavaScript click handlers to update displayed projection
- Responsive layout with instructions

## Future Enhancements

Potential improvements:
1. Add 3D mesh visualization alongside mollweide projection
2. Show multiple projections in a grid for comparison
3. Animate transitions between clicked cells
4. Add density heatmaps on the mollweide projection
5. Include geodesic distance distributions for clicked cells
6. Export clicked cell data for further analysis

## Troubleshooting

**Problem**: Projections don't appear on hover
- **Solution**: Make sure you're using a modern browser (Chrome, Firefox, Safari). Check that the `projections/` folder is in the same directory as the HTML file.

**Problem**: Images take time to load on first hover
- **Solution**: This is normal - images load on-demand. Subsequent hovers on the same cell will be instant (browser caching).

**Problem**: Want higher/lower resolution images
- **Solution**: Edit line 327 to change DPI: `fig.savefig(png_path, format='png', dpi=100, ...)` - try 75 for smaller files or 150 for higher quality.

**Problem**: Need to move visualization to another location
- **Solution**: Copy both the HTML file AND the `projections/` folder together. They must be in the same directory.
