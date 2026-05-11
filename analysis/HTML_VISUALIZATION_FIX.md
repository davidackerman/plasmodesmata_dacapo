# Interactive PCA Mollweide Visualization - Bug Fix

## Issue

The interactive HTML visualizations (`interactive_pca_mollweide_k*.html`) were displaying empty scatter plots with no visible data points, despite having projection images available.

## Root Cause

The bug was in the [cluster_cells_by_pattern.py](cluster_cells_by_pattern.py) script at lines 426-427 and 431. When creating the Plotly scatter plot, NumPy arrays were passed directly to the `go.Scatter()` constructor:

```python
fig = go.Figure(
    data=[
        go.Scatter(
            x=X_pca[:, 0],              # NumPy array
            y=X_pca[:, 1],              # NumPy array
            marker=dict(
                color=cluster_labels,    # NumPy array
                ...
            ),
            ...
        )
    ]
)
```

When these arrays were serialized to JSON using `fig.to_json()`, they were improperly serialized as objects with internal structure (`dtype`, `bdata` fields) rather than plain JavaScript arrays of numbers. This caused the HTML page to render only axis/grid but no actual data points.

## Fix Applied

The fix converts NumPy arrays to Python lists before passing them to Plotly:

```python
fig = go.Figure(
    data=[
        go.Scatter(
            x=X_pca[:, 0].tolist(),     # Convert to list
            y=X_pca[:, 1].tolist(),     # Convert to list
            marker=dict(
                color=cluster_labels.tolist(),  # Convert to list
                ...
            ),
            ...
        )
    ]
)
```

## Files Modified

1. **Source Code**: [cluster_cells_by_pattern.py](cluster_cells_by_pattern.py:426-431)
   - Added `.tolist()` calls to convert NumPy arrays to Python lists

2. **Regenerated HTML Files** (k=2 only):
   - `measurement_results/cell_clustering/figures/k2/interactive_pca_mollweide_k2.html` (1141 cells, all datasets)
   - `measurement_results/cell_clustering/figures/k2/2l/interactive_pca_mollweide_k2.html` (306 cells)
   - `measurement_results/cell_clustering/figures/k2/3m/interactive_pca_mollweide_k2.html` (587 cells)
   - `measurement_results/cell_clustering/figures/k2/3r/interactive_pca_mollweide_k2.html` (248 cells)

## Verification

After the fix, the HTML files now properly display:
- Scatter plot with all data points visible
- Correct x/y coordinates as JavaScript numbers (not NumPy metadata)
- Interactive hover functionality showing mollweide projections
- Click functionality to "pin" a projection

Example before fix:
```javascript
x: {'dtype': 'f8', 'bdata': '...'}  // Wrong!
```

Example after fix:
```javascript
x: [-1.23, -4.08, -2.32, ...]  // Correct!
```

## Next Steps

If you need to regenerate HTML files for other k values (k=3, k=4, etc.), run:

```bash
source /groups/scicompsoft/home/ackermand/miniconda3/etc/profile.d/conda.sh
conda activate plasmodesmata_analysis
python cluster_cells_by_pattern.py
```

The fixed code will now generate proper HTML files for all k values automatically.

## Scripts Created

Three helper scripts were created to fix the existing HTML files:
- `quick_fix_html.py` - Quick fix for single dataset
- `fix_all_k2_htmls.py` - Fix all k=2 dataset-specific HTMLs
- `fix_html_visualizations.py` - General purpose fixer (uses the main function)

These are one-time fix scripts and can be removed after verification.
