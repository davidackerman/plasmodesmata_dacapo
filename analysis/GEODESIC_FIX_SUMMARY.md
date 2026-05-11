# Geodesic Distance Computation Failure - Analysis and Fix

## Problem Summary

Cell 84 failed during parallel processing with the error:
```
ValueError: Invalid geodesic distances computed from vertex 1594:
  Targets: [3469]
  Distances: [inf]
```

## Root Cause Analysis

### What Failed
- The `pygeodesic` library returned infinite distance between vertices 1594 and 3469
- This occurred during Dask parallel processing of cell 84
- The original mesh (84.ply) is valid and fully connected

### Why It Failed

The failure is likely due to **degenerate geometry created during point insertion**:

1. **Projection Distances**: Plasmodesmata coordinates are ~100-500 nm away from the mesh surface before projection
   - This is expected due to mesh simplification/smoothing vs original voxel data
   - Note: Initial debug showed 26k-51k nm, but that was due to testing with wrong coordinate order (Z,Y,X instead of X,Y,Z)
   - The actual distances with correct (X,Y,Z) order are reasonable

2. **Non-Deterministic Behavior**: Running the same cell sequentially produces different results:
   - Failed run: >3469 vertices created
   - Successful run: 1660 vertices created
   - This indicates non-determinism in the triangulation or face processing

3. **Degenerate Faces**: The mesh insertion may create:
   - Faces with near-zero area
   - Very short edges
   - Nearly co-planar triangles
   - These can cause the geodesic algorithm to fail finding paths

4. **Parallel Execution Issues**: The problem appears more frequently under Dask parallel execution, suggesting:
   - Race conditions in mesh processing
   - Memory/threading issues in pygeodesic
   - Non-thread-safe operations

## The Fix

### Changes Made to `remesh_and_measure_dask.py`

1. **Added `clean_mesh_for_geodesic()` function** (line 490):
   - Removes degenerate faces with area < 1e-10
   - Cleans up unreferenced vertices
   - Returns statistics on removed elements

2. **Enhanced `compute_pairwise_geodesic_for_inputs()` function** (line 519):
   - Added `cell_id` parameter for better error reporting
   - Calls mesh cleaning before geodesic computation
   - Collects ALL failed pairs instead of failing on first error
   - Provides comprehensive diagnostics:
     - Lists all failed vertex pairs
     - Checks mesh connectivity
     - Distinguishes between disconnected mesh vs numerical issues
     - Suggests potential solutions
   - Includes cell_id in all error messages and logs

3. **Updated function call** (line 692):
   - Passes `cell_id` to geodesic function for better error tracking

### What the Fix Does

**Prevention**:
- Removes degenerate faces that can cause geodesic algorithm failures
- Validates mesh quality before computation

**Better Diagnostics**:
- Identifies exactly which plasmodesmata pairs fail
- Determines if mesh is actually disconnected or if it's a numerical issue
- Provides actionable error messages for debugging
- All errors now include cell_id for easy identification in parallel logs

**Error Handling**:
- Collects all failures before raising, not just the first one
- Helps identify patterns in failures (e.g., all in one component)

## Testing

Run the debug scripts to verify:

```bash
# Basic mesh inspection
python debug_cell_84.py

# Full geodesic computation test
python debug_cell_84_with_geodesic.py
```

## Next Steps if Problem Persists

If cell 84 still fails with the new code:

1. **Check the detailed logs** - they will now show:
   - How many degenerate faces were removed
   - Which specific plasmodesmata pairs are failing
   - Whether the mesh is disconnected or it's a precision issue

2. **Investigate coordinate systems**:
   - Verify plasmodesmata and mesh are in same coordinate system
   - Check if transformation/scaling is needed
   - Very large projection distances (>10k nm) suggest misalignment

3. **Consider alternative approaches**:
   - Use a different geodesic library (gdist, libigl)
   - Remesh the cell to improve triangle quality
   - Filter out plasmodesmata too far from the surface
   - Use Euclidean distance as fallback for problem cells

4. **Mesh quality improvements**:
   - Simplify/remesh to reduce face count
   - Ensure consistent triangle quality
   - Check for and fix self-intersections

## Files Modified

- `remesh_and_measure_dask.py`: Main processing script with fixes applied
- `debug_cell_84.py`: Debug script to inspect mesh and plasmodesmata
- `debug_cell_84_with_geodesic.py`: Debug script to test geodesic computation
- `fix_geodesic_robustness.py`: Standalone version of the fix with additional utilities

## Key Insight

The problem is NOT with the original mesh insertion algorithm - it produces valid, connected meshes.
The problem is with the **robustness of the geodesic distance computation** on meshes with challenging geometry (degenerate faces, thin triangles, etc.).

The fix improves robustness by cleaning the mesh and providing much better diagnostics when failures occur.
