#!/bin/bash
# Setup script for parallel remeshing

echo "Setting up parallel remeshing environment..."

# Install Python dependencies
echo "Installing Python dependencies..."
pip install pandas numpy trimesh tqdm
pip install pygeodesic

# Optional dependencies for different backends
echo "Installing optional parallel processing libraries..."
pip install joblib  # For joblib backend
# pip install 'dask[complete]'  # Uncomment for Dask backend (larger installation)

echo "Setup complete!"
echo ""
echo "To run the example:"
echo "python example_usage.py"
echo ""
echo "To run with specific parameters:"
echo "python parallel_remesh_flexible.py"
