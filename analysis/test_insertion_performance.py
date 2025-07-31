#!/usr/bin/env python3
"""
Performance comparison script for different mesh insertion methods.
"""

import numpy as np
import trimesh
import time
import pandas as pd
from remesh import insert_points_into_mesh_original, insert_points_allow_duplicates
from optimized_insertion import (
    insert_points_into_mesh_optimized_v1,
    insert_points_into_mesh_optimized_v2,
    insert_points_into_mesh_optimized_v3,
    insert_points_into_mesh_fastest,
    benchmark_insertion_methods,
)


def create_test_mesh_and_points(n_points=50):
    """
    Create a simple test mesh and random points for benchmarking.
    """
    # Create a simple sphere mesh
    mesh = trimesh.creation.icosphere(subdivisions=2)

    # Generate random points near the surface
    # Sample points on the surface and add small noise
    surface_points, _ = trimesh.sample.sample_surface(mesh, n_points)

    # Add small random noise to make points slightly off-surface
    noise = np.random.normal(0, 0.01, surface_points.shape)
    test_points = surface_points + noise

    return mesh, test_points


def run_performance_test():
    """
    Run a comprehensive performance test.
    """
    print("Creating test data...")

    # Test with different point counts
    point_counts = [10, 25, 50, 100]

    all_results = []

    for n_points in point_counts:
        print(f"\n{'='*60}")
        print(f"Testing with {n_points} points")
        print(f"{'='*60}")

        mesh, test_points = create_test_mesh_and_points(n_points)

        print(f"Mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        print(f"Test points: {len(test_points)}")

        # Run benchmark
        results = benchmark_insertion_methods(mesh, test_points, max_points=n_points)

        # Add metadata to results
        for method, result in results.items():
            if result["success"]:
                result["n_points"] = n_points
                result["method"] = method
                result["points_per_second"] = (
                    n_points / result["time"] if result["time"] > 0 else float("inf")
                )
                all_results.append(result)

    # Create summary DataFrame
    if all_results:
        df = pd.DataFrame(all_results)
        print(f"\n{'='*60}")
        print("PERFORMANCE SUMMARY")
        print(f"{'='*60}")

        # Group by method and show average performance
        summary = (
            df.groupby("method")
            .agg({"time": "mean", "points_per_second": "mean", "n_points": "count"})
            .round(3)
        )

        print("\nAverage performance by method:")
        print(summary)

        # Show speedup relative to original
        if "original" in summary.index:
            original_time = summary.loc["original", "time"]
            summary["speedup_vs_original"] = (original_time / summary["time"]).round(1)
            print(f"\nSpeedup relative to original method:")
            for method in summary.index:
                if method != "original":
                    speedup = summary.loc[method, "speedup_vs_original"]
                    print(f"  {method}: {speedup}x faster")

    return all_results


def test_correctness():
    """
    Test that all methods produce similar results.
    """
    print("Testing correctness...")

    mesh, test_points = create_test_mesh_and_points(10)

    methods = {
        "original": insert_points_into_mesh_original,
        "optimized_v2": insert_points_into_mesh_optimized_v2,
        "fastest": insert_points_into_mesh_fastest,
    }

    results = {}

    for name, method in methods.items():
        try:
            test_mesh = mesh.copy()
            vertices, faces = method(test_mesh, test_points)
            results[name] = (vertices, faces)
            print(f"{name}: {len(vertices)} vertices, {len(faces)} faces")
        except Exception as e:
            print(f"{name}: ERROR - {e}")

    # Check if results are similar
    if len(results) > 1:
        vertex_counts = [len(v) for v, f in results.values()]
        face_counts = [len(f) for v, f in results.values()]

        if len(set(vertex_counts)) == 1 and len(set(face_counts)) == 1:
            print("✓ All methods produce the same number of vertices and faces")
        else:
            print("⚠ Methods produce different results:")
            for name, (v, f) in results.items():
                print(f"  {name}: {len(v)} vertices, {len(f)} faces")


def main():
    """
    Main function to run all tests.
    """
    print("Mesh Point Insertion Performance Test")
    print("=" * 50)

    # Test correctness first
    test_correctness()

    print("\n")

    # Run performance tests
    run_performance_test()

    print("\n" + "=" * 60)
    print("RECOMMENDATIONS:")
    print("=" * 60)
    print("1. For maximum speed: use 'fastest' version")
    print("2. For robustness: use 'optimized_v3' (insert_points_allow_duplicates)")
    print("3. For backward compatibility: use 'optimized_v2'")
    print("4. The original method is the slowest due to mesh recreation")


if __name__ == "__main__":
    main()
