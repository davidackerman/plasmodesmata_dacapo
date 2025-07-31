#!/usr/bin/env python3
"""
Example usage script for parallel remeshing.
"""

from parallel_remesh_flexible import process_cells_parallel, analyze_results
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_parallel_processing():
    """
    Example of how to use the parallel remeshing functionality.
    """

    # Configuration
    dataset = "jrc_22ak351-leaf-3m"

    # Example 1: Process a few specific cells for testing
    logger.info("Example 1: Processing specific cells")
    test_cell_ids = [364, 390]  # Small subset for testing

    results = process_cells_parallel(
        dataset=dataset,
        cell_ids=test_cell_ids,
        max_workers=2,
        backend="multiprocessing",  # Options: "multiprocessing", "joblib", "dask", "sequential"
        use_allow_duplicates=True,
        save_results=True,
        output_dir="test_results",
    )

    # Analyze results
    summary_df = analyze_results(results, output_dir="test_results")
    print("\nSummary Statistics:")
    print(summary_df)

    # Example 2: Process all cells (commented out for safety)
    # logger.info("Example 2: Processing all cells")
    # all_results = process_cells_parallel(
    #     dataset=dataset,
    #     cell_ids=None,  # Process all cells
    #     max_workers=4,
    #     backend="multiprocessing",
    #     use_allow_duplicates=True,
    #     save_results=True,
    #     output_dir="all_results"
    # )

    # Example 3: Sequential processing for comparison
    logger.info("Example 3: Sequential processing for comparison")
    sequential_results = process_cells_parallel(
        dataset=dataset,
        cell_ids=test_cell_ids,
        backend="sequential",
        use_allow_duplicates=True,
        save_results=True,
        output_dir="sequential_results",
    )

    # Compare timing
    parallel_times = [
        r.get("timing", {}).get("total_time", 0) for r in results if r.get("success")
    ]
    sequential_times = [
        r.get("timing", {}).get("total_time", 0)
        for r in sequential_results
        if r.get("success")
    ]

    if parallel_times and sequential_times:
        avg_parallel = sum(parallel_times) / len(parallel_times)
        avg_sequential = sum(sequential_times) / len(sequential_times)
        logger.info(
            f"Average time per cell - Parallel: {avg_parallel:.2f}s, Sequential: {avg_sequential:.2f}s"
        )

    return results, summary_df


if __name__ == "__main__":
    results, summary = run_parallel_processing()
