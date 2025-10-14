"""
Simplified map loading utilities for DeepTactics.
Based on linesight but with minimal dependencies.
"""

import numpy as np
import numpy.typing as npt
from pathlib import Path


def load_zone_centers(zone_centers_filename: str, maps_dir: Path,
                      n_extrapolate_before: int = 20,
                      n_extrapolate_after: int = 1000):
    """
    Load a map.npy file containing zone centers (virtual checkpoints).
    Extrapolates additional points before start and after finish.

    Args:
        zone_centers_filename: Name of the .npy file with zone centers
        maps_dir: Directory containing map files
        n_extrapolate_before: Number of points to add before start
        n_extrapolate_after: Number of points to add after finish

    Returns:
        zone_centers: Array of shape (n_points, 3) with 3D coordinates
    """
    zone_centers = np.load(str(maps_dir / zone_centers_filename))

    # Add extrapolated points before start
    before_start = (
        zone_centers[0]
        + np.expand_dims(zone_centers[0] - zone_centers[1], axis=0)
        * np.expand_dims(np.arange(n_extrapolate_before, 0, -1), axis=1)
    )

    # Add extrapolated points after finish
    after_finish = (
        zone_centers[-1]
        + np.expand_dims(zone_centers[-1] - zone_centers[-2], axis=0)
        * np.expand_dims(np.arange(1, 1 + n_extrapolate_after, 1), axis=1)
    )

    zone_centers = np.vstack((before_start, zone_centers, after_finish))

    # Smoothen the trajectory (skip first/last 5 points)
    zone_centers[5:-5] = 0.5 * (zone_centers[:-10] + zone_centers[10:])

    return zone_centers


def precalculate_zone_info(zone_centers: npt.NDArray):
    """
    Precalculate useful information about zones for fast access during rollouts.

    Args:
        zone_centers: Array of shape (n_points, 3)

    Returns:
        zone_transitions: Midpoints between consecutive zone centers
        distance_between_transitions: Distance between consecutive transitions
        cumulative_distance: Cumulative distance from start to each transition
        normalized_vectors: Unit vectors along track direction
    """
    # Transitions are midpoints between zone centers
    zone_transitions = 0.5 * (zone_centers[1:] + zone_centers[:-1])

    # Calculate distances
    delta = zone_transitions[1:] - zone_transitions[:-1]
    distance_between = np.linalg.norm(delta, axis=1)
    cumulative_distance = np.hstack((0, np.cumsum(distance_between)))

    # Normalized direction vectors
    normalized_vectors = delta / np.expand_dims(distance_between, axis=-1)

    return zone_transitions, distance_between, cumulative_distance, normalized_vectors
