import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import proj3d
from math import *
import ast
import re
import numpy as np

from utils import *

VALUE_BY_EXCESS_SPEED = 8e14
COVERAGE_MIN = 0.99
MIN_ELEVATION_DEG = 45

# -------------------------------
# DATA RESULT VISUALISATION
# -------------------------------

def readData(savename):
    save_raw = reading(f'./simu-saves/{savename}.ted')
    TEST_V = save_raw['simu_version']
    best_idx_list = []
    scores_list = []
    best_scores = []

    for i in range(len(save_raw)):
        result_points = save_raw[i]['result_points']

        for point in result_points:
            print(point)

        avg_pseudo_speed_ratio = [result_points[i][0][0] for i in range(len(result_points))]
        costs_for_Ns = [result_points[i][0][1] for i in range(len(result_points))]
        avg_sat_coverage = [result_points[i][3] for i in range(len(result_points))]

        excess_speed = [max(0.0, r - 1 / 300000) for r in avg_pseudo_speed_ratio]

        result_scores = [
            costs_for_Ns[i] + VALUE_BY_EXCESS_SPEED * excess_speed[i]
            for i in range(len(result_points))
        ]
        scores_list.append(result_scores)

        N_planes_list = [result_points[i][1] for i in range(len(result_points))]
        N_per_plane_list = [result_points[i][2] for i in range(len(result_points))]

        valid_values = [i for i in range(len(result_points)) if avg_sat_coverage[i] >= COVERAGE_MIN]

        if len(valid_values) == 0:
            print(f"[WARNING] No configuration with COUVERTURE_MIN={COVERAGE_MIN} on test {i}")
            best_idx = None
            best_idx_list.append(best_idx)
            best_scores.append(np.inf)
        else :
            best_idx = min(valid_values, key=lambda i: result_scores[i])
            best_idx_list.append(best_idx)
            best_scores.append(result_scores[best_idx])

    best_config_idx = np.argmin(best_scores)
    best_config_best_idx = best_idx_list[best_config_idx]
    result_points_best = save_raw[best_config_idx]

    result_scores_best = scores_list[best_config_idx]
    avg_sat_coverage_best = [result_points_best[i][3] for i in range(len(result_points_best))]
    N_planes_list_best = [result_points_best[i][1] for i in range(len(result_points_best))]
    N_per_plane_list_best = [result_points_best[i][2] for i in range(len(result_points_best))]
    H_best = result_points_best["H"]

    valid_values_best = [i for i in range(len(result_points_best)) if avg_sat_coverage_best[i] >= COVERAGE_MIN]

    print(f"\nBest config with coverage >= {COVERAGE_MIN} :")
    print(f"  N_planes={N_planes_list_best[best_config_best_idx]}, N_per_plane={N_per_plane_list_best[best_config_best_idx]}, H={H_best}, "
          f"score={result_scores_best[best_config_best_idx]:.3e}, coverage={avg_sat_coverage_best[best_config_best_idx]:.4f}")
    print(f"  ({len(valid_values_best)}/{len(result_points_best)} configuration passed the constraint)")

    return N_planes_list_best, N_per_plane_list_best, result_scores_best, avg_sat_coverage_best, TEST_V, best_config_best_idx, H_best

def displaying(N_planes_list, N_per_plane_list, result_scores, avg_sat_coverage, TEST_V, best_idx, H):
    fig = plt.figure(figsize=(8, 6))
    axes = fig.add_subplot(111, projection='3d')

    axes.scatter(N_planes_list, N_per_plane_list, avg_sat_coverage, c=result_scores, cmap='viridis')
    axes.set_xlabel('N Planes')
    axes.set_ylabel('N per Planes')
    axes.set_zlabel('Coverage')
    plt.suptitle(f'Constellation Coverage with type : Walker', fontsize=16, fontweight='bold')
    plt.title(f'MIN_ELEVATION_DEG = {MIN_ELEVATION_DEG}°, H = {H}', fontsize=10)

    plt.tight_layout()
    fig.subplots_adjust(right=0.85)
    plt.savefig(f"./pseudo-latency-result/coverage_Nplanes_Nperplane_v{TEST_V}.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.clf()

    fig = plt.figure(figsize=(8, 6))
    axes = fig.add_subplot(111, projection='3d')

    axes.scatter(N_planes_list, N_per_plane_list, result_scores, c=result_scores, cmap='viridis')
    if best_idx is not None:
        axes.scatter([N_planes_list[best_idx]], [N_per_plane_list[best_idx]], [result_scores[best_idx]],
                     c='red', s=120, marker='*', label=f'Optimum ({N_planes_list[best_idx]}, {N_per_plane_list[best_idx]})')
        axes.legend()
    axes.set_xlabel('N Planes')
    axes.set_ylabel('N per Planes')
    axes.set_zlabel('Score')
    plt.suptitle(f'Constellation Score with type : Walker', fontsize=16, fontweight='bold')
    plt.title(f'MIN_ELEVATION_DEG = {MIN_ELEVATION_DEG}°', fontsize=10)

    plt.tight_layout()
    fig.subplots_adjust(right=0.85)
    plt.savefig(f"./pseudo-latency-result/score_Nplanes_Nperplane_v{TEST_V}.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.clf()

N_planes_list, N_per_plane_list, result_scores, avg_sat_coverage, TEST_V, best_idx, H = readData('simu_score_Nplanes_Nperplane_v6.2')
displaying(N_planes_list, N_per_plane_list, result_scores, avg_sat_coverage, TEST_V, best_idx, H)