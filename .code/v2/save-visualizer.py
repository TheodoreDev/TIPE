#from idlelib.browser import transform_children

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from IPython.core.pylabtools import figsize
from matplotlib.lines import Line2D
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

def scoreCalc(cost, inv_speed):
    if not np.isfinite(inv_speed) or not np.isfinite(cost):
        return np.inf
    return cost + VALUE_BY_EXCESS_SPEED * max(0.0, inv_speed - 1/300000)

def paretoFront(inv_speeds, cost, index):
    front, best_inv_speed = [], inf
    for i in sorted(index, key=lambda i: (cost[i], inv_speeds[i])):
        if inv_speeds[i] < best_inv_speed:
            front.append(i)
            best_inv_speed = inv_speeds[i]
    return front

def readData(savename):
    save_raw = reading(f'./simu-saves/{savename}.ted')
    print(save_raw)
    print((len(save_raw)-1))
    TEST_V = save_raw['simu_version']

    h_keys = sorted(k for k in save_raw if k != 'simu_version')
    print(f"{len(h_keys)} altitude(s) in {savename}")

    best_idx_list = []
    scores_list = []
    best_scores = []
    per_h = []

    global MIN_ELEVATION_DEG
    MIN_ELEVATION_DEG = save_raw[h_keys[0]]['MIN_ELEVATION_DEG']

    for i in h_keys:
        result_points = save_raw[i]['raw_data']

        for point in result_points:
            print(point)

        avg_pseudo_speed_ratio = [result_points[j][0][0] for j in range(len(result_points))]
        costs_for_Ns = [result_points[j][0][1] for j in range(len(result_points))]
        avg_sat_coverage = [result_points[j][3] for j in range(len(result_points))]
        N_planes_list = [result_points[j][1] for j in range(len(result_points))]
        N_per_plane_list = [result_points[j][2] for j in range(len(result_points))]

        excess_speed = [max(0.0, r - 1 / 300000) for r in avg_pseudo_speed_ratio]

        result_scores = [
            scoreCalc(costs_for_Ns[j], avg_pseudo_speed_ratio[j])
            for j in range(len(result_points))
        ]
        scores_list.append(result_scores)

        valid_values = [i for i in range(len(result_points))
                        if avg_sat_coverage[i] >= COVERAGE_MIN and np.isfinite(result_scores[i])]

        if len(valid_values) == 0:
            print(f"[WARNING] No configuration with COUVERTURE_MIN={COVERAGE_MIN} on test {i}")
            best_idx = None
            best_idx_list.append(best_idx)
            best_scores.append(np.inf)
        else :
            best_idx = min(valid_values, key=lambda i: result_scores[i])
            best_idx_list.append(best_idx)
            best_scores.append(result_scores[best_idx])

        per_h.append({
            'H': save_raw[i]['H'],
            'inv_speed': avg_pseudo_speed_ratio, 'cost': costs_for_Ns, 'cov': avg_sat_coverage,
            'score': result_scores, 'N_planes': N_planes_list, 'N_per_plane': N_per_plane_list,
            'valid': valid_values, 'best': best_idx,
        })

    pos = int(np.argmin(best_scores))
    best_config_idx = h_keys[pos]
    best_config_best_idx = best_idx_list[pos]

    if best_config_best_idx is None:
        print(f"[ERROR] No configuration reaches coverage >= {COVERAGE_MIN} on any altitude")
        displayingParetoPerH(per_h, TEST_V)

    result_points_best = save_raw[best_config_idx]['raw_data']

    result_scores_best = scores_list[pos]
    avg_sat_coverage_best = [result_points_best[i][3] for i in range(len(result_points_best))]
    N_planes_list_best = [result_points_best[i][1] for i in range(len(result_points_best))]
    N_per_plane_list_best = [result_points_best[i][2] for i in range(len(result_points_best))]
    H_best = save_raw[best_config_idx]['H']

    valid_values_best = [i for i in range(len(result_points_best)) if avg_sat_coverage_best[i] >= COVERAGE_MIN]

    print(f"\nBest config with coverage >= {COVERAGE_MIN} (over {len(h_keys)} simulations) :")
    print(f"  N_planes={N_planes_list_best[best_config_best_idx]}, N_per_plane={N_per_plane_list_best[best_config_best_idx]}, H={H_best}, "
          f"score={result_scores_best[best_config_best_idx]:.3e}, coverage={avg_sat_coverage_best[best_config_best_idx]:.4f}")
    print(f"  ({len(valid_values_best)}/{len(result_points_best)} configuration passed the constraint)")

    print(f"\n{'H':>7} {'N_planes':>9} {'N_per_plane':>12} {'cost (M)':>10} {'inv speed':>11} {'coverage':>9} {'score':>10} {'valid/total':>12}")
    for d in per_h:
        b = d['best']
        counts = f"{len(d['valid'])}/{len(d['cost'])}"
        if b is None:
            print(f"{d['H']:>7} {'-':>9} {'-':>12} {'-':>10} {'-':>11} {'-':>9} {'-':>10} {counts:>12}")
        else:
            print(f"{d['H']:>7} {d['N_planes'][b]:>9} {d['N_per_plane'][b]:>12} {d['cost'][b] / 1e6:>10.1f} "
                  f"{d['inv_speed'][b]:>11.3e} {d['cov'][b]:>9.4f} {d['score'][b]:>10.3e} {counts:>12}")

    displayingParetoPerH(per_h, TEST_V)

    MIN_ELEVATION_DEG = save_raw[best_config_idx]['MIN_ELEVATION_DEG']

    return N_planes_list_best, N_per_plane_list_best, result_scores_best, avg_sat_coverage_best, TEST_V, best_config_best_idx, H_best

def displayingParetoPerH(per_h, TEST_V):
    n = len(per_h)
    n_col = ceil(sqrt(n))
    n_row = int(ceil(n / n_col))
    fig, axes = plt.subplots(n_row, n_col, figsize=(5.5*n_col, 4.5*n_row), sharex=True, sharey=True, squeeze=False)
    axes = axes.ravel()

    scored = [(i, d['score'][d['best']]) for i, d in enumerate(per_h) if d['best'] is not None]
    global_best = min(scored, key=lambda t: t[1])[0] if scored else None

    for i, (axe, d) in enumerate(zip(axes, per_h)):
        finite_values = [i for i in range(len(d['cost'])) if np.isfinite(d['inv_speed'][i]) and np.isfinite(d['cost'][i])]
        excluded = [i for i in finite_values if d['cov'][i] < COVERAGE_MIN]
        front = paretoFront(d['inv_speed'], d['cost'], d['valid'])
        in_front = set(front)
        others = [i for i in d['valid'] if i not in in_front]

        def xy(index):
            return [d['cost'][i] / 1e6 for i in index], [d['inv_speed'][i] for i in index]

        axe.scatter(*xy(excluded), c='lightgray', s=18, zorder=1)
        axe.scatter(*xy(others), c='tab:blue', s=28, alpha=0.5, zorder=2)
        if front:
            fx, fy = xy(front)
            axe.step(fx, fy, where='post', color='tab:orange', lw=1.5, zorder=3)
            axe.scatter(fx, fy, c='tab:orange', s=40, zorder=4)

        b = d['best']
        if b is not None :
            axe.scatter([d['cost'][b] / 1e6], [d['inv_speed'][b]], marker='*', s=260, c='red', edgecolor='black', zorder=5)
            axe.annotate(f"({d['N_planes'][b]}, {d['N_per_plane'][b]})", xy=(d['cost'][b] / 1e6, d['inv_speed'][b]), xytext=(10, 10),
                         textcoords='offset points', fontsize=9, fontweight='bold', zorder=5)
        else :
            axe.text(0.5, 0.5, f'no configuration with\ncoverage >= {COVERAGE_MIN}', transform=axe.transAxes, ha='center', va='center', color='red')

        title = f"H = {d['H']} ({len(d['valid'])}/{len(finite_values)} valid)"
        if i == global_best:
            axe.set_title('* ' + title + ' - global best', fontsize=10, fontweight='bold', color='black')
        else:
            axe.set_title(title, fontsize=10, color='black')
        axe.grid(alpha=0.3)

    for axe, d in zip(axes, per_h):
        front = paretoFront(d['inv_speed'], d['cost'], d['valid'])

        def xy(index):
            return [d['cost'][i] / 1e6 for i in index], [d['inv_speed'][i] for i in index]

        if front:
            fx, fy = xy(front)
            axe.autoscale(False)

            fx, fy = np.array(fx), np.array(fy)
            fx_u, index_u = np.unique(fx, return_index=True)
            fy_u = fy[index_u]

            if len(fx_u) > 1:
                il_x, il_y = getPCHIPInterpolator(fx_u, fy_u, 100)
                axe.plot(il_x, il_y, color='red', linestyle='--', lw=2.5, zorder=5, scalex=False, scaley=False)

    for axe in axes[n:]:
        axe.axis('off')
    for axe in axes[:n]:
        axe.set_xlabel('Cost')
        axe.set_ylabel('Inverse Speed')

    handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray', markersize=8,
               label=f'coverage < {COVERAGE_MIN} (excluded)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:blue', markersize=8,
               label='valid, dominated'),
        Line2D([0], [0], marker='o', color='tab:orange', markersize=8, label='Pareto front (valid'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='red', markeredgecolor='black', markersize=15,
               label='best score'),
        Line2D([0], [0], color='red', linestyle='--', lw=2, label='Lagrange Interpolation'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=10)
    fig.suptitle(f'Constellation Pareto Front with type : Walker - MIN_ELEVATION_DEG = {MIN_ELEVATION_DEG}°', fontsize=16, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f"./result-curves/pareto_Nplanes_Nperplane_h_v{TEST_V}.png", dpi=150, bbox_inches="tight")
    plt.show()

def displaying(N_planes_list, N_per_plane_list, result_scores, avg_sat_coverage, TEST_V, best_idx, H):
    fig = plt.figure(figsize=(8, 6))
    axes = fig.add_subplot(111, projection='3d')
    print(avg_sat_coverage)
    axes.scatter(N_planes_list, N_per_plane_list, avg_sat_coverage, c=result_scores, cmap='viridis')
    axes.set_xlabel('N Planes')
    axes.set_ylabel('N per Planes')
    axes.set_zlabel('Coverage')
    plt.suptitle(f'Constellation Coverage with type : Walker', fontsize=16, fontweight='bold')
    plt.title(f'MIN_ELEVATION_DEG = {MIN_ELEVATION_DEG}°, H = {H}', fontsize=10)

    plt.tight_layout()
    fig.subplots_adjust(right=0.85)
    #plt.savefig(f"./result-curves/coverage_Nplanes_Nperplane_h_v{TEST_V}.png", dpi=150, bbox_inches="tight")
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
    plt.title(f'MIN_ELEVATION_DEG = {MIN_ELEVATION_DEG}°, H = {H}', fontsize=10)

    plt.tight_layout()
    fig.subplots_adjust(right=0.85)
    #plt.savefig(f"./result-curves/score_Nplanes_Nperplane_h_v{TEST_V}.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.clf()

N_planes_list, N_per_plane_list, result_scores, avg_sat_coverage, TEST_V, best_idx, H = readData('simu_score_Nplanes_Nperplane_h_v7.0')
displaying(N_planes_list, N_per_plane_list, result_scores, avg_sat_coverage, TEST_V, best_idx, H)