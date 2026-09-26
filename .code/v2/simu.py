import pygame
import numpy as np
import random
import sys
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import proj3d
import json
from math import *

from utils import *


# -------------------------------
# SIMULATION DATA SAVING AND CALCULATION
# -------------------------------
# One function for each type of experiences

def simuScoreNplanes(avg_pseudo_speed_ratio, costs_for_Ns,  N_planes, TEST_V):
    print(avg_pseudo_speed_ratio, costs_for_Ns)

    # Create different graph for visualizing data
    fig, ax1 = plt.subplots()
    ax1.scatter([i + 1 for i in range(N_planes - len(avg_pseudo_speed_ratio), N_planes)], avg_pseudo_speed_ratio,
                color='blue')
    ax1.set_ylabel('Inv Speed', color='blue')
    ax1.set_xlabel('N planes')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax2 = ax1.twinx()
    ax2.scatter([i + 1 for i in range(N_planes - len(avg_pseudo_speed_ratio), N_planes)], costs_for_Ns, color='red')
    ax2.set_ylabel('Cost', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    plt.savefig(f"./pseudo-latency-result/inv_speed_and_cost_v{TEST_V}.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.clf()

    plt.scatter(costs_for_Ns, avg_pseudo_speed_ratio)
    plt.xlabel('cost')
    plt.ylabel('Inv speed')
    plt.savefig(f"./pseudo-latency-result/pareto_front_v{TEST_V}.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.clf()

    resulty = [
        0.7 * (avg_pseudo_speed_ratio[i] / max(avg_pseudo_speed_ratio)) + 0.3 * (costs_for_Ns[i] / max(costs_for_Ns))
        for i in range(len(costs_for_Ns))
    ]
    plt.scatter([i + 1 for i in range(N_planes - len(avg_pseudo_speed_ratio), N_planes)], resulty)
    plt.ylabel('Score')
    plt.xlabel('N planes')
    plt.savefig(f"./pseudo-latency-result/score_Nplanes_v{TEST_V}.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.clf()

    # Saving data on external file (binary .ted)
    data = {
        "simu_version": TEST_V,
        "avg_pseudo_speed_ratio": avg_pseudo_speed_ratio,
        "costs_for_Ns": costs_for_Ns,
        "result_score": resulty,
        "N_planes_f": N_planes,
    }
    writing(f"./simu-saves/simu_score_Nplanes_v{TEST_V}.ted", data)


def simuScoreNplanesNperplane(result_points, TEST_V, MIN_ELEVATION_DEG, VALUE_BY_EXCESS_SPEED):
    print(result_points)
    avg_pseudo_speed_ratio = [result_points[i][0][0] for i in range(len(result_points))]
    costs_for_Ns = [result_points[i][0][1] for i in range(len(result_points))]
    avg_sat_coverage = [result_points[i][3] for i in range(len(result_points))]

    excess_speed = [max(0.0, r-1/300000) for r in avg_pseudo_speed_ratio]

    result_scores = [
        costs_for_Ns[i] + VALUE_BY_EXCESS_SPEED * excess_speed[i]
        for i in range(len(result_points))
    ]
    N_planes_list = [result_points[i][1] for i in range(len(result_points))]
    N_per_plane_list = [result_points[i][2] for i in range(len(result_points))]

    # Saving data on external file (binary .ted)
    data = {
        "simu_version": TEST_V,
        "MIN_ELEVATION_DEG": MIN_ELEVATION_DEG,
        "VALUE_BY_EXCESS_SPEED": VALUE_BY_EXCESS_SPEED,
        "avg_pseudo_speed_ratio": avg_pseudo_speed_ratio,
        "costs_for_Ns": costs_for_Ns,
        "N_planes_f": N_planes_list,
        "N_per_plane_list" : N_per_plane_list,
        "raw_data" : result_points,
    }

    writing(f"./simu-saves/simu_score_Nplanes_Nperplane_v{TEST_V}.ted", data)
    print("[SUCCESS] simulation saved successfully")

def simuScoreNplanesNperplaneH(result_points, TEST_V, MIN_ELEVATION_DEG, VALUE_BY_EXCESS_SPEED, H, N_hsimu):
    if check_files("./simu-saves/", f"simu_score_Nplanes_Nperplane_h_v{TEST_V}.ted"):
        writing(f"./simu-saves/simu_score_Nplanes_Nperplane_h_v{TEST_V}.ted", {"simu_version": TEST_V})
    #print(result_points)

    avg_pseudo_speed_ratio = [result_points[i][0][0] for i in range(len(result_points))]
    costs_for_Ns = [result_points[i][0][1] for i in range(len(result_points))]
    avg_sat_coverage = [result_points[i][3] for i in range(len(result_points))]

    excess_speed = [max(0.0, r - 1 / 300000) for r in avg_pseudo_speed_ratio]

    result_scores = [
        costs_for_Ns[i] + VALUE_BY_EXCESS_SPEED * excess_speed[i]
        for i in range(len(result_points))
    ]
    N_planes_list = [result_points[i][1] for i in range(len(result_points))]
    N_per_plane_list = [result_points[i][2] for i in range(len(result_points))]

    # Saving data on external file (binary .ted)
    data = {
        "simu_version": TEST_V,
        "MIN_ELEVATION_DEG": MIN_ELEVATION_DEG,
        "VALUE_BY_EXCESS_SPEED": VALUE_BY_EXCESS_SPEED,
        "H" : H,
        "avg_pseudo_speed_ratio": avg_pseudo_speed_ratio,
        "costs_for_Ns": costs_for_Ns,
        "N_planes_f": N_planes_list,
        "N_per_plane_list": N_per_plane_list,
        "raw_data": result_points,
    }

    adding(f"./simu-saves/simu_score_Nplanes_Nperplane_h_v{TEST_V}.ted", data, N_hsimu)
    print(f"[SUCCESS] simulation {N_hsimu} saved successfully, start processing next ...")