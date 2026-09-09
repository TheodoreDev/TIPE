import pygame
import numpy as np
import random
import sys
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
from math import *

from utils import *

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

    plt.scatter(costs_for_Ns, avg_pseudo_speed_ratio)
    plt.xlabel('cost')
    plt.ylabel('Inv speed')
    plt.savefig(f"./pseudo-latency-result/pareto_front_v{TEST_V}.png", dpi=150, bbox_inches="tight")
    plt.show()

    resulty = [
        0.7 * (avg_pseudo_speed_ratio[i] / max(avg_pseudo_speed_ratio)) + 0.3 * (costs_for_Ns[i] / max(costs_for_Ns))
        for i in range(len(costs_for_Ns))
    ]
    plt.scatter([i + 1 for i in range(N_planes - len(avg_pseudo_speed_ratio), N_planes)], resulty)
    plt.ylabel('Score')
    plt.xlabel('N planes')
    plt.savefig(f"./pseudo-latency-result/inv_speed_time_cost_v{TEST_V}.png", dpi=150, bbox_inches="tight")

    # Saving data on external file (binary .ted)
    data = {
        "simu_version" : TEST_V ,
        "avg_pseudo_speed_ratio" : avg_pseudo_speed_ratio,
        "costs_for_Ns" : costs_for_Ns,
        "result_score" : resulty,
        "N_planes_f" : N_planes,
    }
    writing(f"./simu-saves/simu_score_Nplanes_v{TEST_V}.ted", data)

    plt.show()