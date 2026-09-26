from math import *
import numpy as np
import pickle
import os

# -------------------------------
# UTILS FOR THE TIPE
# -------------------------------

# Quick sort algo
def satellitesDistanceSorting(distances):
    if len(distances) <= 0:
        return distances
    else:
        pivot = distances[len(distances) // 2]
        over = [distance for distance in distances if distance[0] > pivot[0]]
        under = [distance for distance in distances if distance[0] < pivot[0]]
        return satellitesDistanceSorting(under) + [pivot] + satellitesDistanceSorting(over)


# Astar algo for sat graph
def Astar(positions, min_index_sa, adj_sat_mat, N_sat):
    sat_path = []
    target_pos = positions[min_index_sa[1]]
    heuristic = np.sqrt(np.sum((positions - target_pos) ** 2, axis=-1))

    sat_distances = np.full(N_sat, np.inf)
    sat_distances[min_index_sa[0]] = 0
    visited = set()
    previous_list = np.full(N_sat, -1, dtype=int)

    while len(visited) < N_sat:
        sat_scores = np.where([i not in visited for i in range(N_sat)], sat_distances + heuristic, np.inf)
        # near_sat = np.where([i not in visited for i in range(N_sat)], sat_distances, np.inf)
        current = np.argmin(sat_scores)

        if current == min_index_sa[1]:
            break
        if sat_distances[current] == np.inf:
            break
        visited.add(current)

        for n in range(N_sat):
            if n in visited:
                continue
            if adj_sat_mat[current][n] == np.inf:
                continue
            candidate = sat_distances[current] + adj_sat_mat[current][n]
            if candidate < sat_distances[n]:
                sat_distances[n] = candidate
                previous_list[n] = current

    if sat_distances[min_index_sa[1]] == np.inf:
        return [], np.inf

    node = min_index_sa[1]
    while node != -1:
        sat_path.append(node)
        node = previous_list[node]
    sat_path.reverse()

    return sat_path, sat_distances

# File gestion function for data save
def writing(filepath, obj):
    file_obj = open(filepath, "wb")
    pickle.dump(obj, file_obj)

def reading(filepath):
    file_obj = open(filepath, "rb")
    obj = pickle.load(file_obj)
    return(obj)

def adding(filepath, obj, N):
    file_obj = reading(filepath)
    file_obj[N] = obj
    pickle.dump(file_obj, open(filepath, "wb"))
    return(file_obj)

def check_files(dirpath, filename):
    files = os.listdir(dirpath)
    if filename in files:
        return True
    else :
        return False