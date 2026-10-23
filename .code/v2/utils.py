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
        if sat_scores[current] == np.inf:
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

def getLagrangeInterpolator(fx, fy, smooth, xstart, xend):
    def getLagrange(i, x):
        L = 1
        for j in range(len(fx)):
            if j != i:
                L *= (x-fx[j])/(fx[i]-fx[j])
        return L

    xs = [xstart + i*(xend-xstart)/(smooth-1) for i in range(smooth)]
    ys = []
    for x in xs:
        P = 0
        for j in range(len(fx)):
            P += fy[j]*getLagrange(j, x)
        ys.append(P)
    return xs, ys

def getPCHIPInterpolator(fx, fy, smooth):
    n = len(fx)
    if n < 2:
        return fx, fy

    h = [fx[i+1] - fx[i] for i in range(n - 1)]         # x step list btw two points
    delta = [(fy[i+1] - fy[i]) / h[i] for i in range(n - 1)]    # slope list btw two points

    d = [0.0]*n
    # Fritsch-Carlson formula to keep monotony
    for i in range(1, n - 1):
        # Blocking oscillation if 2 adjacent slopes are opposite sign
        if delta[i-1] * delta[i] <= 0:
            d[i] = 0.0
        # weighted average favoring the least steep slope
        else:
            w1 = 2*h[i] + h[i-1]
            w2 = h[i] + 2*h[i-1]
            d[i] = (w1 + w2) / (w1/delta[i-1] + w2/delta[i])

    # Extremity slopes
    d[0] = delta[0] if (n > 1 and delta[0] != 0) else 0.0
    d[-1] = delta[-1] if (n > 1 and delta[-1] != 0) else 0.0

    # Interpolation coo
    step = (fx[-1] - fx[0]) / (smooth-1)
    xs = [fx[0] + i*step for i in range(smooth)]
    ys = []

    for x in xs:
        if x <= fx[0]:
            ys.append(fy[0])
            continue
        if x >= fx[-1]:
            ys.append(fy[-1])
            continue

        k = 0
        while k < n-2 and x > fx[k+1]:
            k += 1

        t = (x - fx[k])/h[k]
        h00 = 2*t**3 - 3*t**2 + 1
        h10 = t**3 - 2*t**2 + t
        h01 = -2*t**3 + 3*t**2
        h11 = t**3 - t**2

        ys.append(fy[k]*h00 + h[k]*d[k]*h10 + fy[k+1]*h01 + h[k]*d[k+1]*h11)

    return xs, ys