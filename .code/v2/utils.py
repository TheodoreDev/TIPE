from math import *
import numpy as np


def satellitesDistanceSorting(distances):
    if len(distances) <= 0:
        return distances
    else:
        pivot = distances[len(distances) // 2]
        over = [distance for distance in distances if distance[0] > pivot[0]]
        under = [distance for distance in distances if distance[0] < pivot[0]]
        return satellitesDistanceSorting(under) + [pivot] + satellitesDistanceSorting(over)