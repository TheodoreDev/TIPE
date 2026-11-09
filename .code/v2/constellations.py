import pygame
import numpy as np
import random
import sys
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
from math import *

from satellites import Satellite


# Constellation catalog ([N_planes, N_per_plane, H])
SATCONST_CATALOG = {
  "OneWeb" : [12, 49, 1200, 87.9],
  "Iridium" : [6, 11, 780, 86.4],
}


# -------------------------------
# LOAD/CREATE THE SATELLITE CONSTELLATION
# -------------------------------

def generateWalker(N_planes, N_per_plane, orbital_params):
  N_PLANES = N_planes
  N_PER_PLANE = N_per_plane  # N_planes * N_per_plane = number of sat in the constellation
  INCLINATION = orbital_params["INCLINATION"]

  satellites = []
  for p in range(N_PLANES):
      raan = p * (180.0 / N_PLANES)  # Spaced plan of de 180°/12 = 15°
      for s in range(N_PER_PLANE):
          # Shift between two plans (F parametre of Walker)
          walker_offset = (p * 360.0 / N_PER_PLANE) / N_PLANES
          phase = s * (360.0 / N_PER_PLANE) + walker_offset + p*25
          satellites.append(Satellite(
              inclinaison=INCLINATION,
              raan=raan,
              altitude_km=orbital_params['H'],
              phase=np.radians(phase)
          ))
  return satellites
  

  def generateRandom():
    pass
  

  def generateFromCatalog():
    pass